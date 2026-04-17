from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models import SmallCRNNVAD
from utils_audio import (
    TARGET_SR,
    compute_log_mel,
    load_audio,
    load_jsonl,
    make_balanced_clip,
    make_frame_labels,
    set_seed,
)


@dataclass
class AudioConfig:
    clip_seconds: float = 2.0
    frame_len_ms: float = 25.0
    hop_len_ms: float = 10.0
    n_mels: int = 40
    snr_choices: tuple = (20.0, 10.0, 0.0)
    speech_probability: float = 0.5


class BalancedSyntheticVADDataset(Dataset):
    def __init__(self, manifest_path: str, audio_cfg: AudioConfig):
        self.rows = load_jsonl(manifest_path)
        self.audio_cfg = audio_cfg
        self.clip_samples = int(TARGET_SR * audio_cfg.clip_seconds)
        self.frame_len = int(TARGET_SR * audio_cfg.frame_len_ms / 1000.0)
        self.hop_len = int(TARGET_SR * audio_cfg.hop_len_ms / 1000.0)

    def __len__(self) -> int:
        return len(self.rows)

    def build_example(self, index: int, snr_db: float | None = None, include_speech: bool | None = None) -> dict:
        row = self.rows[index]
        speech_wave = load_audio(row["speech_path"])
        noise_path = random.choice(row["noise_pool"]) if row.get("noise_pool") else None
        noise_wave = load_audio(noise_path) if noise_path else None

        if include_speech is None:
            include_speech = random.random() < self.audio_cfg.speech_probability
        if snr_db is None and include_speech:
            snr_db = random.choice(self.audio_cfg.snr_choices)

        waveform, speech_range, meta = make_balanced_clip(
            speech_wave=speech_wave,
            noise_wave=noise_wave,
            clip_samples=self.clip_samples,
            snr_db=snr_db,
            include_speech=include_speech,
        )
        feats = compute_log_mel(
            waveform,
            n_mels=self.audio_cfg.n_mels,
            frame_len_ms=self.audio_cfg.frame_len_ms,
            hop_len_ms=self.audio_cfg.hop_len_ms,
        )
        labels = make_frame_labels(
            speech_range=speech_range,
            clip_samples=self.clip_samples,
            frame_len=self.frame_len,
            hop_len=self.hop_len,
            positive_overlap_ratio=0.5,
        )
        t = min(feats.shape[0], len(labels))
        return {
            "feats": feats[:t].float(),
            "labels": torch.tensor(labels[:t], dtype=torch.float32),
            "waveform": waveform,
            "speech_range": speech_range,
            "snr_db": meta["snr_db"],
            "include_speech": meta["include_speech"],
        }

    def __getitem__(self, index: int):
        ex = self.build_example(index)
        return ex["feats"], ex["labels"], ex["snr_db"]


def collate_batch(batch):
    feat_list, label_list, snr_list = zip(*batch)
    max_t = max(x.shape[0] for x in feat_list)
    feat_dim = feat_list[0].shape[1]
    feats = torch.zeros(len(batch), max_t, feat_dim)
    labels = torch.zeros(len(batch), max_t)
    mask = torch.zeros(len(batch), max_t)
    for i, (f, y) in enumerate(zip(feat_list, label_list)):
        t = f.shape[0]
        feats[i, :t] = f
        labels[i, :t] = y
        mask[i, :t] = 1.0
    return feats, labels, mask, snr_list


def masked_bce_loss(logits, labels, mask, pos_weight=None):
    loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    loss = loss_fn(logits, labels)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def compute_counts(preds, labels, mask):
    valid = mask > 0
    preds = preds[valid]
    labels = labels[valid]
    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def summarize_counts(counts):
    tp, tn, fp, fn = counts["tp"], counts["tn"], counts["fp"], counts["fn"]
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {**counts, "precision": precision, "recall": recall, "f1": f1}


def run_epoch(model, loader, optimizer, device, pos_weight=None, train=True):
    model.train(train)
    total_loss = 0.0
    total_counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for feats, labels, mask, _ in tqdm(loader, leave=False):
        feats = feats.to(device)
        labels = labels.to(device)
        mask = mask.to(device)
        with torch.set_grad_enabled(train):
            logits = model(feats)
            loss = masked_bce_loss(logits, labels, mask, pos_weight=pos_weight)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item()
        probs = torch.sigmoid(logits.detach())
        preds = (probs >= 0.5).float()
        batch_counts = compute_counts(preds, labels, mask)
        for k in total_counts:
            total_counts[k] += batch_counts[k]
    return {"loss": total_loss / max(1, len(loader)), **summarize_counts(total_counts)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_manifest", type=str, required=True)
    parser.add_argument("--val_manifest", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_path", type=str, default="outputs/vad_model.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device = {device}")
    if device == "cuda":
        print(f"[train] GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    audio_cfg = AudioConfig()

    train_ds = BalancedSyntheticVADDataset(args.train_manifest, audio_cfg)
    val_ds = BalancedSyntheticVADDataset(args.val_manifest, audio_cfg)
    loader_kwargs = {"num_workers": 4, "pin_memory": True, "persistent_workers": True, "prefetch_factor": 4} if device == "cuda" else {}
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch, **loader_kwargs)

    model = SmallCRNNVAD(n_mels=audio_cfg.n_mels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    pos_weight = torch.tensor([2.0], device=device)

    best_f1 = -1.0
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(model, train_loader, optimizer, device, pos_weight=pos_weight, train=True)
        val_stats = run_epoch(model, val_loader, optimizer, device, pos_weight=pos_weight, train=False)
        print(
            f"Epoch {epoch:02d} | train_loss={train_stats['loss']:.4f} train_f1={train_stats['f1']:.4f} | "
            f"val_loss={val_stats['loss']:.4f} val_f1={val_stats['f1']:.4f}"
        )
        if val_stats["f1"] > best_f1:
            best_f1 = val_stats["f1"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "audio_cfg": audio_cfg.__dict__,
                    "best_val_stats": val_stats,
                },
                save_path,
            )
            print(f"Saved best model to {save_path}")


if __name__ == "__main__":
    main()
