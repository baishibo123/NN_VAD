from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from energy_vad import energy_vad, tune_energy_threshold
from models import SmallCRNNVAD
from train_vad import AudioConfig, BalancedSyntheticVADDataset
from utils_audio import TARGET_SR, set_seed


def summarize_counts(tp, tn, fp, fn):
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    fpr = fp / max(1, fp + tn)
    fnr = fn / max(1, fn + tp)
    false_alarm_rate = fpr
    miss_rate = fnr
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr,
        "false_alarm_rate": false_alarm_rate,
        "miss_rate": miss_rate,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def update_counts(counts, y_true, y_pred):
    counts["tp"] += int(np.sum((y_pred == 1) & (y_true == 1)))
    counts["tn"] += int(np.sum((y_pred == 0) & (y_true == 0)))
    counts["fp"] += int(np.sum((y_pred == 1) & (y_true == 0)))
    counts["fn"] += int(np.sum((y_pred == 0) & (y_true == 1)))


def build_fixed_examples(ds: BalancedSyntheticVADDataset, snr_values: list[float | None]):
    examples = []
    for idx in range(len(ds)):
        for snr_db in snr_values:
            include_speech = snr_db is not None
            ex = ds.build_example(idx, snr_db=snr_db, include_speech=include_speech)
            examples.append(ex)
    return examples


def print_stats(title, stats, inference_time_sec):
    print(f"\n{title}")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    print(f"inference_time_sec: {inference_time_sec:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_manifest", type=str, required=True, help="Used to tune the energy threshold.")
    parser.add_argument("--test_manifest", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.model_path, map_location=device)
    audio_cfg = AudioConfig(**ckpt["audio_cfg"])
    model = SmallCRNNVAD(n_mels=audio_cfg.n_mels).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    val_ds = BalancedSyntheticVADDataset(args.val_manifest, audio_cfg)
    test_ds = BalancedSyntheticVADDataset(args.test_manifest, audio_cfg)

    frame_len = int(TARGET_SR * audio_cfg.frame_len_ms / 1000.0)
    hop_len = int(TARGET_SR * audio_cfg.hop_len_ms / 1000.0)

    val_examples = build_fixed_examples(val_ds, [20.0, 10.0, 0.0, None])
    thresholds = np.linspace(1e-5, 0.02, 40).tolist()
    best_threshold, best_val_stats = tune_energy_threshold(val_examples, thresholds, frame_len, hop_len)
    print(f"Selected energy threshold on validation set: {best_threshold:.6f}")
    print(f"Validation F1 at selected threshold: {best_val_stats['f1']:.4f}")

    groups = {
        "clean_noise_only": [None],
        "snr_20db": [20.0],
        "snr_10db": [10.0],
        "snr_0db": [0.0],
        "all_conditions": [20.0, 10.0, 0.0, None],
    }

    for group_name, snr_values in groups.items():
        examples = build_fixed_examples(test_ds, snr_values)
        dl_counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
        en_counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
        dl_time = 0.0
        en_time = 0.0

        for ex in examples:
            y_true = ex["labels"].detach().cpu().numpy()

            start = time.perf_counter()
            with torch.no_grad():
                logits = model(ex["feats"].unsqueeze(0).to(device))
                y_dl = (torch.sigmoid(logits).squeeze(0).cpu().numpy() >= 0.5).astype(np.float32)
            dl_time += time.perf_counter() - start

            start = time.perf_counter()
            y_en = energy_vad(ex["waveform"], frame_len, hop_len, best_threshold)
            en_time += time.perf_counter() - start

            t = min(len(y_true), len(y_dl), len(y_en))
            update_counts(dl_counts, y_true[:t], y_dl[:t])
            update_counts(en_counts, y_true[:t], y_en[:t])

        dl_stats = summarize_counts(**dl_counts)
        en_stats = summarize_counts(**en_counts)
        print_stats(f"Deep-learning VAD [{group_name}]", dl_stats, dl_time)
        print_stats(f"Energy VAD [{group_name}]", en_stats, en_time)


if __name__ == "__main__":
    main()
