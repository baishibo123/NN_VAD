from __future__ import annotations

import numpy as np
import torch


def frame_signal(waveform: torch.Tensor, frame_len: int, hop_len: int) -> torch.Tensor:
    frames = []
    pos = 0
    while pos + frame_len <= waveform.numel():
        frames.append(waveform[pos : pos + frame_len])
        pos += hop_len
    if not frames:
        return torch.zeros(0, frame_len)
    return torch.stack(frames, dim=0)


def compute_frame_energy(frames: torch.Tensor) -> np.ndarray:
    if frames.numel() == 0:
        return np.zeros(0, dtype=np.float32)
    return torch.mean(frames ** 2, dim=1).cpu().numpy().astype(np.float32)


def energy_vad(waveform: torch.Tensor, frame_len: int, hop_len: int, threshold: float) -> np.ndarray:
    frames = frame_signal(waveform, frame_len, hop_len)
    energies = compute_frame_energy(frames)
    return (energies >= threshold).astype(np.float32)


def tune_energy_threshold(val_examples, thresholds, frame_len, hop_len):
    best_threshold = None
    best_stats = None
    best_score = -1.0

    for threshold in thresholds:
        counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

        for ex in val_examples:
            wav = ex["waveform"]
            y = ex["labels"]

            p = energy_vad(wav, frame_len, hop_len, threshold)

            if hasattr(y, "detach"):
                y = y.detach().cpu().numpy()
            else:
                y = np.asarray(y)

            p = np.asarray(p)

            n = min(len(p), len(y))
            p = p[:n]
            y = y[:n]

            counts["tp"] += int(np.sum((p == 1) & (y == 1)))
            counts["tn"] += int(np.sum((p == 0) & (y == 0)))
            counts["fp"] += int(np.sum((p == 1) & (y == 0)))
            counts["fn"] += int(np.sum((p == 0) & (y == 1)))

        precision = counts["tp"] / max(counts["tp"] + counts["fp"], 1)
        recall = counts["tp"] / max(counts["tp"] + counts["fn"], 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        fpr = counts["fp"] / max(counts["fp"] + counts["tn"], 1)
        score = f1 - 0.3 * fpr

        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_stats = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "fpr": fpr,
                "score": score,
                **counts,
            }

    return best_threshold, best_stats
