from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from energy_vad import energy_vad, tune_energy_threshold
from models import SmallCRNNVAD
from train_vad import AudioConfig, BalancedSyntheticVADDataset
from utils_audio import TARGET_SR, set_seed

# Optional: matplotlib for comparison plot
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def summarize_counts(tp, tn, fp, fn):
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    fpr = fp / max(1, fp + tn)
    fnr = fn / max(1, fn + tp)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr,
        "false_alarm_rate": fpr,
        "miss_rate": fnr,
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


def _precompute_dnn_probs(model, examples, device, batch_size=64):
    """Run batched inference once and return (probs_list, labels_list) as numpy arrays."""
    all_probs, all_labels = [], []
    for i in range(0, len(examples), batch_size):
        batch = examples[i:i + batch_size]
        feats = torch.stack([ex["feats"] for ex in batch]).to(device)
        with torch.no_grad():
            probs_batch = torch.sigmoid(model(feats)).cpu().numpy()
        for j, ex in enumerate(batch):
            t_len = min(probs_batch.shape[1], len(ex["labels"]))
            all_probs.append(probs_batch[j, :t_len])
            all_labels.append(ex["labels"].detach().cpu().numpy()[:t_len])
    return all_probs, all_labels


def _run_dnn_batch(model, examples, threshold, device, batch_size=64):
    """Batched DNN inference returning counts dict. Used in the main eval loop."""
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for i in range(0, len(examples), batch_size):
        batch = examples[i:i + batch_size]
        feats = torch.stack([ex["feats"] for ex in batch]).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(feats)).cpu().numpy()
        for j, ex in enumerate(batch):
            y_true = ex["labels"].detach().cpu().numpy()
            y_pred = (probs[j] >= threshold).astype(np.float32)
            t_len = min(len(y_true), len(y_pred))
            update_counts(counts, y_true[:t_len], y_pred[:t_len])
    return counts


def tune_dnn_threshold(model, val_examples, thresholds, device):
    best_threshold = 0.5
    best_score = -1.0
    best_stats = None

    all_probs, all_labels = _precompute_dnn_probs(model, val_examples, device)

    for t in thresholds:
        counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
        for probs_i, labels_i in zip(all_probs, all_labels):
            y_pred = (probs_i >= t).astype(np.float32)
            update_counts(counts, labels_i, y_pred)
        stats = summarize_counts(**counts)
        score = stats["f1"] - 0.3 * stats["fpr"]
        if score > best_score:
            best_score = score
            best_threshold = t
            best_stats = stats

    print(f"Selected DNN threshold on validation set:  {best_threshold:.4f}")
    print(f"Validation F1 at selected DNN threshold:   {best_stats['f1']:.4f}")
    return best_threshold, best_stats


# ── NEW: Side-by-side bar chart comparing DNN vs Energy VAD ──
def plot_comparison(dl_results, en_results, out_path="outputs/comparison_plot.png"):
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available – skipping comparison plot.")
        return

    conditions = list(dl_results.keys())
    metrics = ["f1", "recall", "precision"]
    titles = ["F1 Score", "Recall", "Precision"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    x = np.arange(len(conditions))
    width = 0.35

    for ax, metric, title in zip(axes, metrics, titles):
        dl_vals = [dl_results[c]["stats"][metric] for c in conditions]
        en_vals = [en_results[c]["stats"][metric] for c in conditions]
        bars1 = ax.bar(x - width / 2, dl_vals, width, label="DNN VAD (tuned)", color="#2E75B6", alpha=0.85)
        bars2 = ax.bar(x + width / 2, en_vals, width, label="Energy VAD (tuned)", color="#ED7D31", alpha=0.85)
        ax.set_ylim(0, 1.12)
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=15, ha="right", fontsize=8)
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=8)
        for bar in list(bars1) + list(bars2):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=7)

    fig.suptitle("DNN VAD vs Energy VAD — Apple-to-Apple Comparison\n"
                 "(both thresholds tuned on validation set)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    import os; os.makedirs("outputs", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"\nComparison plot saved to {out_path}")
    plt.close()


def plot_dnn_threshold_sweep(model, val_examples, thresholds, device,
                              out_path="outputs/dnn_threshold_sweep.png"):
    if not MATPLOTLIB_AVAILABLE:
        return

    all_probs, all_labels = _precompute_dnn_probs(model, val_examples, device)

    f1s, precisions, recalls = [], [], []
    for t in thresholds:
        counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
        for probs_i, labels_i in zip(all_probs, all_labels):
            y_pred = (probs_i >= t).astype(np.float32)
            update_counts(counts, labels_i, y_pred)
        s = summarize_counts(**counts)
        f1s.append(s["f1"])
        precisions.append(s["precision"])
        recalls.append(s["recall"])

    best_idx = int(np.argmax(f1s))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, f1s, label="F1", color="#2E75B6", linewidth=2)
    ax.plot(thresholds, precisions, label="Precision", color="#ED7D31", linewidth=2, linestyle="--")
    ax.plot(thresholds, recalls, label="Recall", color="#70AD47", linewidth=2, linestyle=":")
    ax.axvline(thresholds[best_idx], color="red", linestyle="--", alpha=0.7,
               label=f"Best threshold = {thresholds[best_idx]:.3f}")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title("DNN VAD: Metrics vs Decision Threshold (Validation Set)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    import os; os.makedirs("outputs", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"DNN threshold sweep plot saved to {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_manifest", type=str, required=True,
                        help="Used to tune BOTH energy and DNN thresholds.")
    parser.add_argument("--test_manifest", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[evaluate] device = {device}")
    if device == "cuda":
        print(f"[evaluate] GPU: {torch.cuda.get_device_name(0)}")

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

    # ── Tune energy threshold (unchanged) ──
    energy_thresholds = np.linspace(1e-5, 0.02, 40).tolist()
    best_energy_threshold, best_energy_val_stats = tune_energy_threshold(
        val_examples, energy_thresholds, frame_len, hop_len
    )
    print(f"Selected energy threshold on validation set: {best_energy_threshold:.6f}")
    print(f"Validation F1 at selected energy threshold:  {best_energy_val_stats['f1']:.4f}")

    # ── NEW: Tune DNN threshold on the same val set ──
    dnn_thresholds = np.linspace(0.3, 0.95, 40).tolist()
    best_dnn_threshold, _ = tune_dnn_threshold(model, val_examples, dnn_thresholds, device)

    # ── NEW: Plot DNN threshold sweep ──
    plot_dnn_threshold_sweep(model, val_examples, dnn_thresholds, device)

    groups = {
        "clean_noise_only": [None],
        "snr_20db": [20.0],
        "snr_10db": [10.0],
        "snr_0db": [0.0],
        "all_conditions": [20.0, 10.0, 0.0, None],
    }

    dl_results = {}
    en_results = {}

    for group_name, snr_values in groups.items():
        examples = build_fixed_examples(test_ds, snr_values)
        en_counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

        # Counts: batched inference (fast, accurate)
        dl_counts = _run_dnn_batch(model, examples, best_dnn_threshold, device)

        # Timing: single-example latency × N — fair comparison with energy VAD.
        # GPU ops are async so cuda.synchronize() is required before stopping the clock.
        _probe = examples[0]["feats"].unsqueeze(0).to(device)
        with torch.no_grad():  # warm-up
            _ = model(_probe)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(_probe)
        if device == "cuda":
            torch.cuda.synchronize()
        dl_time = (time.perf_counter() - t0) * len(examples)

        en_start = time.perf_counter()
        for ex in examples:
            y_true = ex["labels"].detach().cpu().numpy()
            y_en = energy_vad(ex["waveform"], frame_len, hop_len, best_energy_threshold)
            t = min(len(y_true), len(y_en))
            update_counts(en_counts, y_true[:t], y_en[:t])
        en_time = time.perf_counter() - en_start

        dl_stats = summarize_counts(**dl_counts)
        en_stats = summarize_counts(**en_counts)

        dl_results[group_name] = {"stats": dl_stats, "time": dl_time}
        en_results[group_name] = {"stats": en_stats, "time": en_time}

        print_stats(f"Deep-learning VAD [{group_name}]", dl_stats, dl_time)
        print_stats(f"Energy VAD [{group_name}]", en_stats, en_time)

    # ── NEW: Save comparison bar chart ──
    plot_comparison(dl_results, en_results)


if __name__ == "__main__":
    main()
