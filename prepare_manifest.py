from __future__ import annotations

import argparse
from pathlib import Path

from utils_audio import save_jsonl, set_seed


def find_files(root: Path, suffix: str):
    return sorted(root.rglob(f"*{suffix}"))


def speaker_id_from_librispeech(path: Path) -> str:
    # .../train-clean-100/<speaker>/<chapter>/<file>.flac
    return path.parts[-3]


def build_musan_fixed_split(musan_root: Path) -> tuple[list[Path], list[Path], list[Path]]:
    # MUSAN has no official train/val/test split, so use a fixed deterministic split by category.
    train_noise = sorted(find_files(musan_root / "music", ".wav"))
    val_noise = sorted(find_files(musan_root / "noise", ".wav"))
    test_noise = sorted(find_files(musan_root / "speech", ".wav"))
    return train_noise, val_noise, test_noise


def build_records(split_name: str, speech_files: list[Path], noise_files: list[Path]) -> list[dict]:
    records = []
    for idx, speech_path in enumerate(speech_files):
        records.append(
            {
                "id": f"{split_name}_{idx:07d}",
                "speech_path": str(speech_path),
                "speaker_id": speaker_id_from_librispeech(speech_path),
                # Noise is sampled later, but we save the candidate pool for traceability.
                "noise_pool": [str(p) for p in noise_files],
                "split": split_name,
            }
        )
    return records


def summarize_speakers(files: list[Path]) -> set[str]:
    return {speaker_id_from_librispeech(p) for p in files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--librispeech_root", type=str, required=True)
    parser.add_argument("--musan_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    libri_root = Path(args.librispeech_root)
    musan_root = Path(args.musan_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_speech = find_files(libri_root / "train-clean-100", ".flac")
    val_speech = find_files(libri_root / "dev-clean", ".flac")
    test_speech = find_files(libri_root / "test-clean", ".flac")

    train_noise, val_noise, test_noise = build_musan_fixed_split(musan_root)

    train_records = build_records("train", train_speech, train_noise)
    val_records = build_records("val", val_speech, val_noise)
    test_records = build_records("test", test_speech, test_noise)

    save_jsonl(train_records, out_dir / "train_manifest.jsonl")
    save_jsonl(val_records, out_dir / "val_manifest.jsonl")
    save_jsonl(test_records, out_dir / "test_manifest.jsonl")

    train_spk = summarize_speakers(train_speech)
    val_spk = summarize_speakers(val_speech)
    test_spk = summarize_speakers(test_speech)

    print(f"Saved train: {len(train_records)}")
    print(f"Saved val:   {len(val_records)}")
    print(f"Saved test:  {len(test_records)}")
    print(f"Speaker overlap train∩val:  {len(train_spk & val_spk)}")
    print(f"Speaker overlap train∩test: {len(train_spk & test_spk)}")
    print(f"Speaker overlap val∩test:   {len(val_spk & test_spk)}")
    print("LibriSpeech uses the official train-clean-100 / dev-clean / test-clean partitions.")
    print("MUSAN does not ship with official train/validation/test splits; this script uses a fixed custom split by category.")
    print("Ground-truth frame labels are generated later from the known speech insertion boundaries inside each synthetic clip.")


if __name__ == "__main__":
    main()
