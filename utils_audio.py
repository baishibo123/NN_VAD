from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio

TARGET_SR = 16000


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_audio(audio_path: str, target_sr: int = TARGET_SR) -> torch.Tensor:
    wav, sr = sf.read(audio_path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = torch.tensor(wav, dtype=torch.float32)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, target_sr).squeeze(0)
    return wav


def rms(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(torch.mean(x ** 2) + eps)


def fit_noise_to_length(noise: torch.Tensor, target_len: int) -> torch.Tensor:
    if noise.numel() == 0:
        return torch.zeros(target_len, dtype=torch.float32)
    if noise.numel() < target_len:
        repeats = math.ceil(target_len / noise.numel())
        noise = noise.repeat(repeats)
    start = 0 if noise.numel() == target_len else random.randint(0, noise.numel() - target_len)
    return noise[start : start + target_len]


def mix_at_snr(clean: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    noise = fit_noise_to_length(noise, clean.numel())
    clean_rms = rms(clean)
    noise_rms = rms(noise)
    desired_noise_rms = clean_rms / (10 ** (snr_db / 20.0))
    scale = desired_noise_rms / (noise_rms + 1e-8)
    mixed = clean + noise * scale
    return torch.clamp(mixed, -1.0, 1.0)


def make_balanced_clip(
    speech_wave: torch.Tensor | None,
    noise_wave: torch.Tensor | None,
    clip_samples: int,
    snr_db: float | None,
    include_speech: bool,
) -> tuple[torch.Tensor, tuple[int, int], dict]:
    clip = torch.zeros(clip_samples, dtype=torch.float32)
    label_meta = {
        "include_speech": include_speech,
        "snr_db": snr_db,
    }

    if include_speech and speech_wave is not None and speech_wave.numel() > 0:
        speech_wave = speech_wave[:clip_samples]
        usable_len = speech_wave.numel()
        max_offset = max(0, clip_samples - usable_len)
        offset = random.randint(0, max_offset) if max_offset > 0 else 0
        clip[offset : offset + usable_len] = speech_wave
        speech_range = (offset, offset + usable_len)
    else:
        speech_range = (0, 0)

    if noise_wave is not None:
        background = fit_noise_to_length(noise_wave, clip_samples)
        if include_speech and snr_db is not None and speech_range[1] > speech_range[0]:
            clip = mix_at_snr(clip, background, snr_db=snr_db)
        else:
            # Pure non-speech example: just use background at natural scale.
            peak = torch.max(torch.abs(background)).item()
            if peak > 0:
                clip = 0.3 * background / peak
            else:
                clip = background

    return clip, speech_range, label_meta


def make_frame_labels(
    speech_range: Tuple[int, int],
    clip_samples: int,
    frame_len: int,
    hop_len: int,
    positive_overlap_ratio: float = 0.5,
) -> np.ndarray:
    start_sample, end_sample = speech_range
    labels: list[float] = []
    pos = 0
    while pos + frame_len <= clip_samples:
        frame_start = pos
        frame_end = pos + frame_len
        overlap = max(0, min(frame_end, end_sample) - max(frame_start, start_sample))
        ratio = overlap / frame_len
        labels.append(1.0 if ratio >= positive_overlap_ratio else 0.0)
        pos += hop_len
    return np.asarray(labels, dtype=np.float32)


def compute_log_mel(
    waveform: torch.Tensor,
    sample_rate: int = TARGET_SR,
    n_mels: int = 40,
    frame_len_ms: float = 25.0,
    hop_len_ms: float = 10.0,
) -> torch.Tensor:
    n_fft = int(sample_rate * frame_len_ms / 1000.0)
    hop_length = int(sample_rate * hop_len_ms / 1000.0)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )(waveform.unsqueeze(0))
    return torch.log(mel + 1e-6).squeeze(0).transpose(0, 1)


def save_jsonl(records: List[dict], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row) + "\n")


def load_jsonl(path: str | Path) -> List[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows
