from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from .audio_io import ensure_float32, write_wav
from .preprocess import preprocess_audio
from .vad import clipped_ratio, collect_active_audio, energy_vad, sample_mask_to_segments


def prepare_reference(reference: np.ndarray, sr: int, outdir: str | Path) -> Dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cleaned, info = preprocess_audio(reference, sr, highpass_hz=60.0, normalize=True, target_dbfs=-24.0)
    vad_mask, frame_energy = energy_vad(cleaned, sr)
    active = collect_active_audio(cleaned, vad_mask)
    usable_sec = len(active) / float(sr)
    vad_ratio = float(np.mean(vad_mask)) if vad_mask.size else 0.0
    clip_ratio = clipped_ratio(reference)

    write_wav(outdir / "ref_full.wav", cleaned, sr, subtype="PCM_16")
    for seconds in (10, 20, 60):
        segment = _best_contiguous_segment(cleaned, vad_mask, sr, seconds)
        write_wav(outdir / f"ref_best_{seconds}s.wav", segment, sr, subtype="PCM_16")

    quality = "good"
    warnings = []
    if usable_sec < 10.0:
        quality = "bad_short_reference"
        warnings.append("reference_usable_speech_under_10s")
    elif usable_sec < 30.0:
        quality = "medium"
    if clip_ratio > 0.005:
        warnings.append("reference_clipping_detected")

    return {
        "usable_speech_sec": usable_sec,
        "vad_ratio": vad_ratio,
        "clipping_detected": clip_ratio > 0.0,
        "clipped_ratio": clip_ratio,
        "quality": quality,
        "warnings": warnings,
        "preprocess": info.__dict__,
        "segments": [
            {"start_sample": int(s), "end_sample": int(e)}
            for s, e in sample_mask_to_segments(vad_mask, sr)
        ],
        "frame_energy_mean": float(np.mean(frame_energy)) if frame_energy.size else 0.0,
    }


def _best_contiguous_segment(audio: np.ndarray, vad_mask: np.ndarray, sr: int, seconds: int) -> np.ndarray:
    audio = ensure_float32(audio)
    target_len = min(len(audio), max(1, int(round(seconds * sr))))
    if len(audio) <= target_len:
        return audio.copy()
    if vad_mask.size != len(audio):
        vad_mask = np.ones(len(audio), dtype=bool)
    score = np.convolve(vad_mask.astype(np.float32), np.ones(target_len, dtype=np.float32), mode="valid")
    start = int(np.argmax(score)) if score.size else 0
    return audio[start : start + target_len].copy()
