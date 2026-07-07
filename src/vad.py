from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .audio_io import ensure_float32, rms


def frame_rms(audio: np.ndarray, sr: int, frame_ms: float = 30.0, hop_ms: float = 10.0) -> Tuple[np.ndarray, int]:
    audio = ensure_float32(audio)
    frame = max(1, int(round(sr * frame_ms / 1000.0)))
    hop = max(1, int(round(sr * hop_ms / 1000.0)))
    if len(audio) <= frame:
        return np.array([rms(audio)], dtype=np.float32), hop
    values = []
    for start in range(0, len(audio) - frame + 1, hop):
        chunk = audio[start : start + frame]
        values.append(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
    return np.asarray(values, dtype=np.float32), hop


def energy_vad(
    audio: np.ndarray,
    sr: int,
    *,
    frame_ms: float = 30.0,
    hop_ms: float = 10.0,
    db_below_peak: float = 36.0,
    min_dbfs: float = -55.0,
) -> Tuple[np.ndarray, np.ndarray]:
    values, hop = frame_rms(audio, sr, frame_ms=frame_ms, hop_ms=hop_ms)
    if values.size == 0:
        return np.zeros(0, dtype=bool), values
    db = 20.0 * np.log10(np.maximum(values, 1e-10))
    threshold = max(float(np.max(db) - db_below_peak), min_dbfs)
    mask = db >= threshold
    mask = _smooth_boolean(mask, width=5)
    sample_mask = np.zeros(len(audio), dtype=bool)
    frame = max(1, int(round(sr * frame_ms / 1000.0)))
    for i, active in enumerate(mask):
        if active:
            start = i * hop
            sample_mask[start : min(start + frame, len(audio))] = True
    return sample_mask, values


def sample_mask_to_segments(mask: np.ndarray, sr: int, min_sec: float = 0.1) -> List[Tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    changes = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    min_len = int(round(min_sec * sr))
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s >= min_len]


def collect_active_audio(audio: np.ndarray, mask: np.ndarray) -> np.ndarray:
    audio = ensure_float32(audio)
    if mask.size != audio.size or not np.any(mask):
        return audio.copy()
    return audio[mask].astype(np.float32)


def clipped_ratio(audio: np.ndarray, threshold: float = 0.98) -> float:
    audio = ensure_float32(audio)
    if audio.size == 0:
        return 0.0
    return float(np.mean(np.abs(audio) >= threshold))


def _smooth_boolean(mask: np.ndarray, width: int = 5) -> np.ndarray:
    if mask.size == 0 or width <= 1:
        return mask
    kernel = np.ones(width, dtype=np.float32)
    padded = np.pad(mask.astype(np.float32), (width // 2, width // 2), mode="edge")
    conv = np.convolve(padded, kernel, mode="valid")
    return conv >= max(1.0, width * 0.35)
