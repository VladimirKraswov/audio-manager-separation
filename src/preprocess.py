from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .audio_io import dbfs, ensure_float32, match_length, rms


@dataclass
class PreprocessInfo:
    dc_offset: float
    gain_db: float
    highpass_hz: float | None
    input_dbfs: float
    output_dbfs: float


def remove_dc(audio: np.ndarray) -> Tuple[np.ndarray, float]:
    audio = ensure_float32(audio)
    if audio.size == 0:
        return audio, 0.0
    offset = float(np.mean(audio))
    return (audio - offset).astype(np.float32), offset


def one_pole_highpass(audio: np.ndarray, sr: int, cutoff_hz: float = 70.0) -> np.ndarray:
    audio = ensure_float32(audio)
    if cutoff_hz <= 0.0 or audio.size == 0:
        return audio
    rc = 1.0 / (2.0 * np.pi * cutoff_hz)
    dt = 1.0 / float(sr)
    alpha = rc / (rc + dt)
    out = np.zeros_like(audio, dtype=np.float32)
    prev_y = 0.0
    prev_x = float(audio[0])
    for i, x in enumerate(audio):
        y = alpha * (prev_y + float(x) - prev_x)
        out[i] = y
        prev_y = y
        prev_x = float(x)
    return out


def normalize_rms(audio: np.ndarray, target_dbfs: float = -24.0, max_gain_db: float = 18.0) -> Tuple[np.ndarray, float]:
    audio = ensure_float32(audio)
    current = dbfs(audio)
    if current <= -119.0:
        return audio, 0.0
    gain_db = float(np.clip(target_dbfs - current, -max_gain_db, max_gain_db))
    gain = 10.0 ** (gain_db / 20.0)
    return (audio * gain).astype(np.float32), gain_db


def normalize_rms_asymmetric(
    audio: np.ndarray,
    target_dbfs: float,
    *,
    max_boost_db: float = 6.0,
    max_cut_db: float = 30.0,
) -> Tuple[np.ndarray, float]:
    audio = ensure_float32(audio)
    current = dbfs(audio)
    if current <= -119.0:
        return audio, 0.0
    desired = target_dbfs - current
    gain_db = float(np.clip(desired, -abs(max_cut_db), abs(max_boost_db)))
    gain = 10.0 ** (gain_db / 20.0)
    return (audio * gain).astype(np.float32), gain_db


def preprocess_audio(
    audio: np.ndarray,
    sr: int,
    *,
    highpass_hz: float | None = None,
    normalize: bool = False,
    target_dbfs: float = -24.0,
) -> Tuple[np.ndarray, PreprocessInfo]:
    input_level = dbfs(audio)
    audio, dc = remove_dc(audio)
    if highpass_hz:
        audio = one_pole_highpass(audio, sr, highpass_hz)
    gain_db = 0.0
    if normalize:
        audio, gain_db = normalize_rms(audio, target_dbfs=target_dbfs)
    return audio.astype(np.float32), PreprocessInfo(
        dc_offset=dc,
        gain_db=gain_db,
        highpass_hz=highpass_hz,
        input_dbfs=input_level,
        output_dbfs=dbfs(audio),
    )


def cleanup_manager_speech_intro(
    audio: np.ndarray,
    sr: int,
    *,
    duck_sec: float = 2.2,
    duck_gain: float = 0.18,
    lowpass_sec: float = 6.0,
    lowpass_hz: float = 3200.0,
    lowpass_mix: float = 0.65,
) -> Tuple[np.ndarray, Dict]:
    """Reduce intro applause/transients before final speech loudness normalization."""
    audio = ensure_float32(audio).copy()
    metadata = {
        "duck_sec": duck_sec,
        "duck_gain": duck_gain,
        "lowpass_sec": lowpass_sec,
        "lowpass_hz": lowpass_hz,
        "lowpass_mix": lowpass_mix,
        "lowpass_applied": False,
    }
    if audio.size == 0:
        return audio, metadata

    duck_len = min(len(audio), max(0, int(round(duck_sec * sr))))
    if duck_len > 0:
        envelope = np.ones(duck_len, dtype=np.float32) * float(duck_gain)
        fade = min(max(1, int(round(0.35 * sr))), duck_len)
        envelope[-fade:] = np.linspace(float(duck_gain), 1.0, fade, dtype=np.float32)
        audio[:duck_len] *= envelope

    lowpass_len = min(len(audio), max(0, int(round(lowpass_sec * sr))))
    if lowpass_len > 8 and lowpass_hz > 0.0 and lowpass_mix > 0.0:
        try:
            from scipy.signal import butter, sosfiltfilt  # type: ignore

            cutoff = min(float(lowpass_hz), sr * 0.45)
            sos = butter(4, cutoff, btype="lowpass", fs=sr, output="sos")
            segment = audio[:lowpass_len]
            filtered = sosfiltfilt(sos, segment).astype(np.float32)
            mixed = (float(lowpass_mix) * filtered + (1.0 - float(lowpass_mix)) * segment).astype(np.float32)
            blend = np.ones(lowpass_len, dtype=np.float32)
            fade = min(max(1, int(round(0.8 * sr))), lowpass_len)
            blend[-fade:] = np.linspace(1.0, 0.0, fade, dtype=np.float32)
            audio[:lowpass_len] = blend * mixed + (1.0 - blend) * segment
            metadata["lowpass_applied"] = True
        except Exception as exc:
            metadata["lowpass_error"] = f"{type(exc).__name__}: {exc}"

    return audio.astype(np.float32), metadata


def chunk_audio(
    audio: np.ndarray,
    sr: int,
    *,
    chunk_sec: float = 25.0,
    overlap_sec: float = 4.0,
) -> List[Tuple[int, np.ndarray]]:
    audio = ensure_float32(audio)
    chunk_len = max(1, int(round(chunk_sec * sr)))
    overlap = max(0, int(round(overlap_sec * sr)))
    hop = max(1, chunk_len - overlap)
    chunks: List[Tuple[int, np.ndarray]] = []
    for start in range(0, max(len(audio), 1), hop):
        end = min(start + chunk_len, len(audio))
        if start >= len(audio):
            break
        chunks.append((start, audio[start:end].copy()))
        if end >= len(audio):
            break
    return chunks


def overlap_add(
    chunks: Sequence[Tuple[int, np.ndarray]],
    total_length: int,
    *,
    fade_samples: int,
) -> np.ndarray:
    out = np.zeros(total_length, dtype=np.float32)
    weight = np.zeros(total_length, dtype=np.float32)
    for start, chunk in chunks:
        chunk = ensure_float32(chunk)
        end = min(start + len(chunk), total_length)
        if end <= start:
            continue
        valid = chunk[: end - start]
        win = np.ones(len(valid), dtype=np.float32)
        fade = min(fade_samples, len(valid) // 2)
        if fade > 0:
            ramp = np.linspace(0.0, 1.0, fade, endpoint=False, dtype=np.float32)
            win[:fade] = np.maximum(win[:fade] * ramp, 1e-4)
            win[-fade:] = np.maximum(win[-fade:] * ramp[::-1], 1e-4)
        out[start:end] += valid * win
        weight[start:end] += win
    active = weight > 1e-8
    out[active] /= weight[active]
    return match_length(out, total_length)


def describe_chunks(audio: np.ndarray, sr: int, chunk_sec: float, overlap_sec: float) -> dict:
    chunks = chunk_audio(audio, sr, chunk_sec=chunk_sec, overlap_sec=overlap_sec)
    return {
        "chunk_sec": chunk_sec,
        "overlap_sec": overlap_sec,
        "num_chunks": len(chunks),
        "duration_sec": len(audio) / float(sr) if sr else 0.0,
        "rms": rms(audio),
    }
