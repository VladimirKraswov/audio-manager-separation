from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .audio_io import dbfs, ensure_float32, peak_limit
from .vad import energy_vad


def active_loudness_db(audio: np.ndarray, sr: int, mask: np.ndarray | None = None) -> Tuple[float, Dict]:
    """Return active-region loudness.

    Uses pyloudnorm/BS.1770 when available, otherwise falls back to active RMS
    dBFS. The caller can pass a speech-presence mask so loudness is measured on
    the same speech-like regions for input and output.
    """
    audio = ensure_float32(audio)
    if mask is None or mask.size != len(audio) or not np.any(mask):
        mask, _ = energy_vad(audio, sr, db_below_peak=30.0, min_dbfs=-65.0)
    active = audio[mask] if mask.size == len(audio) and np.any(mask) else audio
    active = ensure_float32(active)
    meta = {
        "active_samples": int(len(active)),
        "active_sec": len(active) / float(sr) if sr else 0.0,
        "method": "active_dbfs",
    }
    if len(active) < max(1, int(round(0.40 * sr))):
        return dbfs(active), meta

    try:
        import pyloudnorm as pyln  # type: ignore

        meter = pyln.Meter(sr)
        loudness = float(meter.integrated_loudness(active))
        if np.isfinite(loudness):
            meta["method"] = "bs1770_lufs"
            return loudness, meta
    except Exception as exc:
        meta["pyloudnorm_error"] = f"{type(exc).__name__}: {exc}"
    return dbfs(active), meta


def match_loudness_to_input(
    speech: np.ndarray,
    original: np.ndarray,
    sr: int,
    *,
    mode: str = "input_matched",
    fixed_target_db: float = -23.0,
    max_gain_db: float = 18.0,
    max_cut_db: float = 24.0,
    true_peak_db: float = -1.0,
) -> Tuple[np.ndarray, Dict]:
    """Match extracted speech loudness to input speech-active regions."""
    speech = ensure_float32(speech)
    original = ensure_float32(original)
    n = min(len(speech), len(original))
    speech = speech[:n]
    original = original[:n]
    mask, frame_energy = energy_vad(speech, sr, db_below_peak=30.0, min_dbfs=-65.0)
    if mask.size != n:
        mask = np.ones(n, dtype=bool)

    speech_loudness, speech_meta = active_loudness_db(speech, sr, mask)
    if mode == "fixed":
        target_loudness = float(fixed_target_db)
        target_meta = {"method": "fixed_target", "target_db": target_loudness}
    elif mode == "input_matched":
        target_loudness, target_meta = active_loudness_db(original, sr, mask)
    else:
        raise ValueError(f"Unknown speech loudness mode: {mode}")

    desired_gain_db = float(target_loudness - speech_loudness)
    gain_db = float(np.clip(desired_gain_db, -abs(max_cut_db), abs(max_gain_db)))
    gain = 10.0 ** (gain_db / 20.0)
    matched = (speech * gain).astype(np.float32)
    peak_ceiling = float(10.0 ** (true_peak_db / 20.0))
    matched = peak_limit(matched, ceiling=peak_ceiling)
    final_loudness, final_meta = active_loudness_db(matched, sr, mask)
    return matched, {
        "mode": mode,
        "target_loudness_db": target_loudness,
        "speech_loudness_before_db": speech_loudness,
        "speech_loudness_after_db": final_loudness,
        "desired_gain_db": desired_gain_db,
        "applied_gain_db": gain_db,
        "max_gain_db": max_gain_db,
        "max_cut_db": max_cut_db,
        "true_peak_db": true_peak_db,
        "peak_ceiling": peak_ceiling,
        "active_mask_ratio": float(np.mean(mask)) if mask.size else 0.0,
        "frame_energy_mean": float(np.mean(frame_energy)) if frame_energy.size else 0.0,
        "target_loudness": target_meta,
        "speech_loudness": speech_meta,
        "final_loudness": final_meta,
    }
