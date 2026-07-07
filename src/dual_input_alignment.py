from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .alignment import estimate_delay_samples, shift_to_align_estimate
from .audio_io import ensure_float32, match_length


def align_dual_inputs(
    call_mix: np.ndarray,
    manager_mic: np.ndarray,
    sr: int,
    *,
    max_delay_ms: float = 3000.0,
    drift_window_sec: float = 30.0,
    drift_hop_sec: float = 15.0,
    correct_drift: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Align a manager mic recording to a call mix.

    Positive delay means the manager mic was late and had to be shifted earlier.
    Drift correction is intentionally optional because a bad drift estimate can
    do more harm than a small uncorrected drift on short clips.
    """
    call_mix = ensure_float32(call_mix)
    manager_mic = ensure_float32(manager_mic)
    target_len = len(call_mix)
    manager_mic = match_length(manager_mic, target_len)

    initial_delay, initial_corr = estimate_delay_samples(
        call_mix,
        manager_mic,
        sr,
        max_delay_ms=max_delay_ms,
        analysis_sec=min(max(10.0, drift_window_sec), max(10.0, target_len / float(sr))),
    )
    aligned_manager = shift_to_align_estimate(manager_mic, initial_delay, target_len)
    drift_curve = estimate_delay_over_time(
        call_mix,
        aligned_manager,
        sr,
        window_sec=drift_window_sec,
        hop_sec=drift_hop_sec,
        max_delay_ms=min(max_delay_ms, 500.0),
    )
    drift_ppm = _estimate_drift_ppm(drift_curve, sr)
    corrected = False
    correction_samples = 0
    if correct_drift and len(drift_curve) >= 2:
        correction_samples = int(round(drift_curve[-1]["delay_samples"] - drift_curve[0]["delay_samples"]))
        if abs(correction_samples) > max(1, int(round(0.005 * sr))):
            aligned_manager = correct_sample_clock_drift(aligned_manager, correction_samples, target_len)
            corrected = True

    confidence = _alignment_confidence(initial_corr, drift_curve, drift_ppm)
    metadata = {
        "initial_delay_samples": initial_delay,
        "initial_delay_ms": initial_delay * 1000.0 / float(sr),
        "initial_correlation": initial_corr,
        "estimated_drift_ppm": drift_ppm,
        "drift_correction_applied": corrected,
        "drift_correction_samples": correction_samples if corrected else 0,
        "alignment_confidence": confidence,
        "drift_curve": drift_curve,
    }
    return match_length(call_mix, target_len), match_length(aligned_manager, target_len), metadata


def estimate_delay_over_time(
    target: np.ndarray,
    reference: np.ndarray,
    sr: int,
    *,
    window_sec: float = 30.0,
    hop_sec: float = 15.0,
    max_delay_ms: float = 500.0,
) -> List[Dict]:
    target = ensure_float32(target)
    reference = match_length(ensure_float32(reference), len(target))
    window = max(1, int(round(window_sec * sr)))
    hop = max(1, int(round(hop_sec * sr)))
    if len(target) < max(1, int(round(2.0 * sr))):
        delay, corr = estimate_delay_samples(target, reference, sr, max_delay_ms=max_delay_ms)
        return [{"center_sec": len(target) / (2.0 * sr), "delay_samples": delay, "delay_ms": delay * 1000.0 / sr, "correlation": corr}]

    out: List[Dict] = []
    last_start = max(0, len(target) - window)
    starts = list(range(0, last_start + 1, hop))
    if starts[-1] != last_start:
        starts.append(last_start)
    for start in starts:
        end = min(start + window, len(target))
        if end - start < max(256, int(round(1.0 * sr))):
            continue
        delay, corr = estimate_delay_samples(
            target[start:end],
            reference[start:end],
            sr,
            max_delay_ms=max_delay_ms,
            analysis_sec=(end - start) / float(sr),
        )
        out.append(
            {
                "center_sec": (start + end) / (2.0 * sr),
                "delay_samples": delay,
                "delay_ms": delay * 1000.0 / float(sr),
                "correlation": corr,
            }
        )
    return out


def correct_sample_clock_drift(audio: np.ndarray, end_delay_samples: int, target_len: int) -> np.ndarray:
    """Apply a simple linear time-scale correction from start to end."""
    audio = ensure_float32(audio)
    corrected_len = max(1, len(audio) - int(end_delay_samples))
    if len(audio) == 0:
        return match_length(audio, target_len)
    old_x = np.arange(len(audio), dtype=np.float64)
    new_x = np.linspace(0.0, max(len(audio) - 1, 0), corrected_len, dtype=np.float64)
    corrected = np.interp(new_x, old_x, audio).astype(np.float32)
    return match_length(corrected, target_len)


def _estimate_drift_ppm(curve: List[Dict], sr: int) -> float:
    points = [p for p in curve if abs(float(p.get("correlation", 0.0))) >= 0.08]
    if len(points) < 2:
        return 0.0
    x = np.asarray([p["center_sec"] for p in points], dtype=np.float64)
    y = np.asarray([p["delay_samples"] for p in points], dtype=np.float64)
    if float(np.max(x) - np.min(x)) <= 1e-6:
        return 0.0
    slope_samples_per_sec = float(np.polyfit(x, y, 1)[0])
    return float(slope_samples_per_sec / float(sr) * 1_000_000.0)


def _alignment_confidence(initial_corr: float, curve: List[Dict], drift_ppm: float) -> str:
    corrs = [abs(float(p.get("correlation", 0.0))) for p in curve]
    median_corr = float(np.median(corrs)) if corrs else abs(initial_corr)
    if abs(initial_corr) >= 0.45 and median_corr >= 0.25 and abs(drift_ppm) <= 120.0:
        return "high"
    if abs(initial_corr) >= 0.20 or median_corr >= 0.12:
        return "medium"
    return "low"
