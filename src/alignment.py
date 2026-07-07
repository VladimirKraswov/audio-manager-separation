from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio_io import ensure_float32, match_length, resample_audio


@dataclass
class AlignmentInfo:
    delay_samples: int
    delay_ms: float
    alpha: float
    correlation: float


def estimate_delay_samples(
    original: np.ndarray,
    estimate: np.ndarray,
    sr: int,
    *,
    max_delay_ms: float = 250.0,
    analysis_sec: float = 30.0,
    analysis_sr: int = 4000,
) -> tuple[int, float]:
    original = ensure_float32(original)
    estimate = ensure_float32(estimate)
    if original.size == 0 or estimate.size == 0:
        return 0, 0.0

    n = min(len(original), len(estimate), int(round(analysis_sec * sr)))
    x = original[:n]
    y = estimate[:n]
    if sr > analysis_sr:
        x = resample_audio(x, sr, analysis_sr)
        y = resample_audio(y, sr, analysis_sr)
        used_sr = analysis_sr
    else:
        used_sr = sr

    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    max_delay = max(1, int(round(max_delay_ms * used_sr / 1000.0)))
    lag, corr = _limited_best_lag(x, y, max_delay)

    # scipy-style lag is inverted for our purpose: positive delay means estimate is late.
    delay_at_used_sr = -lag
    delay_samples = int(round(delay_at_used_sr * float(sr) / float(used_sr)))
    return delay_samples, corr


def _limited_best_lag(x: np.ndarray, y: np.ndarray, max_delay: int) -> tuple[int, float]:
    best_lag = 0
    best_corr = 0.0
    best_abs = -1.0
    for lag in range(-max_delay, max_delay + 1):
        if lag >= 0:
            xs = x[lag:]
            ys = y[: len(xs)]
        else:
            ys = y[-lag:]
            xs = x[: len(ys)]
        n = min(len(xs), len(ys))
        if n < 16:
            continue
        xs = xs[:n]
        ys = ys[:n]
        denom = float(np.linalg.norm(xs) * np.linalg.norm(ys) + 1e-12)
        corr = float(np.dot(xs, ys) / denom)
        if abs(corr) > best_abs:
            best_abs = abs(corr)
            best_corr = corr
            best_lag = lag
    return best_lag, best_corr


def shift_to_align_estimate(estimate: np.ndarray, delay_samples: int, target_len: int) -> np.ndarray:
    estimate = ensure_float32(estimate)
    if delay_samples > 0:
        shifted = estimate[delay_samples:]
    elif delay_samples < 0:
        shifted = np.concatenate([np.zeros(abs(delay_samples), dtype=np.float32), estimate])
    else:
        shifted = estimate
    return match_length(shifted, target_len)


def fit_gain_alpha(original: np.ndarray, estimate: np.ndarray, *, min_alpha: float = 0.7, max_alpha: float = 1.3) -> float:
    original = ensure_float32(original)
    estimate = ensure_float32(estimate)
    n = min(len(original), len(estimate))
    if n == 0:
        return 1.0
    x = original[:n].astype(np.float64)
    y = estimate[:n].astype(np.float64)
    denom = float(np.dot(y, y))
    if denom <= 1e-12:
        return 1.0
    alpha = float(np.dot(x, y) / denom)
    return float(np.clip(alpha, min_alpha, max_alpha))


def align_estimate_and_gain(original: np.ndarray, estimate: np.ndarray, sr: int) -> tuple[np.ndarray, AlignmentInfo]:
    original = ensure_float32(original)
    estimate = ensure_float32(estimate)
    delay, corr = estimate_delay_samples(original, estimate, sr)
    aligned = shift_to_align_estimate(estimate, delay, len(original))
    alpha = fit_gain_alpha(original, aligned)
    return aligned, AlignmentInfo(
        delay_samples=delay,
        delay_ms=delay * 1000.0 / float(sr),
        alpha=alpha,
        correlation=corr,
    )
