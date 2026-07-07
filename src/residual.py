from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .alignment import align_estimate_and_gain, estimate_delay_samples, shift_to_align_estimate
from .audio_io import ensure_float32, match_length, peak_limit
from .preprocess import remove_dc


def make_residual(original: np.ndarray, speech_estimate: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Return residual, aligned speech, and metadata.

    residual = original - alpha * aligned_speech_estimate
    """
    original = ensure_float32(original)
    speech_estimate = ensure_float32(speech_estimate)
    aligned, info = align_estimate_and_gain(original, speech_estimate, sr)
    residual = (original - info.alpha * aligned).astype(np.float32)
    residual, dc = remove_dc(residual)
    metadata = {
        "delay_samples": info.delay_samples,
        "delay_ms": info.delay_ms,
        "alpha": info.alpha,
        "alignment_correlation": info.correlation,
        "residual_dc_offset_removed": dc,
    }
    return residual, aligned, metadata


def finalize_residual_for_listening(residual: np.ndarray) -> np.ndarray:
    residual = ensure_float32(residual)
    residual, _ = remove_dc(residual)
    return peak_limit(residual, ceiling=0.98)


def make_manager_suppressed_residual(
    original: np.ndarray,
    speech_estimate: np.ndarray,
    sr: int,
    *,
    max_gain: float = 8.0,
    attenuation: float = 0.97,
    mask_start_ratio: float = 0.04,
    mask_full_ratio: float = 0.65,
    mask_power: float = 0.55,
) -> Tuple[np.ndarray, Dict]:
    """Suppress target-speaker-like bins using the separated speech as a guide."""
    original = ensure_float32(original)
    speech_estimate = ensure_float32(speech_estimate)
    delay, corr = estimate_delay_samples(original, speech_estimate, sr)
    aligned = shift_to_align_estimate(speech_estimate, delay, len(original))
    alpha_raw = _signed_gain_alpha(original, aligned)
    alpha = float(np.clip(alpha_raw, -max_gain, max_gain))
    guide = (alpha * aligned).astype(np.float32)

    method = "signed_subtract_fallback"
    residual = (original - guide).astype(np.float32)
    try:
        residual = _spectral_suppress(
            original,
            guide,
            sr,
            attenuation=attenuation,
            mask_start_ratio=mask_start_ratio,
            mask_full_ratio=mask_full_ratio,
            mask_power=mask_power,
        )
        method = "spectral_manager_suppression"
    except Exception:
        pass

    residual, dc = remove_dc(match_length(residual, len(original)))
    metadata = {
        "method": method,
        "delay_samples": delay,
        "delay_ms": delay * 1000.0 / float(sr),
        "alignment_correlation": corr,
        "alpha_raw": alpha_raw,
        "alpha": alpha,
        "attenuation": attenuation,
        "mask_start_ratio": mask_start_ratio,
        "mask_full_ratio": mask_full_ratio,
        "mask_power": mask_power,
        "residual_dc_offset_removed": dc,
    }
    return peak_limit(residual, ceiling=0.98), metadata


def _signed_gain_alpha(original: np.ndarray, estimate: np.ndarray) -> float:
    n = min(len(original), len(estimate))
    if n == 0:
        return 0.0
    x = original[:n].astype(np.float64)
    y = estimate[:n].astype(np.float64)
    denom = float(np.dot(y, y))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(x, y) / denom)


def _spectral_suppress(
    original: np.ndarray,
    guide: np.ndarray,
    sr: int,
    *,
    attenuation: float,
    mask_start_ratio: float,
    mask_full_ratio: float,
    mask_power: float,
) -> np.ndarray:
    from scipy.signal import istft, stft  # type: ignore

    n_fft = 2048 if sr >= 16000 else 1024
    hop = n_fft // 4
    _, _, original_stft = stft(original, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, boundary="zeros")
    _, _, guide_stft = stft(guide, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, boundary="zeros")
    ratio = np.abs(guide_stft) / (np.abs(original_stft) + 1e-8)
    transition = max(mask_full_ratio - mask_start_ratio, 1e-6)
    mask = np.clip((ratio - mask_start_ratio) / transition, 0.0, 1.0) ** mask_power
    residual_stft = original_stft * (1.0 - attenuation * mask)
    _, residual = istft(
        residual_stft,
        fs=sr,
        nperseg=n_fft,
        noverlap=n_fft - hop,
        input_onesided=True,
        boundary=True,
    )
    return ensure_float32(residual)
