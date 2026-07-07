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


def denoise_speech_with_residual(
    speech: np.ndarray,
    noise_estimate: np.ndarray,
    sr: int,
    *,
    strength: float = 0.78,
    over_subtract: float = 1.35,
    floor: float = 0.08,
    mask_power: float = 1.0,
) -> Tuple[np.ndarray, Dict]:
    """Reduce residual/background-like bins that leaked into the speech track."""
    speech = ensure_float32(speech)
    noise_estimate = match_length(ensure_float32(noise_estimate), len(speech))
    strength = float(np.clip(strength, 0.0, 1.0))
    over_subtract = float(max(0.0, over_subtract))
    floor = float(np.clip(floor, 0.0, 1.0))
    mask_power = float(max(0.05, mask_power))

    if strength <= 0.0 or over_subtract <= 0.0:
        return peak_limit(speech, ceiling=0.98), {
            "enabled": False,
            "method": "disabled",
            "strength": strength,
            "over_subtract": over_subtract,
            "floor": floor,
            "mask_power": mask_power,
        }

    method = "speech_residual_wiener_mask"
    try:
        from scipy.signal import istft, stft  # type: ignore

        n_fft = 2048 if sr >= 16000 else 1024
        hop = n_fft // 4
        _, _, speech_stft = stft(speech, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, boundary="zeros")
        _, _, noise_stft = stft(noise_estimate, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, boundary="zeros")
        speech_mag = np.abs(speech_stft)
        noise_mag = np.abs(noise_stft)
        speech_ratio = speech_mag / (speech_mag + over_subtract * noise_mag + 1e-8)
        base_mask = floor + (1.0 - floor) * (np.clip(speech_ratio, 0.0, 1.0) ** mask_power)
        mask = 1.0 - strength * (1.0 - base_mask)
        cleaned_stft = speech_stft * mask
        _, cleaned = istft(
            cleaned_stft,
            fs=sr,
            nperseg=n_fft,
            noverlap=n_fft - hop,
            input_onesided=True,
            boundary=True,
        )
        cleaned = match_length(cleaned, len(speech))
        cleaned, dc = remove_dc(cleaned)
        metadata = {
            "enabled": True,
            "method": method,
            "strength": strength,
            "over_subtract": over_subtract,
            "floor": floor,
            "mask_power": mask_power,
            "mean_mask": float(np.mean(mask)) if mask.size else 1.0,
            "min_mask": float(np.min(mask)) if mask.size else 1.0,
            "max_mask": float(np.max(mask)) if mask.size else 1.0,
            "dc_offset_removed": dc,
        }
        return peak_limit(ensure_float32(cleaned), ceiling=0.98), metadata
    except Exception as exc:
        return peak_limit(speech, ceiling=0.98), {
            "enabled": False,
            "method": "unavailable",
            "error": f"{type(exc).__name__}: {exc}",
            "strength": strength,
            "over_subtract": over_subtract,
            "floor": floor,
            "mask_power": mask_power,
        }


def suppress_residual_manager_leak(
    residual: np.ndarray,
    speech_guide: np.ndarray,
    sr: int,
    *,
    attenuation: float = 0.94,
    mask_start_ratio: float = 0.18,
    mask_full_ratio: float = 0.68,
    mask_power: float = 0.50,
) -> Tuple[np.ndarray, Dict]:
    """Extra residual pass: duck bins that strongly resemble the manager guide."""
    residual = ensure_float32(residual)
    speech_guide = match_length(ensure_float32(speech_guide), len(residual))
    attenuation = float(np.clip(attenuation, 0.0, 1.0))
    mask_start_ratio = float(np.clip(mask_start_ratio, 0.0, 1.0))
    mask_full_ratio = float(np.clip(mask_full_ratio, mask_start_ratio + 1e-6, 1.0))
    mask_power = float(max(0.05, mask_power))

    if attenuation <= 0.0:
        return peak_limit(residual, ceiling=0.98), {
            "enabled": False,
            "method": "disabled",
            "attenuation": attenuation,
            "mask_start_ratio": mask_start_ratio,
            "mask_full_ratio": mask_full_ratio,
            "mask_power": mask_power,
        }

    try:
        filtered, mask_meta = _spectral_duck_by_guide(
            residual,
            speech_guide,
            sr,
            attenuation=attenuation,
            mask_start_ratio=mask_start_ratio,
            mask_full_ratio=mask_full_ratio,
            mask_power=mask_power,
            bounded_ratio=True,
        )
        filtered, dc = remove_dc(match_length(filtered, len(residual)))
        metadata = {
            "enabled": True,
            "method": "residual_guided_manager_leak_suppression",
            "attenuation": attenuation,
            "mask_start_ratio": mask_start_ratio,
            "mask_full_ratio": mask_full_ratio,
            "mask_power": mask_power,
            "dc_offset_removed": dc,
            **mask_meta,
        }
        return peak_limit(filtered, ceiling=0.98), metadata
    except Exception as exc:
        return peak_limit(residual, ceiling=0.98), {
            "enabled": False,
            "method": "unavailable",
            "error": f"{type(exc).__name__}: {exc}",
            "attenuation": attenuation,
            "mask_start_ratio": mask_start_ratio,
            "mask_full_ratio": mask_full_ratio,
            "mask_power": mask_power,
        }


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
    residual, _ = _spectral_duck_by_guide(
        original,
        guide,
        sr,
        attenuation=attenuation,
        mask_start_ratio=mask_start_ratio,
        mask_full_ratio=mask_full_ratio,
        mask_power=mask_power,
        bounded_ratio=False,
    )
    return residual


def _spectral_duck_by_guide(
    target: np.ndarray,
    guide: np.ndarray,
    sr: int,
    *,
    attenuation: float,
    mask_start_ratio: float,
    mask_full_ratio: float,
    mask_power: float,
    bounded_ratio: bool,
) -> Tuple[np.ndarray, Dict]:
    from scipy.signal import istft, stft  # type: ignore

    n_fft = 2048 if sr >= 16000 else 1024
    hop = n_fft // 4
    _, _, target_stft = stft(target, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, boundary="zeros")
    _, _, guide_stft = stft(guide, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, boundary="zeros")
    target_mag = np.abs(target_stft)
    guide_mag = np.abs(guide_stft)
    if bounded_ratio:
        ratio = guide_mag / (guide_mag + target_mag + 1e-8)
    else:
        ratio = guide_mag / (target_mag + 1e-8)
    transition = max(mask_full_ratio - mask_start_ratio, 1e-6)
    mask = np.clip((ratio - mask_start_ratio) / transition, 0.0, 1.0) ** mask_power
    residual_stft = target_stft * (1.0 - attenuation * mask)
    _, residual = istft(
        residual_stft,
        fs=sr,
        nperseg=n_fft,
        noverlap=n_fft - hop,
        input_onesided=True,
        boundary=True,
    )
    metadata = {
        "bounded_ratio": bounded_ratio,
        "mean_mask": float(np.mean(mask)) if mask.size else 0.0,
        "max_mask": float(np.max(mask)) if mask.size else 0.0,
        "masked_bin_ratio_50": float(np.mean(mask >= 0.50)) if mask.size else 0.0,
        "masked_bin_ratio_90": float(np.mean(mask >= 0.90)) if mask.size else 0.0,
    }
    return ensure_float32(residual), metadata
