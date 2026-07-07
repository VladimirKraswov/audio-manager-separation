from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .alignment import estimate_delay_samples, shift_to_align_estimate
from .audio_io import dbfs, ensure_float32, match_length, peak_limit
from .preprocess import remove_dc


def cancel_reference_from_mix(
    target_mix: np.ndarray,
    reference_to_remove: np.ndarray,
    sr: int,
    *,
    method: str = "hybrid",
    cancellation_strength: float = 1.0,
    spectral_strength: float = 0.35,
    max_delay_ms: float = 500.0,
    max_filter_gain_db: float = 12.0,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Remove a reference-like signal from a target mix.

    Returns residual, estimated reference as heard in the target, and metadata.
    `hybrid` uses a frequency-domain adaptive transfer estimate first and then a
    light spectral ducking pass for leftover reference bins.
    """
    target = ensure_float32(target_mix)
    reference = ensure_float32(reference_to_remove)
    n = min(len(target), len(reference))
    target = target[:n]
    reference = reference[:n]
    method = method.lower().strip()
    cancellation_strength = float(np.clip(cancellation_strength, 0.0, 1.5))
    spectral_strength = float(np.clip(spectral_strength, 0.0, 1.0))

    delay, corr = estimate_delay_samples(target, reference, sr, max_delay_ms=max_delay_ms)
    aligned_reference = shift_to_align_estimate(reference, delay, len(target))
    simple_estimate, simple_meta = _simple_reference_estimate(target, aligned_reference)

    metadata: Dict = {
        "method": method,
        "delay_samples": delay,
        "delay_ms": delay * 1000.0 / float(sr),
        "alignment_correlation": corr,
        "input_dbfs": dbfs(target),
        "reference_dbfs": dbfs(aligned_reference),
        "cancellation_strength": cancellation_strength,
        "spectral_strength": spectral_strength,
        **simple_meta,
    }

    if method == "simple":
        estimate = simple_estimate
        residual = target - cancellation_strength * estimate
        metadata["selected_stage"] = "simple_gain_subtraction"
    elif method in {"adaptive_fir", "adaptive", "hybrid", "best"}:
        try:
            residual, estimate, adaptive_meta = _adaptive_stft_cancel(
                target,
                aligned_reference,
                sr,
                cancellation_strength=cancellation_strength,
                max_filter_gain_db=max_filter_gain_db,
            )
            metadata.update(adaptive_meta)
            metadata["selected_stage"] = "adaptive_stft_transfer"
        except Exception as exc:
            estimate = simple_estimate
            residual = target - cancellation_strength * estimate
            metadata["adaptive_error"] = f"{type(exc).__name__}: {exc}"
            metadata["selected_stage"] = "simple_gain_subtraction_fallback"
    elif method == "spectral_mask":
        estimate = simple_estimate
        residual, mask_meta = _spectral_mask_suppress(
            target,
            estimate,
            sr,
            attenuation=max(cancellation_strength, spectral_strength),
        )
        metadata.update(mask_meta)
        metadata["selected_stage"] = "spectral_mask"
    else:
        raise ValueError(f"Unknown reference cancellation method: {method}")

    if method in {"hybrid", "best"} and spectral_strength > 0.0:
        residual, mask_meta = _spectral_mask_suppress(
            residual,
            estimate,
            sr,
            attenuation=spectral_strength,
        )
        metadata["hybrid_spectral_mask"] = mask_meta

    residual, dc = remove_dc(match_length(residual, len(target)))
    estimate = match_length(estimate, len(target))
    metadata.update(
        {
            "dc_offset_removed": dc,
            "estimated_reference_dbfs": dbfs(estimate),
            "residual_dbfs": dbfs(residual),
            "target_energy_before": _energy(target),
            "residual_energy_after": _energy(residual),
            "energy_reduction_db": _energy_reduction_db(target, residual),
        }
    )
    return peak_limit(residual, ceiling=0.98), peak_limit(estimate, ceiling=0.98), metadata


def cancel_reference_simple(
    target: np.ndarray,
    reference: np.ndarray,
    sr: int,
    *,
    cancellation_strength: float = 1.0,
    max_delay_ms: float = 500.0,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    return cancel_reference_from_mix(
        target,
        reference,
        sr,
        method="simple",
        cancellation_strength=cancellation_strength,
        spectral_strength=0.0,
        max_delay_ms=max_delay_ms,
    )


def cancel_reference_adaptive_fir(
    target: np.ndarray,
    reference: np.ndarray,
    sr: int,
    *,
    cancellation_strength: float = 1.0,
    max_delay_ms: float = 500.0,
    max_filter_gain_db: float = 12.0,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    return cancel_reference_from_mix(
        target,
        reference,
        sr,
        method="adaptive_fir",
        cancellation_strength=cancellation_strength,
        spectral_strength=0.0,
        max_delay_ms=max_delay_ms,
        max_filter_gain_db=max_filter_gain_db,
    )


def cancel_reference_spectral_mask(
    target: np.ndarray,
    reference: np.ndarray,
    sr: int,
    *,
    attenuation: float = 0.50,
    max_delay_ms: float = 500.0,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    return cancel_reference_from_mix(
        target,
        reference,
        sr,
        method="spectral_mask",
        cancellation_strength=attenuation,
        spectral_strength=attenuation,
        max_delay_ms=max_delay_ms,
    )


def _simple_reference_estimate(target: np.ndarray, aligned_reference: np.ndarray) -> Tuple[np.ndarray, Dict]:
    n = min(len(target), len(aligned_reference))
    if n == 0:
        return aligned_reference, {"simple_alpha": 0.0}
    x = target[:n].astype(np.float64)
    y = aligned_reference[:n].astype(np.float64)
    denom = float(np.dot(y, y))
    alpha = 0.0 if denom <= 1e-12 else float(np.dot(x, y) / denom)
    alpha = float(np.clip(alpha, -6.0, 6.0))
    return (alpha * aligned_reference).astype(np.float32), {"simple_alpha": alpha}


def _adaptive_stft_cancel(
    target: np.ndarray,
    aligned_reference: np.ndarray,
    sr: int,
    *,
    cancellation_strength: float,
    max_filter_gain_db: float,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    from scipy.signal import istft, stft  # type: ignore

    n_fft = 2048 if sr >= 16000 else 1024
    hop = n_fft // 4
    _, _, target_stft = stft(target, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, boundary="zeros")
    _, _, ref_stft = stft(aligned_reference, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, boundary="zeros")
    ref_power = np.abs(ref_stft) ** 2
    raw_h = target_stft * np.conj(ref_stft) / (ref_power + 1e-8)
    max_gain = float(10.0 ** (max_filter_gain_db / 20.0))
    h_mag = np.abs(raw_h)
    raw_h = raw_h * np.minimum(1.0, max_gain / (h_mag + 1e-8))

    active = ref_power >= np.percentile(ref_power, 35.0)
    h_smooth = np.zeros_like(raw_h)
    previous = np.zeros(raw_h.shape[0], dtype=raw_h.dtype)
    smoothing = 0.82
    for frame in range(raw_h.shape[1]):
        current = raw_h[:, frame]
        update = active[:, frame]
        previous = np.where(update, smoothing * previous + (1.0 - smoothing) * current, previous)
        h_smooth[:, frame] = previous

    estimate_stft = h_smooth * ref_stft
    residual_stft = target_stft - cancellation_strength * estimate_stft
    _, residual = istft(
        residual_stft,
        fs=sr,
        nperseg=n_fft,
        noverlap=n_fft - hop,
        input_onesided=True,
        boundary=True,
    )
    _, estimate = istft(
        estimate_stft,
        fs=sr,
        nperseg=n_fft,
        noverlap=n_fft - hop,
        input_onesided=True,
        boundary=True,
    )
    return ensure_float32(residual), ensure_float32(estimate), {
        "adaptive_method": "stft_transfer_estimate",
        "max_filter_gain_db": max_filter_gain_db,
        "mean_filter_gain": float(np.mean(np.abs(h_smooth))) if h_smooth.size else 0.0,
        "active_reference_bin_ratio": float(np.mean(active)) if active.size else 0.0,
    }


def _spectral_mask_suppress(
    target: np.ndarray,
    reference_estimate: np.ndarray,
    sr: int,
    *,
    attenuation: float,
) -> Tuple[np.ndarray, Dict]:
    from scipy.signal import istft, stft  # type: ignore

    target = ensure_float32(target)
    reference_estimate = match_length(ensure_float32(reference_estimate), len(target))
    n_fft = 2048 if sr >= 16000 else 1024
    hop = n_fft // 4
    _, _, target_stft = stft(target, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, boundary="zeros")
    _, _, ref_stft = stft(reference_estimate, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, boundary="zeros")
    target_mag = np.abs(target_stft)
    ref_mag = np.abs(ref_stft)
    ratio = ref_mag / (ref_mag + target_mag + 1e-8)
    mask = np.clip((ratio - 0.12) / 0.58, 0.0, 1.0) ** 0.65
    residual_stft = target_stft * (1.0 - float(attenuation) * mask)
    _, residual = istft(
        residual_stft,
        fs=sr,
        nperseg=n_fft,
        noverlap=n_fft - hop,
        input_onesided=True,
        boundary=True,
    )
    return ensure_float32(residual), {
        "spectral_mask_attenuation": attenuation,
        "spectral_mask_mean": float(np.mean(mask)) if mask.size else 0.0,
        "spectral_mask_bin_ratio_50": float(np.mean(mask >= 0.50)) if mask.size else 0.0,
    }


def _energy(audio: np.ndarray) -> float:
    audio = ensure_float32(audio)
    if audio.size == 0:
        return 0.0
    return float(np.mean(audio.astype(np.float64) ** 2))


def _energy_reduction_db(before: np.ndarray, after: np.ndarray) -> float:
    before_energy = _energy(before)
    after_energy = _energy(after)
    if before_energy <= 1e-12 or after_energy <= 1e-12:
        return 0.0
    return float(10.0 * np.log10(before_energy / after_energy))
