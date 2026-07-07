from __future__ import annotations

from typing import Dict

import numpy as np

from .audio_io import dbfs, ensure_float32, rms


def spectral_similarity(a: np.ndarray, b: np.ndarray, sr: int) -> float:
    """Small proxy for speaker/reference similarity when embeddings are unavailable."""
    a = ensure_float32(a)
    b = ensure_float32(b)
    if a.size == 0 or b.size == 0:
        return 0.0
    n_fft = 1024
    pa = _mean_log_spectrum(a, n_fft)
    pb = _mean_log_spectrum(b, n_fft)
    denom = float(np.linalg.norm(pa) * np.linalg.norm(pb))
    if denom <= 1e-12:
        return 0.0
    sim = float(np.dot(pa, pb) / denom)
    return float(np.clip((sim + 1.0) / 2.0, 0.0, 1.0))


def speech_quality_proxy(audio: np.ndarray) -> float:
    audio = ensure_float32(audio)
    if audio.size == 0:
        return 0.0
    level = dbfs(audio)
    peak = float(np.max(np.abs(audio)))
    clipped = float(np.mean(np.abs(audio) > 0.98))
    level_score = 1.0 - min(abs(level + 24.0) / 36.0, 1.0)
    clipping_score = 1.0 - min(clipped * 100.0, 1.0)
    silence_penalty = 0.0 if rms(audio) < 1e-5 else 1.0
    return float(np.clip(0.65 * level_score + 0.35 * clipping_score, 0.0, 1.0) * silence_penalty)


def background_suppression_proxy(original: np.ndarray, speech: np.ndarray) -> float:
    original = ensure_float32(original)
    speech = ensure_float32(speech)
    n = min(len(original), len(speech))
    if n == 0:
        return 0.0
    original_energy = float(np.mean(original[:n].astype(np.float64) ** 2))
    residual_like = original[:n] - speech[:n]
    removed_energy = float(np.mean(residual_like.astype(np.float64) ** 2))
    if original_energy <= 1e-12:
        return 0.0
    ratio = removed_energy / original_energy
    return float(np.clip(ratio, 0.0, 1.0))


def leakage_proxy(residual: np.ndarray, reference: np.ndarray, sr: int) -> float:
    return spectral_similarity(residual, reference, sr)


def candidate_score(original: np.ndarray, speech: np.ndarray, residual: np.ndarray, reference: np.ndarray, sr: int) -> Dict:
    speaker_sim = spectral_similarity(speech, reference, sr)
    leakage = leakage_proxy(residual, reference, sr)
    quality = speech_quality_proxy(speech)
    suppression = background_suppression_proxy(original, speech)
    low_leakage = 1.0 - leakage
    overall = (
        0.35 * speaker_sim
        + 0.25 * quality
        + 0.20 * suppression
        + 0.15 * low_leakage
        + 0.05 * quality
    )
    return {
        "speaker_similarity_proxy": speaker_sim,
        "speech_quality_proxy": quality,
        "background_suppression_proxy": suppression,
        "target_leakage_proxy": leakage,
        "low_target_leakage_score": low_leakage,
        "asr_confidence_proxy": quality,
        "overall": float(np.clip(overall, 0.0, 1.0)),
        "note": "Proxy metrics only. Install SpeechBrain/DNSMOS/ASR integrations for production scoring.",
    }


def si_sdr(estimate: np.ndarray, target: np.ndarray) -> float:
    estimate = ensure_float32(estimate).astype(np.float64)
    target = ensure_float32(target).astype(np.float64)
    n = min(len(estimate), len(target))
    if n == 0:
        return -120.0
    estimate = estimate[:n] - np.mean(estimate[:n])
    target = target[:n] - np.mean(target[:n])
    scale = np.dot(estimate, target) / (np.dot(target, target) + 1e-12)
    projected = scale * target
    noise = estimate - projected
    return float(10.0 * np.log10((np.sum(projected**2) + 1e-12) / (np.sum(noise**2) + 1e-12)))


def _mean_log_spectrum(audio: np.ndarray, n_fft: int) -> np.ndarray:
    audio = ensure_float32(audio)
    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)))
    hop = n_fft // 2
    window = np.hanning(n_fft).astype(np.float32)
    spectra = []
    limit = len(audio) - n_fft + 1
    for start in range(0, limit, hop):
        frame = audio[start : start + n_fft] * window
        spectra.append(np.log1p(np.abs(np.fft.rfft(frame))))
    if not spectra:
        return np.zeros(n_fft // 2 + 1, dtype=np.float32)
    return np.mean(np.stack(spectra, axis=0), axis=0).astype(np.float32)
