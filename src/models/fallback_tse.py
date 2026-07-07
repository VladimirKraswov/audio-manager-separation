from __future__ import annotations

import numpy as np

from ..audio_io import ensure_float32, match_length
from ..vad import energy_vad


def reference_guided_spectral_tse(
    mixture: np.ndarray,
    reference: np.ndarray,
    sr: int,
    *,
    n_fft: int = 1024,
    hop: int | None = None,
) -> np.ndarray:
    """Lightweight smoke-test extractor.

    This is not a production TSE model. It creates a conservative voice-band mask
    shaped by the reference spectrum so the rest of the pipeline can be tested
    before WeSep/ClearVoice/Metis are installed.
    """
    mixture = ensure_float32(mixture)
    reference = ensure_float32(reference)
    hop = hop or n_fft // 4
    ref_profile = _reference_profile(reference, n_fft)
    spec = _stft(mixture, n_fft, hop)
    mag = np.abs(spec)
    phase = np.exp(1j * np.angle(spec))

    floor = np.percentile(mag, 25, axis=1, keepdims=True)
    prior = ref_profile[:, None]
    speech_power = (mag**2) * (0.15 + prior)
    noise_power = (floor**2) * (1.15 - 0.65 * prior)
    mask = speech_power / (speech_power + noise_power + 1e-10)
    mask = np.clip(mask, 0.02, 0.98)

    vad_mask, _ = energy_vad(mixture, sr)
    frame_vad = _frame_activity(vad_mask, n_fft, hop, spec.shape[1])
    mask *= (0.15 + 0.85 * frame_vad[None, :])

    enhanced = _istft(mask * mag * phase, n_fft, hop, len(mixture))
    return match_length(enhanced, len(mixture))


def _reference_profile(reference: np.ndarray, n_fft: int) -> np.ndarray:
    spec = _stft(reference, n_fft, n_fft // 4)
    mag = np.mean(np.abs(spec), axis=1)
    mag = mag / (np.max(mag) + 1e-10)
    freqs = np.linspace(0.0, 1.0, len(mag), dtype=np.float32)
    voice_band = ((freqs >= 80.0 / 8000.0) & (freqs <= 3800.0 / 8000.0)).astype(np.float32)
    profile = 0.7 * mag + 0.3 * voice_band
    profile = profile / (np.max(profile) + 1e-10)
    return profile.astype(np.float32)


def _frame_activity(mask: np.ndarray, n_fft: int, hop: int, frames: int) -> np.ndarray:
    out = np.zeros(frames, dtype=np.float32)
    for i in range(frames):
        start = i * hop
        end = min(start + n_fft, len(mask))
        out[i] = float(np.mean(mask[start:end])) if end > start else 0.0
    return out


def _stft(audio: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    audio = ensure_float32(audio)
    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)))
    pad = n_fft // 2
    padded = np.pad(audio, (pad, pad))
    window = np.hanning(n_fft).astype(np.float32)
    frames = []
    for start in range(0, len(padded) - n_fft + 1, hop):
        frames.append(np.fft.rfft(padded[start : start + n_fft] * window))
    return np.stack(frames, axis=1)


def _istft(spec: np.ndarray, n_fft: int, hop: int, length: int) -> np.ndarray:
    frames = spec.shape[1]
    window = np.hanning(n_fft).astype(np.float32)
    out_len = hop * (frames - 1) + n_fft
    out = np.zeros(out_len, dtype=np.float32)
    weight = np.zeros(out_len, dtype=np.float32)
    for i in range(frames):
        start = i * hop
        frame = np.fft.irfft(spec[:, i], n=n_fft).astype(np.float32)
        out[start : start + n_fft] += frame * window
        weight[start : start + n_fft] += window**2
    active = weight > 1e-8
    out[active] /= weight[active]
    pad = n_fft // 2
    out = out[pad : pad + length]
    return match_length(out, length)
