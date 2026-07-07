from __future__ import annotations

import math
import wave
from pathlib import Path
from typing import Tuple

import numpy as np


def ensure_float32(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio)
    if audio.dtype.kind in {"f"}:
        out = audio.astype(np.float32, copy=False)
    else:
        max_value = float(np.iinfo(audio.dtype).max)
        out = audio.astype(np.float32) / max_value
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(out, -8.0, 8.0).astype(np.float32, copy=False)


def read_audio(path: str | Path, target_sr: int | None = None, mono: bool = True) -> Tuple[np.ndarray, int]:
    """Read audio as float32 in [-1, 1-ish].

    WAV/FLAC/etc. use soundfile when available. A pure-stdlib WAV PCM fallback is
    included so smoke tests work in minimal Python environments.
    """
    path = Path(path)
    try:
        import soundfile as sf  # type: ignore

        audio, sr = sf.read(str(path), always_2d=False, dtype="float32")
    except Exception:
        if path.suffix.lower() != ".wav":
            raise RuntimeError(
                f"{path} is not a WAV file and optional dependency soundfile/librosa is unavailable"
            )
        audio, sr = _read_wav_pcm(path)

    audio = ensure_float32(audio)
    if mono and audio.ndim == 2:
        audio = audio.mean(axis=1)
    elif not mono and audio.ndim == 1:
        audio = audio[:, None]

    if target_sr and target_sr != sr:
        audio = resample_audio(audio, sr, target_sr)
        sr = target_sr
    return ensure_float32(audio), int(sr)


def write_wav(
    path: str | Path,
    audio: np.ndarray,
    sr: int,
    *,
    subtype: str = "PCM_16",
    prevent_clip: bool = True,
) -> None:
    """Write a mono/stereo WAV.

    If soundfile exists, FLOAT/PCM subtypes are honored. The stdlib fallback writes
    PCM_16 and applies a small safety limiter.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = ensure_float32(audio)
    if prevent_clip:
        audio = peak_limit(audio)

    try:
        import soundfile as sf  # type: ignore

        sf.write(str(path), audio, int(sr), subtype=subtype)
        return
    except Exception:
        _write_wav_pcm16(path, audio, sr)


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if int(orig_sr) == int(target_sr):
        return ensure_float32(audio)

    try:
        from scipy.signal import resample_poly  # type: ignore

        gcd = math.gcd(int(orig_sr), int(target_sr))
        up = int(target_sr) // gcd
        down = int(orig_sr) // gcd
        return ensure_float32(resample_poly(audio, up, down, axis=0))
    except Exception:
        return _linear_resample(audio, int(orig_sr), int(target_sr))


def match_length(audio: np.ndarray, length: int) -> np.ndarray:
    audio = ensure_float32(audio)
    if len(audio) == length:
        return audio
    if len(audio) > length:
        return audio[:length].astype(np.float32, copy=False)
    pad_shape = (length - len(audio),) if audio.ndim == 1 else (length - len(audio), audio.shape[1])
    return np.concatenate([audio, np.zeros(pad_shape, dtype=np.float32)], axis=0)


def peak_limit(audio: np.ndarray, ceiling: float = 0.98) -> np.ndarray:
    audio = ensure_float32(audio)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= ceiling or peak <= 1e-12:
        return audio
    return (audio * (ceiling / peak)).astype(np.float32)


def rms(audio: np.ndarray) -> float:
    audio = ensure_float32(audio)
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def dbfs(audio: np.ndarray) -> float:
    value = rms(audio)
    if value <= 1e-12:
        return -120.0
    return float(20.0 * np.log10(value))


def _linear_resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    audio = ensure_float32(audio)
    if len(audio) == 0:
        return audio
    new_len = max(1, int(round(len(audio) * float(target_sr) / float(orig_sr))))
    old_x = np.arange(len(audio), dtype=np.float64)
    new_x = np.linspace(0.0, max(len(audio) - 1, 0), new_len, dtype=np.float64)
    if audio.ndim == 1:
        return np.interp(new_x, old_x, audio).astype(np.float32)
    channels = [np.interp(new_x, old_x, audio[:, ch]) for ch in range(audio.shape[1])]
    return np.stack(channels, axis=1).astype(np.float32)


def _read_wav_pcm(path: Path) -> Tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if sample_width == 1:
        data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        audio = (data - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(frames, dtype="<i2").astype(np.float32)
        audio = data / 32768.0
    elif sample_width == 3:
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        sign = (raw[:, 2] & 0x80) != 0
        padded = np.zeros((raw.shape[0], 4), dtype=np.uint8)
        padded[:, :3] = raw
        padded[sign, 3] = 0xFF
        data = padded.view("<i4").reshape(-1).astype(np.float32)
        audio = data / 8388608.0
    elif sample_width == 4:
        data = np.frombuffer(frames, dtype="<i4").astype(np.float32)
        audio = data / 2147483648.0
    else:
        raise RuntimeError(f"Unsupported WAV sample width: {sample_width}")

    if channels > 1:
        audio = audio.reshape(-1, channels)
    return ensure_float32(audio), int(sr)


def _write_wav_pcm16(path: Path, audio: np.ndarray, sr: int) -> None:
    audio = np.clip(ensure_float32(audio), -1.0, 1.0)
    if audio.ndim == 1:
        channels = 1
        interleaved = audio
    else:
        channels = audio.shape[1]
        interleaved = audio.reshape(-1)
    pcm = (interleaved * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm.tobytes())
