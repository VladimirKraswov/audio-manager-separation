from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from ..audio_io import ensure_float32, peak_limit, read_audio, write_wav
from ..preprocess import one_pole_highpass


def enhance_speech(
    input_wav_path: str | Path,
    output_wav_path: str | Path,
    sr: int,
    *,
    device: str = "cpu",
) -> dict:
    """Run DeepFilterNet if configured, else apply a mild deterministic polish."""
    input_wav_path = Path(input_wav_path)
    output_wav_path = Path(output_wav_path)

    template = os.environ.get("DEEPFILTERNET_CMD")
    if template:
        cmd = template.format(input=input_wav_path, output=output_wav_path, device=device).split()
        subprocess.run(cmd, check=True)
        return {"model": "deepfilternet", "mode": "external_command", "command": template}

    deep_filter = shutil.which("deepFilter")
    if deep_filter:
        temp_dir = output_wav_path.parent / "_deepfilter_tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([deep_filter, str(input_wav_path), "--output-dir", str(temp_dir)], check=True)
        produced = sorted(temp_dir.glob("*.wav"))
        if produced:
            audio, produced_sr = read_audio(produced[0], target_sr=sr)
            write_wav(output_wav_path, audio, produced_sr, subtype="PCM_16")
            return {"model": "deepfilternet", "mode": "deepFilter_cli"}

    audio, file_sr = read_audio(input_wav_path, target_sr=sr)
    polished = mild_post_enhance(audio, sr)
    write_wav(output_wav_path, polished, file_sr, subtype="PCM_16")
    return {"model": "mild_builtin_postprocess", "mode": "fallback", "warning": "DeepFilterNet unavailable"}


def mild_post_enhance(audio: np.ndarray, sr: int) -> np.ndarray:
    audio = ensure_float32(audio)
    audio = one_pole_highpass(audio, sr, cutoff_hz=60.0)
    return peak_limit(audio, ceiling=0.98)
