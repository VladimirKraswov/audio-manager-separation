from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .base import ModelUnavailableError, format_command


def run_llase(
    mixture_wav_path: str | Path,
    reference_wav_path: str | Path,
    output_wav_path: str | Path,
    sample_rate: int,
    device: str,
    task: str = "TSE",
) -> None:
    template = os.environ.get("LLASE_CMD")
    if not template:
        raise ModelUnavailableError("LLASE_CMD is not configured")
    cmd = format_command(
        template,
        mixture=mixture_wav_path,
        reference=reference_wav_path,
        output=output_wav_path,
        sample_rate=sample_rate,
        device=device,
        task=task,
    )
    subprocess.run(cmd, check=True)
