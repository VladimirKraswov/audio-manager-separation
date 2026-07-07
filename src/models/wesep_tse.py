from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .base import ModelUnavailableError, format_command


def run_wesep_tse(
    mixture_wav_path: str | Path,
    reference_wav_path: str | Path,
    output_wav_path: str | Path,
    sample_rate: int,
    device: str,
) -> None:
    """Run WeSep through an external command template.

    Set WESEP_TSE_CMD, for example:
      WESEP_TSE_CMD='python tools/infer.py --mix {mixture} --enroll {reference} --out {output} --device {device}'
    """
    template = os.environ.get("WESEP_TSE_CMD")
    if not template:
        raise ModelUnavailableError("WESEP_TSE_CMD is not configured")
    cmd = format_command(
        template,
        mixture=mixture_wav_path,
        reference=reference_wav_path,
        output=output_wav_path,
        sample_rate=sample_rate,
        device=device,
    )
    subprocess.run(cmd, check=True)
