#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.audio_io import read_audio
from src.scoring import candidate_score, si_sdr


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate one pipeline output with proxy metrics.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--speech", required=True)
    parser.add_argument("--residual", required=True)
    parser.add_argument("--clean-target", default=None, help="Optional synthetic ground truth")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    original, sr = read_audio(args.input, mono=True)
    reference, _ = read_audio(args.reference, target_sr=sr, mono=True)
    speech, _ = read_audio(args.speech, target_sr=sr, mono=True)
    residual, _ = read_audio(args.residual, target_sr=sr, mono=True)
    report = candidate_score(original, speech, residual, reference, sr)

    if args.clean_target:
        clean, _ = read_audio(args.clean_target, target_sr=sr, mono=True)
        report["si_sdr_speech_vs_clean_target"] = si_sdr(speech, clean)
        report["si_sdr_input_vs_clean_target"] = si_sdr(original, clean)
        report["si_sdr_improvement"] = report["si_sdr_speech_vs_clean_target"] - report["si_sdr_input_vs_clean_target"]

    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
