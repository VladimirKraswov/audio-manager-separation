#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

from src.dual_input import (
    finalize_dual_client_audio,
    prepare_dual_input_artifacts,
    run_manager_pipeline,
    write_dual_report,
)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    started = time.time()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    manager_input, dual_metadata = prepare_dual_input_artifacts(
        args.mix,
        args.manager_mic,
        outdir,
        highpass_hz=args.highpass_hz,
        cancel_method=args.dual_cancel_method,
        cancel_strength=args.dual_cancel_strength,
        spectral_strength=args.dual_spectral_strength,
        client_leak_strength=args.dual_client_leak_strength,
        client_leak_spectral_strength=args.dual_client_leak_spectral_strength,
        max_delay_ms=args.dual_max_delay_ms,
        drift_window_sec=args.dual_drift_window_sec,
        drift_hop_sec=args.dual_drift_hop_sec,
        correct_drift=args.dual_correct_drift,
    )
    write_json(outdir / "dual_prepare_report.json", dual_metadata)

    manager_command = run_manager_pipeline(
        Path(__file__).resolve().parent,
        manager_input,
        args.reference,
        outdir,
        build_manager_process_args(args),
    )
    final_client_meta = finalize_dual_client_audio(
        outdir,
        cancel_method=args.dual_cancel_method,
        cancel_strength=args.dual_final_cancel_strength,
        spectral_strength=args.dual_final_spectral_strength,
        max_delay_ms=args.dual_max_delay_ms,
    )
    report = write_dual_report(
        outdir,
        dual_metadata,
        final_client_meta,
        manager_command,
        reference_path=args.reference,
    )
    report["runtime_sec"] = time.time() - started
    write_json(outdir / "report.json", report)
    print(
        json.dumps(
            {
                "outdir": str(outdir),
                "mode": "mix_plus_manager_mic",
                "selected_tse_model": report["manager_separation"].get("selected_tse_model"),
                "alignment_confidence": report["alignment"].get("alignment_confidence"),
                "warnings": report["manager_separation"].get("warnings", []),
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dual-input call mix + manager mic separation pipeline.")
    parser.add_argument("--mix", required=True, help="Common call mix: client + manager side")
    parser.add_argument("--manager-mic", required=True, help="Separate manager mic recording")
    parser.add_argument("--reference", required=True, help="Clean manager voice reference")
    parser.add_argument("--outdir", default="output_dual", help="Output directory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--quality", default="max", choices=["smoke", "fast", "max"])
    parser.add_argument("--models", default="wesep", help="Comma-separated TSE candidates")
    parser.add_argument("--disable-fallback", action="store_true")
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument("--chunk-sec", type=float, default=25.0)
    parser.add_argument("--overlap-sec", type=float, default=4.0)
    parser.add_argument("--highpass-hz", type=float, default=None)

    parser.add_argument("--dual-cancel-method", default="hybrid", choices=["simple", "adaptive_fir", "spectral_mask", "hybrid", "best"])
    parser.add_argument("--dual-cancel-strength", type=float, default=1.0)
    parser.add_argument("--dual-spectral-strength", type=float, default=0.35)
    parser.add_argument("--dual-client-leak-strength", type=float, default=0.80)
    parser.add_argument("--dual-client-leak-spectral-strength", type=float, default=0.20)
    parser.add_argument("--dual-final-cancel-strength", type=float, default=1.0)
    parser.add_argument("--dual-final-spectral-strength", type=float, default=0.35)
    parser.add_argument("--dual-max-delay-ms", type=float, default=3000.0)
    parser.add_argument("--dual-drift-window-sec", type=float, default=30.0)
    parser.add_argument("--dual-drift-hop-sec", type=float, default=15.0)
    parser.add_argument("--dual-correct-drift", action="store_true")

    parser.add_argument("--speech-loudness-mode", default="input_matched", choices=["input_matched", "fixed"])
    parser.add_argument("--speech-target-dbfs", type=float, default=-23.0)
    parser.add_argument("--speech-max-gain-db", type=float, default=18.0)
    parser.add_argument("--speech-true-peak-db", type=float, default=-1.0)
    parser.add_argument("--speech-intro-duck-sec", type=float, default=2.2)
    parser.add_argument("--speech-intro-lowpass-sec", type=float, default=6.0)
    parser.add_argument("--speech-noise-filter-strength", type=float, default=0.78)
    parser.add_argument("--speech-noise-filter-over-subtract", type=float, default=1.35)
    parser.add_argument("--speech-noise-filter-floor", type=float, default=0.08)
    parser.add_argument("--speech-noise-filter-mask-power", type=float, default=1.0)
    parser.add_argument("--speech-postfilter-max-gain-db", type=float, default=4.0)
    parser.add_argument("--residual-base-attenuation", type=float, default=0.97)
    parser.add_argument("--residual-target-dbfs", type=float, default=-45.0)
    parser.add_argument("--residual-leak-suppression", type=float, default=0.94)
    parser.add_argument("--residual-leak-mask-start-ratio", type=float, default=0.18)
    parser.add_argument("--residual-leak-mask-full-ratio", type=float, default=0.68)
    parser.add_argument("--residual-leak-mask-power", type=float, default=0.50)
    parser.add_argument("--require-deepfilternet", action="store_true")
    return parser


def build_manager_process_args(args: argparse.Namespace) -> List[str]:
    cmd = [
        "--device",
        args.device,
        "--quality",
        args.quality,
        "--models",
        args.models,
        "--chunk-sec",
        str(args.chunk_sec),
        "--overlap-sec",
        str(args.overlap_sec),
        "--speech-loudness-mode",
        args.speech_loudness_mode,
        "--speech-target-dbfs",
        str(args.speech_target_dbfs),
        "--speech-max-gain-db",
        str(args.speech_max_gain_db),
        "--speech-true-peak-db",
        str(args.speech_true_peak_db),
        "--speech-intro-duck-sec",
        str(args.speech_intro_duck_sec),
        "--speech-intro-lowpass-sec",
        str(args.speech_intro_lowpass_sec),
        "--speech-noise-filter-strength",
        str(args.speech_noise_filter_strength),
        "--speech-noise-filter-over-subtract",
        str(args.speech_noise_filter_over_subtract),
        "--speech-noise-filter-floor",
        str(args.speech_noise_filter_floor),
        "--speech-noise-filter-mask-power",
        str(args.speech_noise_filter_mask_power),
        "--speech-postfilter-max-gain-db",
        str(args.speech_postfilter_max_gain_db),
        "--residual-base-attenuation",
        str(args.residual_base_attenuation),
        "--residual-target-dbfs",
        str(args.residual_target_dbfs),
        "--residual-leak-suppression",
        str(args.residual_leak_suppression),
        "--residual-leak-mask-start-ratio",
        str(args.residual_leak_mask_start_ratio),
        "--residual-leak-mask-full-ratio",
        str(args.residual_leak_mask_full_ratio),
        "--residual-leak-mask-power",
        str(args.residual_leak_mask_power),
    ]
    if args.disable_fallback:
        cmd.append("--disable-fallback")
    if args.allow_fallback:
        cmd.append("--allow-fallback")
    if args.require_deepfilternet:
        cmd.append("--require-deepfilternet")
    if args.highpass_hz is not None:
        cmd.extend(["--highpass-hz", str(args.highpass_hz)])
    return cmd


def write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
