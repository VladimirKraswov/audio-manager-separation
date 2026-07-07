#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

from src.alignment import estimate_delay_samples, shift_to_align_estimate
from src.audio_io import dbfs, match_length, peak_limit, read_audio, resample_audio, write_wav
from src.loudness import match_loudness_to_input
from src.models.base import ModelUnavailableError
from src.models.clearvoice_tse import run_clearvoice_tse
from src.models.deepfilternet import enhance_speech
from src.models.fallback_tse import reference_guided_spectral_tse
from src.models.llase import run_llase
from src.models.metis_tse import run_metis_tse
from src.models.wesep_tse import run_wesep_tse
from src.preprocess import (
    cleanup_manager_speech_intro,
    describe_chunks,
    normalize_rms_asymmetric,
    overlap_add,
    preprocess_audio,
    chunk_audio,
)
from src.reference import prepare_reference
from src.residual import finalize_residual_for_listening, make_manager_suppressed_residual, make_residual
from src.scoring import candidate_score


TSE_MODELS: Dict[str, Dict] = {
    "wesep": {
        "sample_rate": 16000,
        "filename": "wesep_manager_speech_raw.wav",
        "runner": run_wesep_tse,
    },
    "clearvoice": {
        "sample_rate": 8000,
        "filename": "clearvoice_manager_speech_raw.wav",
        "runner": run_clearvoice_tse,
    },
    "metis": {
        "sample_rate": 16000,
        "filename": "metis_manager_speech.wav",
        "runner": run_metis_tse,
    },
    "llase": {
        "sample_rate": 16000,
        "filename": "llase_manager_speech.wav",
        "runner": run_llase,
    },
    "fallback": {
        "sample_rate": None,
        "filename": "fallback_manager_speech_raw.wav",
        "runner": None,
    },
}


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    started = time.time()

    outdir = Path(args.outdir)
    candidates_dir = outdir / "candidates"
    prepared_dir = outdir / "prepared"
    refs_dir = outdir / "references"
    for directory in (outdir, candidates_dir, prepared_dir, refs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    warnings: List[str] = []
    original, original_sr = read_audio(args.input, mono=True)
    original_aligned, original_info = preprocess_audio(
        original,
        original_sr,
        highpass_hz=args.highpass_hz,
        normalize=False,
    )
    original_aligned = match_length(original_aligned, len(original))
    write_wav(outdir / "original_aligned.wav", original_aligned, original_sr, subtype="FLOAT", prevent_clip=False)

    reference, _ = read_audio(args.reference, target_sr=original_sr, mono=True)
    reference_info = prepare_reference(reference, original_sr, refs_dir)
    warnings.extend(reference_info.get("warnings", []))

    disable_fallback = args.disable_fallback or (args.quality == "max" and not args.allow_fallback)
    model_names = [name.strip().lower() for name in args.models.split(",") if name.strip()]
    if disable_fallback:
        model_names = [name for name in model_names if name != "fallback"]
    if not model_names:
        raise RuntimeError("No TSE models requested after disabling fallback")
    if "fallback" not in model_names and not disable_fallback:
        model_names.append("fallback")

    candidate_paths: Dict[str, Path] = {}
    candidate_scores: Dict[str, Dict] = {}
    candidate_failures: Dict[str, str] = {}

    for model_name in model_names:
        if model_name not in TSE_MODELS:
            candidate_failures[model_name] = "unknown_model_name"
            continue
        try:
            path = run_tse_candidate(
                model_name,
                TSE_MODELS[model_name],
                original_aligned,
                reference,
                original_sr,
                prepared_dir,
                candidates_dir,
                args.device,
                args.chunk_sec,
                args.overlap_sec,
            )
            candidate_paths[model_name] = path
            speech, _ = read_audio(path, target_sr=original_sr, mono=True)
            speech = match_length(speech, len(original_aligned))
            residual, _, residual_meta = make_residual(original_aligned, speech, original_sr)
            score = candidate_score(original_aligned, speech, residual, reference, original_sr)
            score["alignment"] = residual_meta
            candidate_scores[model_name] = score
        except ModelUnavailableError as exc:
            candidate_failures[model_name] = str(exc)
        except Exception as exc:  # keep trying other candidates
            candidate_failures[model_name] = f"{type(exc).__name__}: {exc}"

    if not candidate_paths:
        raise RuntimeError(f"No TSE candidates succeeded. Failures: {candidate_failures}")

    selected_model = max(candidate_scores, key=lambda name: candidate_scores[name]["overall"])
    selected_speech, _ = read_audio(candidate_paths[selected_model], target_sr=original_sr, mono=True)
    selected_speech = match_length(selected_speech, len(original_aligned))
    write_wav(outdir / "manager_speech_tse_raw.wav", selected_speech, original_sr, subtype="FLOAT", prevent_clip=False)

    tse_delay, tse_corr = estimate_delay_samples(original_aligned, selected_speech, original_sr)
    selected_speech_aligned = shift_to_align_estimate(selected_speech, tse_delay, len(original_aligned))
    write_wav(
        outdir / "manager_speech_tse_aligned.wav",
        selected_speech_aligned,
        original_sr,
        subtype="FLOAT",
        prevent_clip=False,
    )
    selected_speech_gainmatched, tse_gain_meta = match_loudness_to_input(
        selected_speech_aligned,
        original_aligned,
        original_sr,
        mode=args.speech_loudness_mode,
        fixed_target_db=args.speech_target_dbfs,
        max_gain_db=args.speech_max_gain_db,
        true_peak_db=args.speech_true_peak_db,
    )
    tse_gain_meta["alignment_delay_samples"] = tse_delay
    tse_gain_meta["alignment_delay_ms"] = tse_delay * 1000.0 / float(original_sr)
    tse_gain_meta["alignment_correlation"] = tse_corr
    write_wav(
        outdir / "manager_speech_tse_gainmatched.wav",
        selected_speech_gainmatched,
        original_sr,
        subtype="FLOAT",
        prevent_clip=False,
    )

    residual_raw, aligned_speech, residual_meta = make_residual(original_aligned, selected_speech_gainmatched, original_sr)
    write_wav(outdir / "manager_noise_residual_raw.wav", residual_raw, original_sr, subtype="FLOAT", prevent_clip=False)
    subtract_residual = finalize_residual_for_listening(residual_raw)
    write_wav(outdir / "manager_noise_residual_subtract.wav", subtract_residual, original_sr, subtype="PCM_16")

    post_path = candidates_dir / "manager_speech_deepfilternet.wav"
    post_info = enhance_speech(
        outdir / "manager_speech_tse_gainmatched.wav",
        post_path,
        original_sr,
        device=args.device,
        require_real=args.require_deepfilternet,
    )
    if post_info.get("mode") == "fallback":
        warnings.append("deepfilternet_unavailable_builtin_postprocess_used")
    speech_clean, _ = read_audio(post_path, target_sr=original_sr, mono=True)
    speech_clean = peak_limit(match_length(speech_clean, len(original_aligned)))
    speech_clean, speech_cleanup_meta = cleanup_manager_speech_intro(
        speech_clean,
        original_sr,
        duck_sec=args.speech_intro_duck_sec,
        lowpass_sec=args.speech_intro_lowpass_sec,
    )
    speech_clean, speech_loudness_meta = match_loudness_to_input(
        speech_clean,
        original_aligned,
        original_sr,
        mode=args.speech_loudness_mode,
        fixed_target_db=args.speech_target_dbfs,
        max_gain_db=args.speech_max_gain_db,
        true_peak_db=args.speech_true_peak_db,
    )
    write_wav(outdir / "manager_speech_clean.wav", speech_clean, original_sr, subtype="PCM_16")
    final_residual, suppression_meta = make_manager_suppressed_residual(
        original_aligned,
        speech_clean,
        original_sr,
        attenuation=0.97,
    )
    final_residual, residual_gain_db = normalize_rms_asymmetric(
        final_residual,
        target_dbfs=args.residual_target_dbfs,
        max_boost_db=6.0,
        max_cut_db=30.0,
    )
    final_residual = peak_limit(final_residual, ceiling=0.85)
    write_wav(outdir / "manager_noise_residual.wav", final_residual, original_sr, subtype="PCM_16")

    if selected_model == "fallback":
        warnings.append(
            "fallback_dsp_tse_used: install/configure WeSep before production use"
        )
    if candidate_failures:
        warnings.append("some_tse_candidates_unavailable_or_failed")

    final_score = candidate_score(original_aligned, speech_clean, final_residual, reference, original_sr)
    report = {
        "input_file": str(Path(args.input).resolve()),
        "reference_file": str(Path(args.reference).resolve()),
        "duration_sec": len(original_aligned) / float(original_sr),
        "device": args.device,
        "quality": args.quality,
        "selected_tse_model": selected_model,
        "selected_speech_enhancement_model": post_info["model"],
        "sample_rates": {
            "original": original_sr,
            "tse": original_sr,
            "final": original_sr,
        },
        "preprocess": original_info.__dict__,
        "chunking": describe_chunks(original_aligned, original_sr, args.chunk_sec, args.overlap_sec),
        "reference_quality": reference_info,
        "candidate_scores": candidate_scores,
        "candidate_failures": candidate_failures,
        "residual": residual_meta,
        "residual_suppression": suppression_meta,
        "tse_gain_matching": tse_gain_meta,
        "speech_enhancement": {
            **post_info,
            "intro_cleanup": speech_cleanup_meta,
            "final_loudness": speech_loudness_meta,
            "final_loudness_mode": args.speech_loudness_mode,
            "final_target_dbfs_when_fixed": args.speech_target_dbfs,
            "final_output_dbfs": dbfs(speech_clean),
            "final_true_peak_db": args.speech_true_peak_db,
        },
        "residual_loudness": {
            "final_gain_db": residual_gain_db,
            "final_target_dbfs": args.residual_target_dbfs,
            "final_output_dbfs": dbfs(final_residual),
            "final_peak_ceiling": 0.85,
        },
        "final_scores": {
            "speaker_similarity_speech_vs_reference": final_score["speaker_similarity_proxy"],
            "speaker_similarity_residual_vs_reference": final_score["target_leakage_proxy"],
            "estimated_target_leakage_in_residual": final_score["target_leakage_proxy"],
            "estimated_noise_leakage_in_speech": 1.0 - final_score["background_suppression_proxy"],
            "asr_confidence": final_score["asr_confidence_proxy"],
            "overall_confidence": confidence_from_score(final_score["overall"], selected_model),
            "proxy_overall": final_score["overall"],
        },
        "warnings": warnings,
        "quality_readiness": {
            "fallback_disabled": disable_fallback,
            "real_tse_selected": selected_model != "fallback",
            "real_speech_enhancement": post_info.get("model") == "deepfilternet",
            "deepfilternet_required": args.require_deepfilternet,
        },
        "runtime_sec": time.time() - started,
    }
    write_json(outdir / "report.json", report)
    print(json.dumps({"outdir": str(outdir), "selected_tse_model": selected_model, "warnings": warnings}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Target-speaker-first manager audio separation pipeline.")
    parser.add_argument("--input", required=True, help="Path to input/manager_mic_mono.wav")
    parser.add_argument("--reference", required=True, help="Path to input/manager_reference_clean.wav")
    parser.add_argument("--outdir", default="output", help="Output directory")
    parser.add_argument("--device", default="cuda:0", help="Model device, e.g. cuda:0 or cpu")
    parser.add_argument("--quality", default="max", choices=["smoke", "fast", "max"])
    parser.add_argument(
        "--models",
        default="wesep,fallback",
        help="Comma-separated TSE candidates: wesep,clearvoice,metis,llase,fallback",
    )
    parser.add_argument("--disable-fallback", action="store_true", help="Fail if configured TSE models are unavailable")
    parser.add_argument("--allow-fallback", action="store_true", help="Allow DSP fallback even when --quality=max")
    parser.add_argument("--chunk-sec", type=float, default=25.0)
    parser.add_argument("--overlap-sec", type=float, default=4.0)
    parser.add_argument("--highpass-hz", type=float, default=None)
    parser.add_argument("--speech-loudness-mode", default="input_matched", choices=["input_matched", "fixed"])
    parser.add_argument("--speech-target-dbfs", type=float, default=-23.0)
    parser.add_argument("--speech-max-gain-db", type=float, default=18.0)
    parser.add_argument("--speech-true-peak-db", type=float, default=-1.0)
    parser.add_argument("--speech-intro-duck-sec", type=float, default=2.2)
    parser.add_argument("--speech-intro-lowpass-sec", type=float, default=6.0)
    parser.add_argument("--residual-target-dbfs", type=float, default=-45.0)
    parser.add_argument("--require-deepfilternet", action="store_true")
    return parser


def run_tse_candidate(
    model_name: str,
    config: Dict,
    original: np.ndarray,
    reference: np.ndarray,
    original_sr: int,
    prepared_dir: Path,
    candidates_dir: Path,
    device: str,
    chunk_sec: float,
    overlap_sec: float,
) -> Path:
    output_path = candidates_dir / config["filename"]
    model_sr = config["sample_rate"] or original_sr

    if model_name == "fallback":
        ref_model = resample_audio(reference, original_sr, model_sr) if model_sr != original_sr else reference
        chunks = []
        for start, chunk in chunk_audio(original, original_sr, chunk_sec=chunk_sec, overlap_sec=overlap_sec):
            estimate = reference_guided_spectral_tse(chunk, ref_model, original_sr)
            chunks.append((start, estimate))
        speech = overlap_add(chunks, len(original), fade_samples=int(round(overlap_sec * original_sr / 2.0)))
        write_wav(output_path, speech, original_sr, subtype="FLOAT", prevent_clip=False)
        return output_path

    runner: Callable = config["runner"]
    mixture_model = resample_audio(original, original_sr, model_sr)
    reference_model = resample_audio(reference, original_sr, model_sr)
    mixture_path = prepared_dir / f"mixture_{model_sr}.wav"
    reference_path = prepared_dir / f"reference_{model_sr}.wav"
    raw_model_output = prepared_dir / f"{model_name}_raw_{model_sr}.wav"
    write_wav(mixture_path, mixture_model, model_sr, subtype="PCM_16")
    write_wav(reference_path, reference_model, model_sr, subtype="PCM_16")
    runner(mixture_path, reference_path, raw_model_output, model_sr, device)
    model_audio, model_output_sr = read_audio(raw_model_output, mono=True)
    candidate = resample_audio(model_audio, model_output_sr, original_sr)
    candidate = match_length(candidate, len(original))
    write_wav(output_path, candidate, original_sr, subtype="FLOAT", prevent_clip=False)
    return output_path


def confidence_from_score(score: float, selected_model: str) -> str:
    if selected_model == "fallback":
        return "low"
    if score >= 0.78:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


def write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    sys.exit(main())
