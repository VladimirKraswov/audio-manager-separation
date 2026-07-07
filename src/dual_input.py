from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from .audio_io import dbfs, read_audio, write_wav
from .dual_input_alignment import align_dual_inputs
from .preprocess import preprocess_audio
from .reference_cancel import cancel_reference_from_mix


def prepare_dual_input_artifacts(
    mix_path: str | Path,
    manager_mic_path: str | Path,
    outdir: str | Path,
    *,
    highpass_hz: float | None = None,
    cancel_method: str = "hybrid",
    cancel_strength: float = 1.0,
    spectral_strength: float = 0.35,
    client_leak_strength: float = 0.80,
    client_leak_spectral_strength: float = 0.20,
    max_delay_ms: float = 3000.0,
    drift_window_sec: float = 30.0,
    drift_hop_sec: float = 15.0,
    correct_drift: bool = False,
) -> Tuple[Path, Dict]:
    """Prepare dual-input files and return the manager track for TSE."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    call_mix, sr = read_audio(mix_path, mono=True)
    manager_mic, _ = read_audio(manager_mic_path, target_sr=sr, mono=True)
    call_mix, call_preprocess = preprocess_audio(call_mix, sr, highpass_hz=highpass_hz, normalize=False)
    manager_mic, manager_preprocess = preprocess_audio(manager_mic, sr, highpass_hz=highpass_hz, normalize=False)

    aligned_call_mix, aligned_manager_mic, alignment_meta = align_dual_inputs(
        call_mix,
        manager_mic,
        sr,
        max_delay_ms=max_delay_ms,
        drift_window_sec=drift_window_sec,
        drift_hop_sec=drift_hop_sec,
        correct_drift=correct_drift,
    )
    write_wav(outdir / "aligned_call_mix.wav", aligned_call_mix, sr, subtype="FLOAT", prevent_clip=False)
    write_wav(outdir / "aligned_manager_mic.wav", aligned_manager_mic, sr, subtype="FLOAT", prevent_clip=False)

    client_audio_0, manager_side_estimate_0, client0_meta = cancel_reference_from_mix(
        aligned_call_mix,
        aligned_manager_mic,
        sr,
        method=cancel_method,
        cancellation_strength=cancel_strength,
        spectral_strength=spectral_strength,
        max_delay_ms=max_delay_ms,
    )
    write_wav(outdir / "client_audio_raw_iter0.wav", client_audio_0, sr, subtype="PCM_16")
    write_wav(outdir / "manager_side_estimate_iter0.wav", manager_side_estimate_0, sr, subtype="PCM_16")

    manager_mic_no_client, client_leak_estimate, manager_leak_meta = cancel_reference_from_mix(
        aligned_manager_mic,
        client_audio_0,
        sr,
        method=cancel_method,
        cancellation_strength=client_leak_strength,
        spectral_strength=client_leak_spectral_strength,
        max_delay_ms=max_delay_ms,
    )
    manager_mic_no_client_path = outdir / "manager_mic_no_client_leak.wav"
    write_wav(manager_mic_no_client_path, manager_mic_no_client, sr, subtype="PCM_16")
    write_wav(outdir / "client_leak_estimate_in_manager_mic.wav", client_leak_estimate, sr, subtype="PCM_16")

    metadata = {
        "sample_rate": sr,
        "inputs": {
            "call_mix": str(Path(mix_path).resolve()),
            "manager_mic": str(Path(manager_mic_path).resolve()),
        },
        "preprocess": {
            "call_mix": call_preprocess.__dict__,
            "manager_mic": manager_preprocess.__dict__,
        },
        "alignment": alignment_meta,
        "client_extraction_iter0": client0_meta,
        "manager_mic_client_leakage": manager_leak_meta,
        "levels_dbfs": {
            "aligned_call_mix": dbfs(aligned_call_mix),
            "aligned_manager_mic": dbfs(aligned_manager_mic),
            "client_audio_raw_iter0": dbfs(client_audio_0),
            "manager_mic_no_client_leak": dbfs(manager_mic_no_client),
        },
        "artifacts": {
            "aligned_call_mix": "aligned_call_mix.wav",
            "aligned_manager_mic": "aligned_manager_mic.wav",
            "client_audio_raw_iter0": "client_audio_raw_iter0.wav",
            "manager_side_estimate_iter0": "manager_side_estimate_iter0.wav",
            "manager_mic_no_client_leak": "manager_mic_no_client_leak.wav",
            "client_leak_estimate_in_manager_mic": "client_leak_estimate_in_manager_mic.wav",
        },
    }
    return manager_mic_no_client_path, metadata


def finalize_dual_client_audio(
    outdir: str | Path,
    *,
    cancel_method: str = "hybrid",
    cancel_strength: float = 1.0,
    spectral_strength: float = 0.35,
    max_delay_ms: float = 3000.0,
) -> Dict:
    outdir = Path(outdir)
    aligned_call_mix, sr = read_audio(outdir / "aligned_call_mix.wav", mono=True)
    manager_mic_no_client, _ = read_audio(outdir / "manager_mic_no_client_leak.wav", target_sr=sr, mono=True)
    client_audio, manager_side_estimate, final_meta = cancel_reference_from_mix(
        aligned_call_mix,
        manager_mic_no_client,
        sr,
        method=cancel_method,
        cancellation_strength=cancel_strength,
        spectral_strength=spectral_strength,
        max_delay_ms=max_delay_ms,
    )
    write_wav(outdir / "client_audio.wav", client_audio, sr, subtype="PCM_16")
    write_wav(outdir / "manager_side_estimate_in_mix.wav", manager_side_estimate, sr, subtype="PCM_16")
    final_meta["levels_dbfs"] = {
        "client_audio": dbfs(client_audio),
        "manager_side_estimate_in_mix": dbfs(manager_side_estimate),
    }
    return final_meta


def run_manager_pipeline(
    project_root: str | Path,
    manager_input: str | Path,
    reference: str | Path,
    outdir: str | Path,
    process_args: List[str],
) -> Dict:
    cmd = [
        sys.executable,
        str(Path(project_root) / "process_call.py"),
        "--input",
        str(manager_input),
        "--reference",
        str(reference),
        "--outdir",
        str(outdir),
        *process_args,
    ]
    completed = subprocess.run(
        cmd,
        cwd=str(project_root),
        text=True,
        capture_output=True,
        check=False,
    )
    command_info = {
        "argv": cmd,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-12000:],
        "stderr": completed.stderr[-12000:],
    }
    if completed.returncode != 0:
        raise RuntimeError(f"process_call.py failed with code {completed.returncode}: {completed.stderr[-2000:]}")
    return command_info


def write_dual_report(
    outdir: str | Path,
    dual_metadata: Dict,
    final_client_meta: Dict,
    manager_command: Dict,
    *,
    reference_path: str | Path,
) -> Dict:
    outdir = Path(outdir)
    manager_report_path = outdir / "report.json"
    manager_report = json.loads(manager_report_path.read_text(encoding="utf-8")) if manager_report_path.exists() else {}
    report = {
        "mode": "mix_plus_manager_mic",
        "inputs": {
            **dual_metadata.get("inputs", {}),
            "reference": str(Path(reference_path).resolve()),
        },
        "sample_rate": dual_metadata.get("sample_rate"),
        "alignment": dual_metadata.get("alignment", {}),
        "client_extraction": {
            "iter0": dual_metadata.get("client_extraction_iter0", {}),
            "final": final_client_meta,
        },
        "manager_mic_client_leakage": dual_metadata.get("manager_mic_client_leakage", {}),
        "manager_separation": {
            "selected_tse_model": manager_report.get("selected_tse_model"),
            "selected_speech_enhancement_model": manager_report.get("selected_speech_enhancement_model"),
            "final_scores": manager_report.get("final_scores", {}),
            "warnings": manager_report.get("warnings", []),
        },
        "outputs": {
            "client_audio": "client_audio.wav",
            "manager_speech_clean": "manager_speech_clean.wav",
            "manager_noise_residual": "manager_noise_residual.wav",
            "manager_mic_no_client_leak": "manager_mic_no_client_leak.wav",
            "client_audio_raw_iter0": "client_audio_raw_iter0.wav",
            "report": "report.json",
        },
        "dual_input": dual_metadata,
        "manager_pipeline_report": manager_report,
        "manager_pipeline_command": manager_command,
    }
    manager_report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report
