from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .audio_io import dbfs, read_audio, write_wav
from .preprocess import preprocess_audio


PROJECT_ROOT = Path(os.environ.get("AMS_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
RUNS_ROOT = Path(os.environ.get("AMS_RUNS_DIR", PROJECT_ROOT / "service_runs")).resolve()
MAX_WORKERS = max(1, int(os.environ.get("AMS_MAX_WORKERS", "1")))
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

ARTIFACTS = {
    "speech": "manager_speech_clean.wav",
    "speech_prefilter": "manager_speech_clean_prefilter.wav",
    "noise": "manager_noise_residual.wav",
    "noise_prefilter": "manager_noise_residual_prefilter.wav",
    "noise_subtract": "manager_noise_residual_subtract.wav",
    "raw_speech": "manager_speech_tse_raw.wav",
    "aligned_speech": "manager_speech_tse_aligned.wav",
    "gainmatched_speech": "manager_speech_tse_gainmatched.wav",
    "original": "original_aligned.wav",
    "report": "report.json",
}


class JobSummary(BaseModel):
    job_id: str
    status: str
    created_at: str
    updated_at: str
    settings: Dict[str, Any]
    selected_tse_model: Optional[str] = None
    error: Optional[str] = None


class JobCreated(BaseModel):
    job_id: str
    status: str
    status_url: str
    report_url: str
    artifacts_url: str


class Defaults(BaseModel):
    device: str = "cuda:0"
    quality: str = "max"
    models: str = "wesep"
    disable_fallback: bool = True
    chunk_sec: float = 25.0
    overlap_sec: float = 4.0
    speech_loudness_mode: str = "input_matched"
    speech_target_dbfs: float = -23.0
    speech_max_gain_db: float = 18.0
    speech_true_peak_db: float = -1.0
    speech_intro_duck_sec: float = 2.2
    speech_intro_lowpass_sec: float = 6.0
    speech_noise_filter_strength: float = 0.78
    speech_noise_filter_over_subtract: float = 1.35
    speech_noise_filter_floor: float = 0.08
    speech_noise_filter_mask_power: float = 1.0
    speech_postfilter_max_gain_db: float = 4.0
    residual_base_attenuation: float = 0.97
    residual_target_dbfs: float = -45.0
    residual_leak_suppression: float = 0.94
    residual_leak_mask_start_ratio: float = 0.18
    residual_leak_mask_full_ratio: float = 0.68
    residual_leak_mask_power: float = 0.50
    require_deepfilternet: bool = False
    auto_reference: bool = True
    auto_reference_sec: float = 20.0


app = FastAPI(
    title="Audio Manager Separation Service",
    description="API service for WeSep-based manager speech separation and residual noise extraction.",
    version="1.0.0",
)


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "service": "audio_manager_separation",
        "docs": "/docs",
        "health": "/health",
        "defaults": "/v1/defaults",
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    env = _runtime_env()
    deepfilternet_available = bool(env.get("DEEPFILTERNET_CMD") or shutil.which("deepFilter"))
    wesep_configured = bool(env.get("WESEP_TSE_CMD"))
    return {
        "status": "ok",
        "project_root": str(PROJECT_ROOT),
        "runs_root": str(RUNS_ROOT),
        "max_workers": MAX_WORKERS,
        "wesep_configured": wesep_configured,
        "deepfilternet_available": deepfilternet_available,
        "ready_for_quality_processing": bool(wesep_configured and deepfilternet_available),
    }


@app.get("/v1/defaults", response_model=Defaults)
def defaults() -> Defaults:
    return Defaults(device=os.environ.get("AMS_DEVICE", "cuda:0"))


@app.post("/v1/jobs", response_model=JobCreated, status_code=202)
async def create_job(
    audio: UploadFile = File(..., description="Input call/noisy audio file"),
    reference: Optional[UploadFile] = File(None, description="Optional clean target-speaker reference"),
    device: str = Form("cuda:0"),
    quality: str = Form("max"),
    models: str = Form("wesep"),
    disable_fallback: bool = Form(True),
    chunk_sec: float = Form(25.0),
    overlap_sec: float = Form(4.0),
    highpass_hz: Optional[float] = Form(None),
    speech_loudness_mode: str = Form("input_matched"),
    speech_target_dbfs: float = Form(-23.0),
    speech_max_gain_db: float = Form(18.0),
    speech_true_peak_db: float = Form(-1.0),
    speech_intro_duck_sec: float = Form(2.2),
    speech_intro_lowpass_sec: float = Form(6.0),
    speech_noise_filter_strength: float = Form(0.78),
    speech_noise_filter_over_subtract: float = Form(1.35),
    speech_noise_filter_floor: float = Form(0.08),
    speech_noise_filter_mask_power: float = Form(1.0),
    speech_postfilter_max_gain_db: float = Form(4.0),
    residual_base_attenuation: float = Form(0.97),
    residual_target_dbfs: float = Form(-45.0),
    residual_leak_suppression: float = Form(0.94),
    residual_leak_mask_start_ratio: float = Form(0.18),
    residual_leak_mask_full_ratio: float = Form(0.68),
    residual_leak_mask_power: float = Form(0.50),
    require_deepfilternet: bool = Form(False),
    auto_reference: bool = Form(True),
    auto_reference_sec: float = Form(20.0),
) -> JobCreated:
    if not audio.filename:
        raise HTTPException(status_code=400, detail="audio filename is required")
    if reference is None and not auto_reference:
        raise HTTPException(status_code=400, detail="reference is required when auto_reference=false")

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex
    job_dir = RUNS_ROOT / job_id
    input_dir = job_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=False)

    audio_path = input_dir / _safe_upload_name(audio.filename, "input_audio")
    await _save_upload(audio, audio_path)

    reference_path = None
    if reference is not None and reference.filename:
        reference_path = input_dir / _safe_upload_name(reference.filename, "reference_audio")
        await _save_upload(reference, reference_path)

    settings = {
        "device": device,
        "quality": quality,
        "models": models,
        "disable_fallback": disable_fallback,
        "chunk_sec": chunk_sec,
        "overlap_sec": overlap_sec,
        "highpass_hz": highpass_hz,
        "speech_loudness_mode": speech_loudness_mode,
        "speech_target_dbfs": speech_target_dbfs,
        "speech_max_gain_db": speech_max_gain_db,
        "speech_true_peak_db": speech_true_peak_db,
        "speech_intro_duck_sec": speech_intro_duck_sec,
        "speech_intro_lowpass_sec": speech_intro_lowpass_sec,
        "speech_noise_filter_strength": speech_noise_filter_strength,
        "speech_noise_filter_over_subtract": speech_noise_filter_over_subtract,
        "speech_noise_filter_floor": speech_noise_filter_floor,
        "speech_noise_filter_mask_power": speech_noise_filter_mask_power,
        "speech_postfilter_max_gain_db": speech_postfilter_max_gain_db,
        "residual_base_attenuation": residual_base_attenuation,
        "residual_target_dbfs": residual_target_dbfs,
        "residual_leak_suppression": residual_leak_suppression,
        "residual_leak_mask_start_ratio": residual_leak_mask_start_ratio,
        "residual_leak_mask_full_ratio": residual_leak_mask_full_ratio,
        "residual_leak_mask_power": residual_leak_mask_power,
        "require_deepfilternet": require_deepfilternet,
        "auto_reference": auto_reference,
        "auto_reference_sec": auto_reference_sec,
    }
    state = {
        "job_id": job_id,
        "status": "queued",
        "created_at": _now(),
        "updated_at": _now(),
        "input_file": str(audio_path),
        "reference_file": str(reference_path) if reference_path else None,
        "settings": settings,
    }
    _write_state(job_dir, state)
    executor.submit(_run_job, job_id)
    return JobCreated(
        job_id=job_id,
        status="queued",
        status_url=f"/v1/jobs/{job_id}",
        report_url=f"/v1/jobs/{job_id}/report",
        artifacts_url=f"/v1/jobs/{job_id}/artifacts",
    )


@app.get("/v1/jobs", response_model=List[JobSummary])
def list_jobs(limit: int = 50) -> List[JobSummary]:
    if not RUNS_ROOT.exists():
        return []
    states = []
    for state_path in sorted(RUNS_ROOT.glob("*/job.json"), key=lambda path: path.stat().st_mtime, reverse=True):
        states.append(JobSummary(**_read_json(state_path)))
        if len(states) >= limit:
            break
    return states


@app.get("/v1/jobs/{job_id}", response_model=JobSummary)
def get_job(job_id: str) -> JobSummary:
    state = _load_state(job_id)
    return JobSummary(**state)


@app.get("/v1/jobs/{job_id}/report")
def get_report(job_id: str) -> Dict[str, Any]:
    state = _load_state(job_id)
    report_path = Path(state.get("output_dir", "")) / "report.json"
    if state["status"] != "succeeded" or not report_path.exists():
        raise HTTPException(status_code=404, detail="report is not ready")
    return _read_json(report_path)


@app.get("/v1/jobs/{job_id}/artifacts")
def list_artifacts(job_id: str) -> Dict[str, Any]:
    state = _load_state(job_id)
    output_dir = Path(state.get("output_dir", ""))
    artifacts = {}
    for key, filename in ARTIFACTS.items():
        path = output_dir / filename
        artifacts[key] = {
            "filename": filename,
            "ready": path.exists(),
            "url": f"/v1/jobs/{job_id}/artifacts/{key}",
        }
    return {
        "job_id": job_id,
        "status": state["status"],
        "artifacts": artifacts,
        "zip_url": f"/v1/jobs/{job_id}/artifacts.zip",
    }


@app.get("/v1/jobs/{job_id}/artifacts/{artifact}")
def download_artifact(job_id: str, artifact: str) -> FileResponse:
    state = _load_state(job_id)
    if state["status"] != "succeeded":
        raise HTTPException(status_code=409, detail="job is not finished")
    filename = ARTIFACTS.get(artifact)
    if not filename:
        raise HTTPException(status_code=404, detail="unknown artifact")
    path = Path(state["output_dir"]) / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="artifact is missing")
    media_type = "application/json" if path.suffix == ".json" else "audio/wav"
    return FileResponse(path, media_type=media_type, filename=path.name)


@app.get("/v1/jobs/{job_id}/artifacts.zip")
def download_zip(job_id: str) -> FileResponse:
    state = _load_state(job_id)
    if state["status"] != "succeeded":
        raise HTTPException(status_code=409, detail="job is not finished")
    zip_path = Path(state["job_dir"]) / "artifacts.zip"
    if not zip_path.exists():
        _make_zip(Path(state["output_dir"]), zip_path)
    return FileResponse(zip_path, media_type="application/zip", filename=f"{job_id}_artifacts.zip")


@app.delete("/v1/jobs/{job_id}")
def delete_job(job_id: str) -> Dict[str, Any]:
    state = _load_state(job_id)
    if state["status"] in {"queued", "running"}:
        raise HTTPException(status_code=409, detail="running jobs cannot be deleted")
    shutil.rmtree(Path(state["job_dir"]), ignore_errors=True)
    return {"job_id": job_id, "deleted": True}


async def _save_upload(upload: UploadFile, path: Path) -> None:
    with path.open("wb") as f:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def _run_job(job_id: str) -> None:
    job_dir = RUNS_ROOT / job_id
    state = _read_json(job_dir / "job.json")
    started = time.time()
    command_info = None
    _update_state(job_dir, {"status": "running", "started_at": _now()})
    try:
        settings = state["settings"]
        input_file = Path(state["input_file"])
        reference_file = Path(state["reference_file"]) if state.get("reference_file") else None
        if reference_file is None:
            reference_file = job_dir / "input" / "reference_auto.wav"
            ref_meta = _make_auto_reference(input_file, reference_file, float(settings["auto_reference_sec"]))
            _update_state(job_dir, {"reference_file": str(reference_file), "auto_reference": ref_meta})

        output_dir = job_dir / "output"
        cmd = _build_process_command(input_file, reference_file, output_dir, settings)
        env = _runtime_env()
        completed = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
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
            raise RuntimeError(f"process_call.py failed with code {completed.returncode}")

        report = _read_json(output_dir / "report.json")
        _make_zip(output_dir, job_dir / "artifacts.zip")
        _update_state(
            job_dir,
            {
                "status": "succeeded",
                "finished_at": _now(),
                "runtime_sec": time.time() - started,
                "output_dir": str(output_dir),
                "selected_tse_model": report.get("selected_tse_model"),
                "command": command_info,
            },
        )
    except Exception as exc:
        _update_state(
            job_dir,
            {
                "status": "failed",
                "finished_at": _now(),
                "runtime_sec": time.time() - started,
                "error": f"{type(exc).__name__}: {exc}",
                "command": command_info,
            },
        )


def _build_process_command(input_file: Path, reference_file: Path, output_dir: Path, settings: Dict[str, Any]) -> List[str]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "process_call.py"),
        "--input",
        str(input_file),
        "--reference",
        str(reference_file),
        "--outdir",
        str(output_dir),
        "--device",
        str(settings["device"]),
        "--quality",
        str(settings["quality"]),
        "--models",
        str(settings["models"]),
        "--chunk-sec",
        str(settings["chunk_sec"]),
        "--overlap-sec",
        str(settings["overlap_sec"]),
        "--speech-loudness-mode",
        str(settings["speech_loudness_mode"]),
        "--speech-target-dbfs",
        str(settings["speech_target_dbfs"]),
        "--speech-max-gain-db",
        str(settings["speech_max_gain_db"]),
        "--speech-true-peak-db",
        str(settings["speech_true_peak_db"]),
        "--speech-intro-duck-sec",
        str(settings["speech_intro_duck_sec"]),
        "--speech-intro-lowpass-sec",
        str(settings["speech_intro_lowpass_sec"]),
        "--speech-noise-filter-strength",
        str(settings["speech_noise_filter_strength"]),
        "--speech-noise-filter-over-subtract",
        str(settings["speech_noise_filter_over_subtract"]),
        "--speech-noise-filter-floor",
        str(settings["speech_noise_filter_floor"]),
        "--speech-noise-filter-mask-power",
        str(settings["speech_noise_filter_mask_power"]),
        "--speech-postfilter-max-gain-db",
        str(settings["speech_postfilter_max_gain_db"]),
        "--residual-base-attenuation",
        str(settings["residual_base_attenuation"]),
        "--residual-target-dbfs",
        str(settings["residual_target_dbfs"]),
        "--residual-leak-suppression",
        str(settings["residual_leak_suppression"]),
        "--residual-leak-mask-start-ratio",
        str(settings["residual_leak_mask_start_ratio"]),
        "--residual-leak-mask-full-ratio",
        str(settings["residual_leak_mask_full_ratio"]),
        "--residual-leak-mask-power",
        str(settings["residual_leak_mask_power"]),
    ]
    if settings.get("disable_fallback"):
        cmd.append("--disable-fallback")
    if settings.get("require_deepfilternet"):
        cmd.append("--require-deepfilternet")
    if settings.get("highpass_hz") is not None:
        cmd.extend(["--highpass-hz", str(settings["highpass_hz"])])
    return cmd


def _make_auto_reference(input_file: Path, output_file: Path, seconds: float) -> Dict[str, Any]:
    audio, sr = read_audio(input_file, mono=True)
    cleaned, prep = preprocess_audio(audio, sr, highpass_hz=60.0, normalize=True, target_dbfs=-24.0)
    frame = max(1, int(round(sr * 0.10)))
    energies = []
    for start in range(0, max(1, len(cleaned) - frame + 1), frame):
        chunk = cleaned[start : start + frame]
        energies.append(float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2) + 1e-12)))
    if not energies:
        start = 0
    else:
        win = max(1, int(round(seconds / 0.10)))
        values = np.asarray(energies, dtype=np.float64)
        if len(values) <= win:
            start = 0
        else:
            scores = np.convolve(values, np.ones(win, dtype=np.float64), mode="valid")
            start = int(np.argmax(scores)) * frame
    length = min(len(cleaned) - start, max(1, int(round(seconds * sr))))
    reference = cleaned[start : start + length]
    write_wav(output_file, reference, sr, subtype="PCM_16")
    return {
        "path": str(output_file),
        "sample_rate": sr,
        "duration_sec": len(reference) / float(sr),
        "start_sec": start / float(sr),
        "end_sec": (start + len(reference)) / float(sr),
        "input_duration_sec": len(audio) / float(sr),
        "input_dbfs": dbfs(audio),
        "prepared_reference_dbfs": dbfs(reference),
        "preprocess": prep.__dict__,
    }


def _runtime_env() -> Dict[str, str]:
    env = os.environ.copy()
    venv_bin = PROJECT_ROOT / ".venv" / "bin"
    if venv_bin.exists():
        env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"
    env_file = PROJECT_ROOT / "env.tse"
    if not env_file.exists():
        env_file = PROJECT_ROOT / "env.tse.example"
    if env_file.exists():
        env.update(_parse_export_env(env_file))
    return env


def _parse_export_env(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or not line.startswith("export "):
            continue
        try:
            parts = shlex.split(line, posix=True)
        except ValueError:
            continue
        for part in parts[1:]:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            values[key] = value
    return values


def _make_zip(output_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in ARTIFACTS.values():
            path = output_dir / name
            if path.exists():
                zf.write(path, arcname=name)
        for folder in ("references", "candidates"):
            root = output_dir / folder
            if root.exists():
                for path in root.rglob("*"):
                    if path.is_file():
                        zf.write(path, arcname=str(path.relative_to(output_dir)))


def _load_state(job_id: str) -> Dict[str, Any]:
    if not _valid_job_id(job_id):
        raise HTTPException(status_code=404, detail="job not found")
    state_path = RUNS_ROOT / job_id / "job.json"
    if not state_path.exists():
        raise HTTPException(status_code=404, detail="job not found")
    return _read_json(state_path)


def _update_state(job_dir: Path, patch: Dict[str, Any]) -> None:
    state_path = job_dir / "job.json"
    state = _read_json(state_path)
    state.update(patch)
    state["updated_at"] = _now()
    _write_state(job_dir, state)


def _write_state(job_dir: Path, state: Dict[str, Any]) -> None:
    state["job_dir"] = str(job_dir)
    state_path = job_dir / "job.json"
    temp_path = job_dir / "job.json.tmp"
    temp_path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp_path.replace(state_path)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_upload_name(filename: str, fallback: str) -> str:
    suffix = Path(filename).suffix.lower()
    stem = Path(filename).stem
    safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem).strip("_")
    return f"{safe_stem or fallback}{suffix}"


def _valid_job_id(job_id: str) -> bool:
    return len(job_id) == 32 and all(ch in "0123456789abcdef" for ch in job_id)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
