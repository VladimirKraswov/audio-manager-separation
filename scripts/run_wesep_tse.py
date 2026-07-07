#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import types

import numpy as np
import soundfile as sf


def main() -> int:
    parser = argparse.ArgumentParser(description="Single-file WeSep TSE wrapper.")
    parser.add_argument("--mixture", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pretrain", default="", help="Optional WeSep model dir with avg_model.pt/config.yaml")
    parser.add_argument("--vad", action="store_true")
    parser.add_argument("--chunk-sec", type=float, default=float(os.environ.get("WESEP_CHUNK_SEC", "25.0")))
    parser.add_argument("--overlap-sec", type=float, default=float(os.environ.get("WESEP_OVERLAP_SEC", "4.0")))
    parser.add_argument("--progress-file", default=os.environ.get("WESEP_PROGRESS_FILE", ""))
    args = parser.parse_args()
    started = time.time()

    import torchaudio

    # s3prl/WeSpeaker still calls the legacy backend switch removed from recent
    # torchaudio releases. Newer torchaudio selects the backend internally.
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
    if not hasattr(torchaudio, "get_audio_backend"):
        torchaudio.get_audio_backend = lambda: None  # type: ignore[attr-defined]

    def _soundfile_load(path, normalize=True, channels_first=True, **_kwargs):
        import numpy as np
        import torch

        audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
        if channels_first:
            audio = audio.T
        tensor = torch.from_numpy(np.ascontiguousarray(audio))
        return tensor, int(sample_rate)

    torchaudio.load = _soundfile_load  # type: ignore[assignment]

    # Current WeSpeaker imports several optional frontends at module import time.
    # WeSep's default BSRNN checkpoint uses ResNet34, so these heavy optional
    # modules are not needed and can be stubbed to avoid old s3prl/torchaudio APIs.
    speaker_cli = types.ModuleType("wespeaker.cli.speaker")

    def _speaker_cli_stub(*_args, **_kwargs):
        raise RuntimeError("wespeaker CLI helpers are not used by this WeSep wrapper")

    speaker_cli.load_model = _speaker_cli_stub
    speaker_cli.load_model_pt = _speaker_cli_stub
    sys.modules.setdefault("wespeaker.cli.speaker", speaker_cli)
    for module_name in [
        "wespeaker.models.redimnet2",
        "wespeaker.models.redimnet",
        "wespeaker.models.whisper_PMFA",
        "wespeaker.models.w2vbert_adapter_mfa",
    ]:
        sys.modules.setdefault(module_name, types.ModuleType(module_name))

    from wesep.cli.extractor import load_model, load_model_local

    model = load_model_local(args.pretrain) if args.pretrain else load_model("english")
    model.set_resample_rate(args.sample_rate)
    model.set_vad(args.vad)
    model.set_device(args.device)
    model.set_output_norm(False)
    progress_path = Path(args.progress_file) if args.progress_file else None
    speech = _extract_speech_chunked(
        model,
        Path(args.mixture),
        Path(args.reference),
        args.sample_rate,
        chunk_sec=args.chunk_sec,
        overlap_sec=args.overlap_sec,
        progress_path=progress_path,
        started=started,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), speech, args.sample_rate)
    _write_progress(
        progress_path,
        stage="wesep_done",
        progress=1.0,
        message="WeSep extraction finished",
        details={"runtime_sec": time.time() - started},
    )
    return 0


def _extract_speech_chunked(
    model,
    mixture: Path,
    reference: Path,
    sample_rate: int,
    *,
    chunk_sec: float,
    overlap_sec: float,
    progress_path: Path | None,
    started: float,
) -> np.ndarray:
    info = sf.info(str(mixture))
    total_len = int(info.frames)
    if int(info.samplerate) != int(sample_rate):
        raise RuntimeError(f"Expected {sample_rate} Hz mixture, got {info.samplerate} Hz")

    chunk_len = max(1, int(round(float(chunk_sec) * sample_rate)))
    overlap = max(0, int(round(float(overlap_sec) * sample_rate)))
    if overlap >= chunk_len:
        overlap = max(0, chunk_len // 4)
    if chunk_sec <= 0.0 or total_len <= chunk_len:
        _write_progress(
            progress_path,
            stage="wesep_extract",
            progress=0.0,
            message="Running WeSep on full file",
            details={"num_chunks": 1, "duration_sec": total_len / float(sample_rate)},
        )
        speech = model.extract_speech(str(mixture), str(reference))
        if speech is None:
            raise RuntimeError("WeSep returned no speech. Check reference VAD/model compatibility.")
        wav = speech[0].detach().cpu().numpy().astype(np.float32)
        return _match_length(wav, total_len)

    hop = max(1, chunk_len - overlap)
    starts = list(range(0, max(total_len, 1), hop))
    starts = [start for start in starts if start < total_len]
    if starts and starts[-1] + chunk_len < total_len:
        starts.append(max(0, total_len - chunk_len))
    if not starts:
        starts = [0]
    seen = set()
    starts = [start for start in starts if not (start in seen or seen.add(start))]

    output = np.zeros(total_len, dtype=np.float32)
    weight = np.zeros(total_len, dtype=np.float32)
    fade = max(1, overlap // 2)
    _write_progress(
        progress_path,
        stage="wesep_extract",
        progress=0.0,
        message="Running chunked WeSep",
        details={
            "chunk_sec": chunk_sec,
            "overlap_sec": overlap / float(sample_rate),
            "hop_sec": hop / float(sample_rate),
            "num_chunks": len(starts),
            "duration_sec": total_len / float(sample_rate),
        },
    )

    with tempfile.TemporaryDirectory(prefix="wesep_chunks_") as tmpdir:
        chunk_path = Path(tmpdir) / "mixture_chunk.wav"
        with sf.SoundFile(str(mixture), "r") as src:
            for idx, start in enumerate(starts, 1):
                src.seek(start)
                length = min(chunk_len, total_len - start)
                audio = src.read(length, dtype="float32", always_2d=False)
                if getattr(audio, "ndim", 1) > 1:
                    audio = audio.mean(axis=1).astype(np.float32)
                sf.write(str(chunk_path), audio, sample_rate, subtype="PCM_16")
                speech = model.extract_speech(str(chunk_path), str(reference))
                if speech is None:
                    raise RuntimeError(f"WeSep returned no speech for chunk {idx}/{len(starts)}")
                wav = speech[0].detach().cpu().numpy().astype(np.float32)
                wav = _match_length(wav, length)
                window = _overlap_window(length, fade)
                end = start + length
                output[start:end] += wav * window
                weight[start:end] += window
                _write_progress(
                    progress_path,
                    stage="wesep_extract",
                    progress=idx / float(len(starts)),
                    message=f"Processed WeSep chunk {idx}/{len(starts)}",
                    details={
                        "chunk_current": idx,
                        "chunk_total": len(starts),
                        "chunk_start_sec": start / float(sample_rate),
                        "chunk_duration_sec": length / float(sample_rate),
                        "runtime_sec": time.time() - started,
                    },
                )

    active = weight > 1e-8
    output[active] /= weight[active]
    return output.astype(np.float32)


def _overlap_window(length: int, fade: int) -> np.ndarray:
    window = np.ones(length, dtype=np.float32)
    fade = min(max(0, fade), length // 2)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, endpoint=False, dtype=np.float32)
        window[:fade] = np.maximum(ramp, 1e-4)
        window[-fade:] = np.maximum(ramp[::-1], 1e-4)
    return window


def _match_length(audio: np.ndarray, length: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) == length:
        return audio
    if len(audio) > length:
        return audio[:length]
    return np.pad(audio, (0, length - len(audio))).astype(np.float32)


def _write_progress(progress_path: Path | None, *, stage: str, progress: float, message: str, details: dict) -> None:
    if not progress_path:
        return
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "progress": float(max(0.0, min(1.0, progress))),
        "message": message,
        "details": details,
        "updated_at": time.time(),
    }
    temp_path = progress_path.with_suffix(progress_path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp_path.replace(progress_path)


if __name__ == "__main__":
    raise SystemExit(main())
