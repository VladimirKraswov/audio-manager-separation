#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch
import yaml
from huggingface_hub import snapshot_download


def _to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Single-file ClearVoice SpEx+ audio-only TSE wrapper.")
    parser.add_argument("--mixture", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-rate", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--clearvoice-root",
        default=os.environ.get(
            "CLEARVOICE_TSE_ROOT",
            "external/ClearerVoice-Studio/train/target_speaker_extraction",
        ),
    )
    parser.add_argument(
        "--repo",
        default="alibabasglab/log_wsj0-2mix_speech_SpEx-plus_2spk",
        help="HuggingFace repo containing ClearVoice SpEx+ checkpoint",
    )
    args = parser.parse_args()

    project_root = Path.cwd()
    sys.path.insert(0, str(project_root))
    clearvoice_root = (project_root / args.clearvoice_root).resolve()
    if not clearvoice_root.exists():
        raise RuntimeError(f"ClearVoice TSE root not found: {clearvoice_root}")

    local_ckpt_root = project_root / "external" / "clearvoice_checkpoints"
    snapshot_dir = snapshot_download(
        args.repo,
        repo_type="model",
        local_dir=str(local_ckpt_root),
        allow_patterns=[
            "checkpoints/log_wsj0-2mix_speech_SpEx-plus_2spk/config.yaml",
            "checkpoints/log_wsj0-2mix_speech_SpEx-plus_2spk/last_best_checkpoint.pt",
        ],
    )
    checkpoint_dir = (
        Path(snapshot_dir)
        / "checkpoints"
        / "log_wsj0-2mix_speech_SpEx-plus_2spk"
    )
    config_path = checkpoint_dir / "config.yaml"
    checkpoint_path = checkpoint_dir / "last_best_checkpoint.pt"

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    config.update(
        {
            "device": device,
            "distributed": False,
            "world_size": 1,
            "local_rank": 0,
            "use_cuda": int(device.type == "cuda"),
            "checkpoint_dir": str(checkpoint_dir),
            "train_from_last_checkpoint": 0,
            "evaluate_only": 1,
            "audio_sr": int(args.sample_rate),
            "ref_sr": int(args.sample_rate),
        }
    )
    cv_args = _to_namespace(config)

    sys.path.insert(0, str(clearvoice_root))
    os.chdir(str(clearvoice_root))
    from networks import network_wrapper

    model = network_wrapper(cv_args).to(device)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    pretrained = checkpoint["model"]
    state = model.state_dict()
    for key in list(state.keys()):
        if key in pretrained and state[key].shape == pretrained[key].shape:
            state[key] = pretrained[key]
        elif f"module.{key}" in pretrained and state[key].shape == pretrained[f"module.{key}"].shape:
            state[key] = pretrained[f"module.{key}"]
    model.load_state_dict(state)
    model.eval()

    from src.audio_io import read_audio, write_wav

    mixture, sr = read_audio(project_root / args.mixture if not Path(args.mixture).is_absolute() else args.mixture, target_sr=args.sample_rate, mono=True)
    reference, _ = read_audio(project_root / args.reference if not Path(args.reference).is_absolute() else args.reference, target_sr=args.sample_rate, mono=True)

    mix_tensor = torch.from_numpy(mixture.astype(np.float32)).unsqueeze(0).to(device)
    aux_tensor = torch.from_numpy(reference.astype(np.float32)).unsqueeze(0)
    aux_len = torch.tensor([reference.shape[0]], dtype=torch.long)
    speakers = torch.tensor([-1], dtype=torch.long)

    with torch.no_grad():
        estimate = model(mix_tensor, (aux_tensor, aux_len, speakers))
    if isinstance(estimate, (tuple, list)):
        estimate = estimate[0]
    estimate_np = estimate.squeeze(0).detach().cpu().numpy().astype(np.float32)
    output_path = project_root / args.output if not Path(args.output).is_absolute() else Path(args.output)
    write_wav(output_path, estimate_np, sr, subtype="PCM_16")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
