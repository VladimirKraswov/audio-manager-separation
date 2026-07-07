#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import types

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
    args = parser.parse_args()

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
    speech = model.extract_speech(args.mixture, args.reference)
    if speech is None:
        raise RuntimeError("WeSep returned no speech. Check reference VAD/model compatibility.")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    wav = speech[0].detach().cpu().numpy()
    sf.write(str(output), wav, args.sample_rate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
