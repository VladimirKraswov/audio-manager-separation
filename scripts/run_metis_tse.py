#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import site
import sys
import types
from pathlib import Path

import soundfile as sf


def _install_langsegment_compat() -> None:
    """Expose the LangSegment names expected by Amphion when package init is stale."""
    if "LangSegment" in sys.modules:
        return

    search_paths = [Path(path) for path in site.getsitepackages()]
    user_site = site.getusersitepackages()
    if user_site:
        search_paths.append(Path(user_site))
    search_paths.extend(Path(path) for path in sys.path if path)

    impl_path = next(
        (path / "LangSegment" / "LangSegment.py" for path in search_paths if (path / "LangSegment" / "LangSegment.py").exists()),
        None,
    )
    if impl_path is None:
        return

    spec = importlib.util.spec_from_file_location("_langsegment_impl", impl_path)
    if spec is None or spec.loader is None:
        return
    impl = importlib.util.module_from_spec(spec)
    sys.modules["_langsegment_impl"] = impl
    spec.loader.exec_module(impl)

    shim = types.ModuleType("LangSegment")
    shim.__file__ = str(impl_path)
    shim.__path__ = [str(impl_path.parent)]
    for name in ("LangSegment", "setfilters", "getfilters", "getTexts", "getCounts", "classify", "printList"):
        if hasattr(impl, name):
            setattr(shim, name, getattr(impl, name))
    if hasattr(impl, "setfilters"):
        shim.setLangfilters = impl.setfilters
    if hasattr(impl, "getfilters"):
        shim.getLangfilters = impl.getfilters
    sys.modules["LangSegment"] = shim
    sys.modules["LangSegment.LangSegment"] = impl


def main() -> int:
    parser = argparse.ArgumentParser(description="Single-file Amphion/Metis TSE wrapper.")
    parser.add_argument("--mixture", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amphion-root", default=os.environ.get("AMPHION_ROOT", "external/Amphion"))
    parser.add_argument("--n-timesteps", type=int, default=10)
    parser.add_argument("--cfg", type=float, default=0.0)
    args = parser.parse_args()

    mixture_path = Path(args.mixture).resolve()
    reference_path = Path(args.reference).resolve()
    output_path = Path(args.output).resolve()
    amphion_root = Path(args.amphion_root).resolve()
    if not amphion_root.exists():
        raise RuntimeError(f"Amphion root not found: {amphion_root}")
    sys.path.insert(0, str(amphion_root))
    os.chdir(str(amphion_root))
    _install_langsegment_compat()

    from huggingface_hub import snapshot_download
    from models.tts.metis.metis import Metis
    from utils.util import load_config

    ckpt_dir = amphion_root / "models" / "tts" / "metis" / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    base_ckpt_dir = snapshot_download(
        "amphion/metis",
        repo_type="model",
        local_dir=str(ckpt_dir),
        allow_patterns=["metis_base/model.safetensors"],
    )
    lora_ckpt_dir = snapshot_download(
        "amphion/metis",
        repo_type="model",
        local_dir=str(ckpt_dir),
        allow_patterns=["metis_tse/metis_tse_lora_32.safetensors"],
    )
    adapter_ckpt_dir = snapshot_download(
        "amphion/metis",
        repo_type="model",
        local_dir=str(ckpt_dir),
        allow_patterns=["metis_tse/metis_tse_lora_32_adapter.safetensors"],
    )

    cfg = load_config(str(amphion_root / "models" / "tts" / "metis" / "config" / "tse.json"))
    metis = Metis(
        base_ckpt_path=os.path.join(base_ckpt_dir, "metis_base/model.safetensors"),
        lora_ckpt_path=os.path.join(lora_ckpt_dir, "metis_tse/metis_tse_lora_32.safetensors"),
        adapter_ckpt_path=os.path.join(adapter_ckpt_dir, "metis_tse/metis_tse_lora_32_adapter.safetensors"),
        cfg=cfg,
        device=args.device,
        model_type="tse",
    )

    speech = metis(
        prompt_speech_path=str(reference_path),
        source_speech_path=str(mixture_path),
        cfg=args.cfg,
        n_timesteps=args.n_timesteps,
        model_type="tse",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), speech, 24000)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
