#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src.audio_io import match_length, peak_limit, read_audio, write_wav
from src.scoring import si_sdr


NOISE_DIRS = [
    "noise_tv",
    "noise_dog",
    "noise_music",
    "noise_background_speech",
    "noise_household",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthetic inputs and benchmark helpers.")
    parser.add_argument("--make-smoke-inputs", action="store_true", help="Create input manager_mic/reference WAVs")
    parser.add_argument("--generate", action="store_true", help="Generate synthetic benchmark mixtures")
    parser.add_argument("--benchmark-dir", default="benchmark")
    parser.add_argument("--outdir", default="input", help="Target folder for smoke inputs")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir)
    ensure_benchmark_dirs(benchmark_dir)

    if args.make_smoke_inputs:
        create_smoke_inputs(Path(args.outdir), args.sample_rate, args.duration)
    if args.generate:
        generate_benchmark(benchmark_dir, args.sample_rate, args.count, args.duration)
    if not args.make_smoke_inputs and not args.generate:
        parser.print_help()
    return 0


def ensure_benchmark_dirs(root: Path) -> None:
    for folder in ["clean_manager", *NOISE_DIRS, "generated_mixes"]:
        (root / folder).mkdir(parents=True, exist_ok=True)


def create_smoke_inputs(outdir: Path, sr: int, duration: float) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    reference = synthetic_manager_voice(sr, min(duration, 20.0), seed=42)
    clean = synthetic_manager_voice(sr, duration, seed=7)
    noise = 0.35 * synthetic_tv_noise(sr, duration, seed=3)
    noise += 0.20 * synthetic_household_noise(sr, duration, seed=4)
    mixture = peak_limit(clean + noise)
    write_wav(outdir / "manager_reference_clean.wav", reference, sr, subtype="PCM_16")
    write_wav(outdir / "manager_mic_mono.wav", mixture, sr, subtype="PCM_16")
    print(f"Created smoke inputs in {outdir}")


def generate_benchmark(root: Path, sr: int, count: int, duration: float) -> None:
    generated = root / "generated_mixes"
    rows = []
    snrs = [-10, -5, 0, 5, 10]
    for idx in range(count):
        snr_db = snrs[idx % len(snrs)]
        clean = synthetic_manager_voice(sr, duration, seed=1000 + idx)
        noise = synthetic_noise_combo(sr, duration, seed=2000 + idx)
        mixture = mix_at_snr(clean, noise, snr_db)
        clip_dir = generated / f"clip_{idx:04d}_snr_{snr_db:+d}db"
        clip_dir.mkdir(parents=True, exist_ok=True)
        write_wav(clip_dir / "clean_target.wav", clean, sr, subtype="PCM_16")
        write_wav(clip_dir / "true_noise.wav", noise, sr, subtype="PCM_16")
        write_wav(clip_dir / "mixture.wav", mixture, sr, subtype="PCM_16")
        rows.append(
            {
                "clip": clip_dir.name,
                "snr_db": snr_db,
                "mixture_si_sdr_vs_clean": si_sdr(mixture, clean),
                "duration_sec": duration,
            }
        )

    results_path = root / "results.csv"
    with results_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = root / "summary.md"
    avg = sum(row["mixture_si_sdr_vs_clean"] for row in rows) / max(len(rows), 1)
    summary.write_text(
        "\n".join(
            [
                "# Synthetic Benchmark Summary",
                "",
                f"Generated clips: {count}",
                f"Sample rate: {sr}",
                f"Duration per clip: {duration:.2f} sec",
                f"Average mixture SI-SDR vs clean target: {avg:.2f} dB",
                "",
                "Run `process_call.py` on generated mixtures, then extend `results.csv` with model outputs.",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Generated benchmark in {generated}")


def synthetic_manager_voice(sr: int, duration: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * duration), dtype=np.float32) / float(sr)
    f0 = 115.0 + rng.uniform(-10.0, 10.0)
    vibrato = 1.0 + 0.015 * np.sin(2.0 * np.pi * 5.0 * t)
    voice = (
        0.45 * np.sin(2.0 * np.pi * f0 * vibrato * t)
        + 0.25 * np.sin(2.0 * np.pi * 2.0 * f0 * vibrato * t)
        + 0.12 * np.sin(2.0 * np.pi * 3.0 * f0 * vibrato * t)
    )
    envelope = speech_like_envelope(sr, duration, rng)
    formant = 0.65 + 0.35 * np.sin(2.0 * np.pi * 2.3 * t + 0.4)
    return peak_limit((voice * envelope * formant).astype(np.float32), ceiling=0.75)


def synthetic_tv_noise(sr: int, duration: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * duration), dtype=np.float32) / float(sr)
    chatter = 0.18 * synthetic_manager_voice(sr, duration, seed + 50)
    hum = 0.05 * np.sin(2.0 * np.pi * 60.0 * t)
    hiss = 0.08 * rng.normal(size=len(t)).astype(np.float32)
    music = 0.12 * np.sin(2.0 * np.pi * (220.0 + 20.0 * np.sin(2 * np.pi * 0.4 * t)) * t)
    return peak_limit(chatter + hum + hiss + music, ceiling=0.9)


def synthetic_household_noise(sr: int, duration: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(sr * duration)
    noise = 0.06 * rng.normal(size=n).astype(np.float32)
    for _ in range(max(1, int(duration))):
        start = int(rng.integers(0, max(1, n - sr // 10)))
        length = int(rng.integers(sr // 100, sr // 6))
        burst = np.hanning(length).astype(np.float32) * rng.uniform(0.1, 0.35)
        noise[start : start + length] += burst[: max(0, min(length, n - start))]
    return peak_limit(noise, ceiling=0.9)


def synthetic_noise_combo(sr: int, duration: float, seed: int) -> np.ndarray:
    return peak_limit(
        0.65 * synthetic_tv_noise(sr, duration, seed)
        + 0.45 * synthetic_household_noise(sr, duration, seed + 1),
        ceiling=0.95,
    )


def speech_like_envelope(sr: int, duration: float, rng: np.random.Generator) -> np.ndarray:
    n = int(sr * duration)
    env = np.zeros(n, dtype=np.float32)
    pos = 0
    while pos < n:
        silence = int(rng.uniform(0.05, 0.35) * sr)
        speech = int(rng.uniform(0.35, 1.5) * sr)
        pos += silence
        end = min(n, pos + speech)
        if end > pos:
            ramp = min(sr // 20, max(1, (end - pos) // 3))
            env[pos:end] = 1.0
            env[pos : pos + ramp] *= np.linspace(0.0, 1.0, ramp, dtype=np.float32)
            env[end - ramp : end] *= np.linspace(1.0, 0.0, ramp, dtype=np.float32)
        pos = end
    return env


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    clean = match_length(clean, min(len(clean), len(noise)))
    noise = match_length(noise, len(clean))
    clean_power = float(np.mean(clean.astype(np.float64) ** 2) + 1e-12)
    noise_power = float(np.mean(noise.astype(np.float64) ** 2) + 1e-12)
    target_noise_power = clean_power / (10.0 ** (snr_db / 10.0))
    scaled_noise = noise * np.sqrt(target_noise_power / noise_power)
    return peak_limit(clean + scaled_noise, ceiling=0.98)


if __name__ == "__main__":
    raise SystemExit(main())
