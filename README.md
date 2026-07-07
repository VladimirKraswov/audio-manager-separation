# audio_manager_separation

Offline pipeline for separating a manager's voice from background audio using a
target-speaker-first design:

```text
input/manager_mic_mono.wav + input/manager_reference_clean.wav
        -> TSE raw speech
        -> sample-aligned residual subtraction
        -> optional speech enhancement
```

Primary outputs:

- `output/manager_speech_tse_raw.wav`
- `output/manager_speech_clean_prefilter.wav`
- `output/manager_speech_clean.wav`
- `output/manager_noise_residual_prefilter.wav`
- `output/manager_noise_residual.wav`
- `output/report.json`

## Install

Core smoke tests need only Python and NumPy:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For RTX 3090/CUDA, install PyTorch with the CUDA wheel recommended by PyTorch
for your driver, then install the target model repositories you want to test.
Optional Python packages used by richer audio/model integrations are listed in
`requirements-optional.txt`.

Conda option:

```bash
conda env create -f environment.yml
conda activate audio-manager-separation
```

## Smoke Test

Generate synthetic input/reference audio:

```bash
python benchmark.py --make-smoke-inputs --outdir input --sample-rate 16000 --duration 12
```

Run the pipeline:

```bash
python process_call.py \
  --input input/manager_mic_mono.wav \
  --reference input/manager_reference_clean.wav \
  --outdir output \
  --device cuda:0 \
  --quality max
```

Without configured WeSep commands, the pipeline uses the
built-in DSP fallback so the file flow, alignment, residual generation, and JSON
report can be tested. The fallback is intentionally marked low confidence and is
not a production TSE model.

## Run One Real File

Place files here:

```text
input/manager_mic_mono.wav
input/manager_reference_clean.wav
```

Then run:

```bash
python process_call.py \
  --input input/manager_mic_mono.wav \
  --reference input/manager_reference_clean.wav \
  --outdir output \
  --device cuda:0 \
  --quality max
```

## Configure Real TSE Models

The adapters use command templates. Copy and source the checked-in template from
the project root:

```bash
cp env.tse.example env.tse
source env.tse
```

The template wires the selected WeSep wrapper:

```bash
export WESEP_TSE_CMD='python scripts/run_wesep_tse.py --mixture {mixture} --reference {reference} --output {output} --sample-rate {sample_rate} --device {device}'
```

Supported template placeholders:

- `{mixture}`
- `{reference}`
- `{output}`
- `{sample_rate}`
- `{device}`

## HTTP Service

Run the API service:

```bash
python service.py --host 0.0.0.0 --port 8088
```

Main endpoints:

- `POST /v1/jobs` - upload audio and optional manager reference.
- `GET /v1/jobs/{job_id}` - check status.
- `GET /v1/jobs/{job_id}/artifacts/speech` - download `manager_speech_clean.wav`.
- `GET /v1/jobs/{job_id}/artifacts/noise` - download `manager_noise_residual.wav`.
- `GET /v1/jobs/{job_id}/artifacts.zip` - download all artifacts.

Russian API documentation: `docs/service_api_ru.md`.

## Benchmark

Create synthetic mixtures:

```bash
python benchmark.py --generate --benchmark-dir benchmark --count 200 --duration 8
```

This creates:

- `benchmark/generated_mixes/*/mixture.wav`
- `benchmark/generated_mixes/*/clean_target.wav`
- `benchmark/generated_mixes/*/true_noise.wav`
- `benchmark/results.csv`
- `benchmark/summary.md`

Evaluate a processed clip:

```bash
python evaluate.py \
  --input benchmark/generated_mixes/clip_0000_snr_-10db/mixture.wav \
  --reference input/manager_reference_clean.wav \
  --speech output/manager_speech_clean.wav \
  --residual output/manager_noise_residual.wav \
  --clean-target benchmark/generated_mixes/clip_0000_snr_-10db/clean_target.wav
```

## Output Contract

Every run writes:

```text
output/original_aligned.wav
output/manager_speech_tse_raw.wav
output/manager_speech_tse_aligned.wav
output/manager_speech_tse_gainmatched.wav
output/manager_speech_clean_prefilter.wav
output/manager_speech_clean.wav
output/manager_noise_residual_raw.wav
output/manager_noise_residual_subtract.wav
output/manager_noise_residual_prefilter.wav
output/manager_noise_residual.wav
output/report.json
output/candidates/*.wav
output/references/*.wav
```

`manager_speech_tse_aligned.wav` is the selected TSE output after delay
alignment. `manager_speech_tse_gainmatched.wav` is aligned and loudness-matched
to active speech-like regions in the input. `manager_speech_clean.wav` is the
post-enhanced speech before the final residual-guided denoise pass.
`manager_speech_clean.wav` is the final listening speech; by default it uses
`input_matched` active loudness rather than a fixed -23 dBFS target, applies a
true-peak style sample peak limiter, and removes residual/background-like bins.
`manager_noise_residual_subtract.wav` is the conservative direct-subtraction
residual. `manager_noise_residual_prefilter.wav` is the residual after the base
manager suppression pass. `manager_noise_residual.wav` is the final listening
residual: it uses `manager_speech_clean.wav` as a spectral guide, adds an extra
manager-leak suppression pass, and is gently lifted toward -45 dBFS for easier
listening.

## Known Limitations

- Real production quality depends on installing and validating the selected
  WeSep runtime on representative calls.
- In `quality=max`, DSP fallback is disabled unless `--allow-fallback` is passed.
- If DeepFilterNet is unavailable, the report now records
  `deepfilternet_unavailable_builtin_postprocess_used`; pass
  `--require-deepfilternet` to fail instead.
- Built-in scoring metrics are proxies. Add SpeechBrain embeddings, DNSMOS/NISQA,
  and ASR confidence before making production decisions.
- Direct residual subtraction is only physically meaningful when the selected TSE
  output is sample-aligned and phase-compatible with the input. The final
  listening residual therefore uses manager suppression rather than only direct
  subtraction.
- MP3/M4A/FLAC support depends on installing optional audio I/O dependencies.
