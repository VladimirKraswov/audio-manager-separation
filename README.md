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
  --quality max \
  --processing-sample-rate 16000 \
  --tse-chunk-sec 25 \
  --tse-overlap-sec 4
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
- `POST /v1/jobs-dual` - upload call mix + separate manager mic + optional manager reference.
- `GET /v1/jobs/{job_id}` - check status.
- `GET /v1/jobs/{job_id}/artifacts/client` - download `client_audio.wav` for dual jobs.
- `GET /v1/jobs/{job_id}/artifacts/speech` - download `manager_speech_clean.wav`.
- `GET /v1/jobs/{job_id}/artifacts/noise` - download `manager_noise_residual.wav`.
- `GET /v1/jobs/{job_id}/artifacts.zip` - build and download all artifacts on demand.

Russian API documentation: `docs/service_api_ru.md`.

## Dual Input Mode

Use this mode when you have both a common call mix and a separate manager
microphone track:

```bash
python process_dual_input.py \
  --mix input/call_mix.wav \
  --manager-mic input/manager_mic.wav \
  --reference input/manager_reference_clean.wav \
  --outdir output_dual \
  --device cuda:0 \
  --quality max \
  --models wesep \
  --disable-fallback
```

The dual pipeline writes `client_audio.wav`, `manager_speech_clean.wav`, and
`manager_noise_residual.wav`. It first uses the manager mic as a reference to
cancel the manager side from the call mix, then removes the rough client track
from the manager mic before running the existing target-speaker pipeline.

## Long Files

Real WeSep inference is chunked by default:

```text
processing_sample_rate=16000
tse_chunk_sec=25
tse_overlap_sec=4
```

The service decodes/resamples speech work to 16 kHz, then the wrapper loads
WeSep once, processes each chunk with the same manager reference, and stitches
the output with overlap-add. This keeps VRAM bounded for long recordings, while
16 kHz processing keeps the resulting WAV artifacts smaller than full-rate
48 kHz output. The API exposes live chunk progress through
`GET /v1/jobs/{job_id}`. On the 1:47:57 `long_test.mp3` benchmark,
25-second chunks on RTX 3060 used about 3.8 GB VRAM, processed the TSE stage at
about 23x realtime, and completed the full service pipeline in about
702 seconds.

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

Dual-input runs additionally write:

```text
output/client_audio.wav
output/client_audio_raw_iter0.wav
output/manager_mic_no_client_leak.wav
output/manager_side_estimate_in_mix.wav
output/client_leak_estimate_in_manager_mic.wav
output/dual_prepare_report.json
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
