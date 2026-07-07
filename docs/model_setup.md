# Model Setup Notes

The repository ships with command-template adapters so the pipeline can call
real TSE/enhancement systems without baking one research repo into this project.
Copy `env.tse.example` to `env.tse`, then source it from the project root before
running `process_call.py`.

```bash
cp env.tse.example env.tse
source env.tse
```

The selected project wrapper is WeSep:

```bash
export WESEP_TSE_CMD='python scripts/run_wesep_tse.py --mixture {mixture} --reference {reference} --output {output} --sample-rate {sample_rate} --device {device}'
```

## WeSep

Install WeSep from its official repository under `external/wesep`, install the
package in the main `.venv`, and install WeSpeaker because the pretrained WeSep
model depends on its speaker encoder definitions.

The placeholders available to every command template are:

- `{mixture}`
- `{reference}`
- `{output}`
- `{sample_rate}`
- `{device}`

## Residual Noise Track

The selected speech chain now keeps the intermediate stages:

- `manager_speech_tse_raw.wav` - direct selected TSE output.
- `manager_speech_tse_aligned.wav` - delay-aligned TSE output.
- `manager_speech_tse_gainmatched.wav` - aligned and loudness-matched to active
  input speech-like regions.
- `manager_speech_clean.wav` - post-enhanced listening speech.

By default speech loudness uses `input_matched`, not a fixed -23 dBFS target.
When `pyloudnorm` is installed, active loudness uses BS.1770/LUFS; otherwise the
pipeline falls back to active RMS dBFS. The final residual noise track,
`manager_noise_residual.wav`, is built by strongly suppressing bins that look
like the cleaned manager voice and is lifted toward -45 dBFS. The direct
subtraction audit file is preserved as `manager_noise_residual_subtract.wav`.

In `quality=max`, DSP fallback is disabled unless `--allow-fallback` is passed.
If DeepFilterNet is not installed, the pipeline records a warning and uses the
built-in postprocess; pass `--require-deepfilternet` to fail instead.

## Smoke Tests

```bash
python process_call.py --input input/manager_mic_mono.wav --reference input/manager_reference_clean.wav --outdir output_wesep_test --device cuda:0 --quality smoke --models wesep --disable-fallback --chunk-sec 6 --overlap-sec 1
```

## DeepFilterNet

If `deepFilter` is on `PATH`, `process_call.py` will try to use it. Otherwise it
falls back to a mild high-pass/limiter pass so the smoke test remains runnable.
You can also provide a custom command:

```bash
export DEEPFILTERNET_CMD='deepFilter {input} --output {output}'
```
