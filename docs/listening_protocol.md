# Real-World Listening Protocol

Create 20-50 short fragments from real calls and score each output manually.

Suggested columns:

- file
- fragment_start_sec
- fragment_end_sec
- manager_speech_clarity: 1-5
- background_left_in_speech: 1-5
- manager_leakage_in_residual: 1-5
- residual_preserves_background: 1-5
- neural_artifacts: 1-5
- confidence: high/medium/low
- notes

Use `manager_speech_tse_raw.wav` to judge separation and
`manager_speech_clean.wav` to judge perceptual quality. Judge
`manager_noise_residual.wav` separately: it is supposed to preserve TV, dog
barks, music, background voices, and room noise.
