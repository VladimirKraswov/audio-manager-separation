# audio_manager_separation

Офлайн-пайплайн и HTTP-сервис для отделения голоса менеджера от фонового аудио.
Основная идея: сначала получить целевую речь менеджера через TSE
(`target speaker extraction`), затем построить остаточный шумовой трек и
дочистить обе дорожки спектральными фильтрами.

```text
input/manager_mic_mono.wav + input/manager_reference_clean.wav
        -> сырой TSE-голос менеджера
        -> delay alignment и loudness matching
        -> residual/subtract и spectral masks
        -> чистый голос менеджера + шумовой остаток
```

Главные выходные файлы:

- `output/manager_speech_clean.wav` - финальный чистый голос менеджера;
- `output/manager_noise_residual.wav` - фон/шум с подавленным менеджером;
- `output/manager_speech_tse_raw.wav` - сырой вывод WeSep/TSE;
- `output/manager_speech_clean_prefilter.wav` - речь до финального residual-фильтра;
- `output/manager_noise_residual_prefilter.wav` - шум до финального подавления следов менеджера;
- `output/report.json` - полный отчёт с настройками, метриками и предупреждениями.

## Установка

Минимальный smoke-тест требует только Python и базовые зависимости:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Для GPU-инференса установите PyTorch с CUDA wheel, подходящим под драйвер
сервера, затем установите WeSep и дополнительные модели. Необязательные
пакеты для расширенных аудио/model-интеграций перечислены в
`requirements-optional.txt`.

Вариант через Conda:

```bash
conda env create -f environment.yml
conda activate audio-manager-separation
```

## Быстрый smoke-тест

Сгенерировать синтетический вход и reference:

```bash
python benchmark.py --make-smoke-inputs --outdir input --sample-rate 16000 --duration 12
```

Запустить пайплайн:

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

Если реальные TSE-команды не настроены, пайплайн может использовать встроенный
резервный DSP-режим (`fallback`). Он нужен только для проверки файлового потока,
alignment, residual-генерации и `report.json`; для продакшена это не замена
WeSep.

## Запуск на одном реальном файле

Положите файлы:

```text
input/manager_mic_mono.wav
input/manager_reference_clean.wav
```

Запустите:

```bash
python process_call.py \
  --input input/manager_mic_mono.wav \
  --reference input/manager_reference_clean.wav \
  --outdir output \
  --device cuda:0 \
  --quality max \
  --models wesep \
  --disable-fallback \
  --processing-sample-rate 16000 \
  --tse-chunk-sec 25 \
  --tse-overlap-sec 4
```

`--processing-sample-rate 16000` уменьшает потребление памяти и размер WAV для
длинных звонков. Если нужно сохранить исходную частоту дискретизации, передайте
`--processing-sample-rate 0`, но для длинных файлов это заметно тяжелее.

## Настройка реального WeSep/TSE

Адаптеры вызывают внешние модели через command templates. Скопируйте шаблон:

```bash
cp env.tse.example env.tse
source env.tse
```

Основная команда WeSep:

```bash
export WESEP_TSE_CMD='python scripts/run_wesep_tse.py --mixture {mixture} --reference {reference} --output {output} --sample-rate {sample_rate} --device {device}'
```

Доступные placeholders:

- `{mixture}` - подготовленный входной микс;
- `{reference}` - reference-голос менеджера;
- `{output}` - путь для результата модели;
- `{sample_rate}` - рабочая частота дискретизации;
- `{device}` - `cuda:0`, `cuda:1` или `cpu`.

Подробности установки моделей: [docs/model_setup.md](docs/model_setup.md).

## HTTP-сервис

Запустить API:

```bash
python service.py --host 0.0.0.0 --port 8088
```

Основные ручки:

- `GET /health` - состояние сервиса и наличие WeSep/DeepFilterNet;
- `GET /v1/defaults` - текущие настройки по умолчанию;
- `POST /v1/jobs` - загрузить аудио и опциональный reference менеджера;
- `POST /v1/jobs-dual` - загрузить общий микс и отдельный микрофон менеджера;
- `GET /v1/jobs/{job_id}` - статус и прогресс задачи;
- `GET /v1/jobs/{job_id}/artifacts` - список доступных артефактов;
- `GET /v1/jobs/{job_id}/artifacts/speech` - скачать `manager_speech_clean.wav`;
- `GET /v1/jobs/{job_id}/artifacts/noise` - скачать `manager_noise_residual.wav`;
- `GET /v1/jobs/{job_id}/artifacts.zip` - собрать и скачать архив всех артефактов.

Полная документация API: [docs/service_api_ru.md](docs/service_api_ru.md).

## Dual-input режим

Используйте этот режим, если есть два файла:

```text
call_mix.wav      - общий микс: клиент + менеджер + фон
manager_mic.wav   - отдельный микрофон менеджера
```

Запуск из CLI:

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

Dual-input сначала выравнивает общий микс и микрофон менеджера, затем использует
эту дорожку как опорный сигнал для удаления менеджера из общего микса. После
этого грубая оценка клиента удаляется из микрофона менеджера, и уже очищенный
сигнал отправляется в тот же TSE-пайплайн. На выходе появляются
`client_audio.wav`, `manager_speech_clean.wav` и `manager_noise_residual.wav`.

## Длинные файлы

Для длинных записей WeSep работает чанками по умолчанию:

```text
processing_sample_rate=16000
tse_chunk_sec=25
tse_overlap_sec=4
```

Wrapper загружает WeSep один раз, обрабатывает каждый чанк с тем же reference
менеджера и склеивает результат через overlap-add. Это удерживает VRAM в рамках
выбранного размера чанка. Рабочая частота `16 kHz` уменьшает размер итоговых WAV
и память CPU-постфильтров.

Прогресс виден через `GET /v1/jobs/{job_id}`. Во время WeSep-инференса сервис
возвращает `stage=wesep_extract`, `chunk_current`, `chunk_total` и процент
готовности.

Проверенный benchmark на `long_test.mp3`:

```text
длительность: 1:47:57 / 6477 sec
GPU: RTX 3060
tse_chunk_sec: 25
tse_overlap_sec: 4
peak VRAM WeSep: ~3.8 GB
скорость WeSep stage: ~23x realtime
полный сервисный пайплайн: ~702 sec / ~9.2x realtime
```

Если нужно снизить VRAM ещё сильнее, поставьте `tse_chunk_sec=15`; качество
стыков обычно остаётся нормальным, но чанков становится больше.

## Бенчмарк

Сгенерировать синтетические смеси:

```bash
python benchmark.py --generate --benchmark-dir benchmark --count 200 --duration 8
```

Будут созданы:

- `benchmark/generated_mixes/*/mixture.wav`;
- `benchmark/generated_mixes/*/clean_target.wav`;
- `benchmark/generated_mixes/*/true_noise.wav`;
- `benchmark/results.csv`;
- `benchmark/summary.md`.

Оценить обработанный фрагмент:

```bash
python evaluate.py \
  --input benchmark/generated_mixes/clip_0000_snr_-10db/mixture.wav \
  --reference input/manager_reference_clean.wav \
  --speech output/manager_speech_clean.wav \
  --residual output/manager_noise_residual.wav \
  --clean-target benchmark/generated_mixes/clip_0000_snr_-10db/clean_target.wav
```

Для ручной оценки качества используйте протокол:
[docs/listening_protocol.md](docs/listening_protocol.md).

## Контракт выходных файлов

Каждый single-input запуск пишет:

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

Dual-input дополнительно пишет:

```text
output/client_audio.wav
output/client_audio_raw_iter0.wav
output/manager_mic_no_client_leak.wav
output/manager_side_estimate_in_mix.wav
output/client_leak_estimate_in_manager_mic.wav
output/dual_prepare_report.json
```

Ключевые стадии:

- `manager_speech_tse_aligned.wav` - выбранный TSE после delay alignment;
- `manager_speech_tse_gainmatched.wav` - TSE после выравнивания громкости по активным участкам входа;
- `manager_speech_clean_prefilter.wav` - речь после speech enhancement, но до финального residual-guided фильтра;
- `manager_speech_clean.wav` - финальная дорожка речи для прослушивания;
- `manager_noise_residual_subtract.wav` - контрольный прямой subtract;
- `manager_noise_residual_prefilter.wav` - residual после базового подавления менеджера;
- `manager_noise_residual.wav` - финальный шумовой трек для прослушивания.

По умолчанию речь использует `input_matched` loudness вместо фиксированных
`-23 dBFS`: сервис измеряет активные участки входа и применяет ограниченный
gain к извлечённой речи. Финальный шумовой трек дополнительно использует
`manager_speech_clean.wav` как spectral guide, подавляет следы менеджера и
поднимается примерно к `-45 dBFS`, чтобы его было удобно слушать.

## Ограничения

- Production-качество зависит от корректной установки и проверки WeSep на
  реальных звонках.
- В `quality=max` резервный DSP-режим отключён, если явно не передать
  `--allow-fallback`.
- Если DeepFilterNet недоступен, `report.json` получает warning
  `deepfilternet_unavailable_builtin_postprocess_used`; `--require-deepfilternet`
  заставляет пайплайн падать без реального DeepFilterNet.
- Встроенные метрики качества пока приближённые. Для продакшен-решений стоит
  добавить speaker embeddings, DNSMOS/NISQA и ASR confidence.
- Прямой residual subtraction физически корректен только когда TSE-выход
  sample-aligned и phase-compatible с входом. Поэтому финальный residual для
  прослушивания использует spectral manager suppression, а не только subtract.
- Поддержка MP3/M4A/FLAC зависит от установленных optional audio I/O зависимостей.
