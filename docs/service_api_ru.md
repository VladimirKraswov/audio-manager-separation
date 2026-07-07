# HTTP-сервис разделения голоса менеджера

Сервис оборачивает выбранный пайплайн WeSep в удобный API:

- принимает исходный шумный аудиофайл;
- принимает чистый образец голоса менеджера или сам выбирает auto-reference;
- возвращает чистый голос менеджера;
- возвращает шумовой/фоновый трек с подавленным голосом менеджера;
- сохраняет полный `report.json` и служебные артефакты.

## Быстрый запуск

```bash
cd /home/vladimir/audio_manager_separation
source .venv/bin/activate
cp env.tse.example env.tse
python service.py --host 0.0.0.0 --port 8088
```

Интерактивная OpenAPI-документация будет доступна по адресу:

```text
http://<host>:8088/docs
```

## Проверка состояния

```bash
curl http://localhost:8088/health
```

Ответ показывает, где лежат результаты и видит ли сервис команду WeSep:

```json
{
  "status": "ok",
  "project_root": "/home/vladimir/audio_manager_separation",
  "runs_root": "/home/vladimir/audio_manager_separation/service_runs",
  "max_workers": 1,
  "wesep_configured": true,
  "deepfilternet_available": false,
  "ready_for_quality_processing": false
}
```

`ready_for_quality_processing=true` означает, что настроены и реальный WeSep, и
реальный DeepFilterNet. Если DeepFilterNet не установлен, сервис всё равно может
работать, но `report.json` явно получит warning
`deepfilternet_unavailable_builtin_postprocess_used`.

## Дефолтные настройки

```bash
curl http://localhost:8088/v1/defaults
```

Текущие дефолты подобраны под тестовый файл `audio_test_noize.ogg` и выбранный
подход WeSep:

| параметр | дефолт | смысл |
|---|---:|---|
| `models` | `wesep` | использовать только выбранный WeSep-подход |
| `disable_fallback` | `true` | не подменять WeSep простым резервным DSP-режимом |
| `device` | `cuda:0` | GPU для инференса |
| `processing_sample_rate` | `16000` | рабочая частота пайплайна; `0` сохраняет sample rate входа |
| `tse_chunk_sec` | `25.0` | размер чанка для реального WeSep/TSE |
| `tse_overlap_sec` | `4.0` | overlap между TSE-чанками для гладкой склейки |
| `speech_loudness_mode` | `input_matched` | матчить громкость речи к активным участкам входа |
| `speech_target_dbfs` | `-23.0` | используется только в режиме `fixed` |
| `speech_max_gain_db` | `18.0` | максимум усиления речи |
| `speech_true_peak_db` | `-1.0` | потолок пиков речи |
| `speech_intro_duck_sec` | `2.2` | приглушение первых секунд, где часто лезут хлопки/аплодисменты |
| `speech_intro_lowpass_sec` | `6.0` | мягкая фильтрация начала чистого голоса |
| `speech_noise_filter_strength` | `0.78` | сила финального подавления residual-шумов в речи, `0` отключает |
| `speech_noise_filter_over_subtract` | `1.35` | насколько агрессивно считать residual шумом при чистке речи |
| `speech_noise_filter_floor` | `0.08` | нижняя граница маски речи, меньше = чище, но больше риск артефактов |
| `speech_noise_filter_mask_power` | `1.0` | форма маски речи, выше = резче режет шумовые bins |
| `speech_postfilter_max_gain_db` | `4.0` | максимум дополнительного усиления после финальной чистки речи |
| `residual_base_attenuation` | `0.97` | базовое подавление менеджера в шумовом треке |
| `residual_target_dbfs` | `-45.0` | громкость шумового трека |
| `residual_leak_suppression` | `0.94` | дополнительное подавление следов менеджера в шумовом треке |
| `residual_leak_mask_start_ratio` | `0.18` | порог начала suppression-маски для manager leak |
| `residual_leak_mask_full_ratio` | `0.68` | порог полной suppression-маски для manager leak |
| `residual_leak_mask_power` | `0.50` | форма маски manager leak, меньше = мягче переход |
| `require_deepfilternet` | `false` | если `true`, job падает без реального DeepFilterNet |
| `auto_reference` | `true` | если reference не передан, взять лучший 20-секундный участок из входа |
| `auto_reference_sec` | `20.0` | длина auto-reference |

`input_matched` не нормализует речь в фиксированные `-23 dBFS`. Сервис берёт
активные speech-like участки извлечённого менеджера, измеряет те же участки во
входном аудио и применяет gain, ограниченный `speech_max_gain_db` и
`speech_true_peak_db`. Если установлен `pyloudnorm`, используется BS.1770/LUFS;
иначе сервис использует active RMS dBFS.

## Создать задачу

Минимальный запрос только с исходным аудио:

```bash
curl -X POST http://localhost:8088/v1/jobs \
  -F "audio=@/Users/vladimirkrasov/Downloads/audio_test_noize.ogg"
```

В этом режиме сервис сам создаст reference из наиболее активного 20-секундного
участка. Для продакшена лучше передавать отдельный чистый reference менеджера:

```bash
curl -X POST http://localhost:8088/v1/jobs \
  -F "audio=@call.ogg" \
  -F "reference=@manager_reference.wav"
```

Пример с ручной настройкой:

```bash
curl -X POST http://localhost:8088/v1/jobs \
  -F "audio=@call.ogg" \
  -F "reference=@manager_reference.wav" \
  -F "device=cuda:0" \
  -F "speech_loudness_mode=input_matched" \
  -F "speech_max_gain_db=18" \
  -F "speech_true_peak_db=-1" \
  -F "speech_intro_duck_sec=2.0" \
  -F "speech_intro_lowpass_sec=5.0" \
  -F "speech_noise_filter_strength=0.82" \
  -F "residual_leak_suppression=0.96" \
  -F "residual_target_dbfs=-43"
```

Ответ:

```json
{
  "job_id": "3f4b...",
  "status": "queued",
  "status_url": "/v1/jobs/3f4b...",
  "report_url": "/v1/jobs/3f4b.../report",
  "artifacts_url": "/v1/jobs/3f4b.../artifacts"
}
```

## Создать dual-input задачу

Этот режим стоит использовать, если есть два файла:

```text
call_mix.wav      - общий микс: клиент + менеджер + фон
manager_mic.wav   - отдельный микрофон менеджера
```

Минимальный запрос:

```bash
curl -X POST http://localhost:8088/v1/jobs-dual \
  -F "mix_audio=@call_mix.wav" \
  -F "manager_mic_audio=@manager_mic.wav"
```

Если чистый reference менеджера есть отдельно, лучше передать его:

```bash
curl -X POST http://localhost:8088/v1/jobs-dual \
  -F "mix_audio=@call_mix.wav" \
  -F "manager_mic_audio=@manager_mic.wav" \
  -F "reference=@manager_reference.wav"
```

Ручки качества dual-input:

| параметр | дефолт | смысл |
|---|---:|---|
| `dual_cancel_method` | `hybrid` | `simple`, `adaptive_fir`, `spectral_mask`, `hybrid` |
| `dual_cancel_strength` | `1.0` | сила удаления микрофона менеджера из общего микса на первой итерации |
| `dual_spectral_strength` | `0.35` | мягкая spectral-mask дочистка клиента |
| `dual_client_leak_strength` | `0.80` | сила удаления грубого клиента из микрофона менеджера |
| `dual_final_cancel_strength` | `1.0` | финальное удаление очищенного микрофона менеджера из общего микса |
| `dual_max_delay_ms` | `3000.0` | максимум поиска задержки между файлами |
| `dual_correct_drift` | `false` | включать только для длинных файлов с заметным drift |

Главные выходы dual-input:

| ключ | файл | назначение |
|---|---|---|
| `client` | `client_audio.wav` | клиентская сторона из общего микса |
| `client_raw` | `client_audio_raw_iter0.wav` | первая грубая оценка клиента |
| `manager_mic_no_client_leak` | `manager_mic_no_client_leak.wav` | микрофон менеджера после удаления утечки клиента |
| `manager_side_estimate` | `manager_side_estimate_in_mix.wav` | оценка стороны менеджера в общем миксе |
| `speech` | `manager_speech_clean.wav` | чистая речь менеджера |
| `noise` | `manager_noise_residual.wav` | шумы на стороне менеджера |
| `report` | `report.json` | полный dual-input отчёт |

## Статус задачи

```bash
curl http://localhost:8088/v1/jobs/<job_id>
```

Во время обработки `progress` показывает текущую стадию. Для длинных файлов в
момент WeSep-инференса там будут поля:

```json
{
  "stage": "wesep_extract",
  "progress": 0.42,
  "message": "Processed WeSep chunk 130/309",
  "details": {
    "chunk_current": 130,
    "chunk_total": 309,
    "chunk_start_sec": 2709.0
  }
}
```

Статусы:

- `queued` - задача поставлена в очередь;
- `running` - идёт обработка;
- `succeeded` - артефакты готовы;
- `failed` - ошибка, поле `error` содержит причину.

## Скачать артефакты

Список доступных файлов:

```bash
curl http://localhost:8088/v1/jobs/<job_id>/artifacts
```

Основные файлы:

| ключ | файл | назначение |
|---|---|---|
| `speech` | `manager_speech_clean.wav` | чистый голос менеджера, нормализован по громкости |
| `speech_prefilter` | `manager_speech_clean_prefilter.wav` | речь до финального residual-guided фильтра |
| `noise` | `manager_noise_residual.wav` | фон/шум с подавленным менеджером |
| `noise_prefilter` | `manager_noise_residual_prefilter.wav` | шум до дополнительного подавления manager leak |
| `noise_subtract` | `manager_noise_residual_subtract.wav` | контрольный прямой subtract |
| `raw_speech` | `manager_speech_tse_raw.wav` | сырой вывод WeSep |
| `aligned_speech` | `manager_speech_tse_aligned.wav` | TSE после delay alignment |
| `gainmatched_speech` | `manager_speech_tse_gainmatched.wav` | TSE после gain/loudness matching |
| `original` | `original_aligned.wav` | исходник после выравнивания формата |
| `report` | `report.json` | полный отчёт с метриками и настройками |

Скачать чистый голос:

```bash
curl -L http://localhost:8088/v1/jobs/<job_id>/artifacts/speech \
  -o manager_speech_clean.wav
```

Скачать шум:

```bash
curl -L http://localhost:8088/v1/jobs/<job_id>/artifacts/noise \
  -o manager_noise_residual.wav
```

Скачать всё архивом:

```bash
curl -L http://localhost:8088/v1/jobs/<job_id>/artifacts.zip \
  -o artifacts.zip
```

Для длинных файлов архив может быть большим. Сервис не собирает его заранее:
`artifacts.zip` создаётся лениво при первом запросе, а статус задачи становится
`succeeded` сразу после готовности WAV/JSON в `output/`.

## Удалить задачу

```bash
curl -X DELETE http://localhost:8088/v1/jobs/<job_id>
```

Удаление доступно только для завершённых задач. Запущенную задачу сервис не
прерывает через API.

## Практические рекомендации

- Если в чистом голосе слышны аплодисменты в начале, увеличьте
  `speech_intro_duck_sec` до `2.5-3.0`.
- Если начало голоса становится слишком глухим, уменьшите
  `speech_intro_lowpass_sec` или поставьте `0`.
- Если шумовой трек слишком тихий, поднимите `residual_target_dbfs`, например
  с `-45` до `-43`.
- Если чистый голос всё ещё слишком тихий, сначала оставьте
  `speech_loudness_mode=input_matched`, но увеличьте `speech_max_gain_db`.
- Если в чистом голосе ещё слышен фон, поднимите
  `speech_noise_filter_strength` до `0.85-0.92`; если появляются водянистые
  артефакты, верните ближе к `0.70` или поднимите `speech_noise_filter_floor`.
- Если в шумовом треке ещё слышен менеджер, поднимите
  `residual_leak_suppression` до `0.96-0.99` или снизьте
  `residual_leak_mask_start_ratio` до `0.12-0.15`.
- Если нужна старая фиксированная громкость, используйте
  `speech_loudness_mode=fixed` и `speech_target_dbfs=-23`.
- Если в шумовом треке всё ещё слышен менеджер, лучше сначала улучшить образец
  голоса. Auto-reference удобен для черновой обработки, но отдельный чистый
  reference обычно даёт более стабильное разделение.
- Если есть общий микс и отдельный микрофон менеджера, используйте
  `/v1/jobs-dual`: это качественнее для `client_audio.wav`, потому что микрофон
  менеджера становится полноценной опорной дорожкой для cancellation.
- Для длинных файлов оставляйте `tse_chunk_sec=25` и `tse_overlap_sec=4`.
  На RTX 3060 это дало около `3.8 GB` peak VRAM на файле `1:47:57`. Если нужно
  ещё ниже по VRAM, можно поставить `tse_chunk_sec=15`, но стыков будет больше.
- Для звонков и длинных записей оставляйте `processing_sample_rate=16000`.
  Это сильно уменьшает размер артефактов и память постфильтров. `0` имеет смысл
  только если нужно сохранить исходную частоту дискретизации.
- Если в `client_audio.wav` остаётся менеджер, поднимите
  `dual_spectral_strength` до `0.45-0.60`; если клиент становится водянистым,
  верните ближе к `0.25-0.35`.
- Если в `manager_noise_residual.wav` слышен клиент из колонок менеджера,
  поднимите `dual_client_leak_strength` до `0.9`.

## Ограничения текущей версии

Сервис уже не использует fixed `-23 dBFS` как единственный режим и явно пишет в
отчёт, когда вместо модели используется резервная обработка или отсутствует
DeepFilterNet. Но несколько улучшений оставлены как следующий этап:

- speaker embeddings ECAPA-TDNN для настоящей проверки “это менеджер или нет”;
- YAMNet/AudioSet для детекции аплодисментов, музыки, лая и фоновой речи;
- multi-reference прогон `ref_best_10s/ref_best_20s/ref_full` с выбором лучшего;
- замена proxy spectral scoring на speaker-verification scoring.

## Где лежат результаты

По умолчанию:

```text
service_runs/<job_id>/
  input/
  output/
    manager_speech_clean_prefilter.wav
    manager_speech_clean.wav
    manager_noise_residual_prefilter.wav
    manager_noise_residual.wav
    manager_noise_residual_subtract.wav
    report.json
  artifacts.zip              # появляется только после запроса /artifacts.zip
  job.json
```

Папку можно изменить переменной окружения:

```bash
export AMS_RUNS_DIR=/data/audio_manager_jobs
```

Количество параллельных задач:

```bash
export AMS_MAX_WORKERS=1
```

Для GPU-сервера обычно лучше оставить `1`, чтобы несколько тяжёлых WeSep задач
не конкурировали за одну видеокарту.
