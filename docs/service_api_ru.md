# HTTP-сервис разделения голоса менеджера

Сервис оборачивает выбранный пайплайн WeSep в удобный API:

- принимает исходный шумный аудиофайл;
- принимает чистый reference-голос менеджера или сам выбирает auto-reference;
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
  "wesep_configured": true
}
```

## Дефолтные настройки

```bash
curl http://localhost:8088/v1/defaults
```

Текущие дефолты подобраны под тестовый файл `audio_test_noize.ogg` и выбранный
подход WeSep:

| параметр | дефолт | смысл |
|---|---:|---|
| `models` | `wesep` | использовать только выбранный WeSep-подход |
| `disable_fallback` | `true` | не подменять WeSep простым DSP fallback |
| `device` | `cuda:0` | GPU для инференса |
| `speech_target_dbfs` | `-23.0` | громкость чистого голоса |
| `speech_intro_duck_sec` | `2.2` | приглушение первых секунд, где часто лезут хлопки/аплодисменты |
| `speech_intro_lowpass_sec` | `6.0` | мягкая фильтрация начала чистого голоса |
| `residual_target_dbfs` | `-45.0` | громкость шумового трека |
| `auto_reference` | `true` | если reference не передан, взять лучший 20-секундный участок из входа |
| `auto_reference_sec` | `20.0` | длина auto-reference |

## Создать задачу

Минимальный запрос только с исходным аудио:

```bash
curl -X POST http://localhost:8088/v1/jobs \
  -F "audio=@/Users/vladimirkrasov/Downloads/audio_test_noize.ogg"
```

В этом режиме сервис сам создаст reference из наиболее активного 20-секундного
участка. Для production лучше передавать отдельный чистый reference менеджера:

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
  -F "speech_target_dbfs=-22" \
  -F "speech_intro_duck_sec=2.0" \
  -F "speech_intro_lowpass_sec=5.0" \
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

## Статус задачи

```bash
curl http://localhost:8088/v1/jobs/<job_id>
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
| `noise` | `manager_noise_residual.wav` | фон/шум с подавленным менеджером |
| `noise_subtract` | `manager_noise_residual_subtract.wav` | контрольный прямой subtract |
| `raw_speech` | `manager_speech_tse_raw.wav` | сырой вывод WeSep |
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
- Если в шумовом треке всё ещё слышен менеджер, лучше сначала улучшить reference.
  Auto-reference удобен для черновой обработки, но отдельный чистый reference
  обычно даёт более стабильное разделение.

## Где лежат результаты

По умолчанию:

```text
service_runs/<job_id>/
  input/
  output/
    manager_speech_clean.wav
    manager_noise_residual.wav
    manager_noise_residual_subtract.wav
    report.json
  artifacts.zip
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
