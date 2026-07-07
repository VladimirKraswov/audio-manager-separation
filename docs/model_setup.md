# Настройка моделей

Репозиторий использует адаптеры шаблонных команд: основной пайплайн умеет
вызывать реальные TSE-системы и модели улучшения речи, но не вшивает
исследовательские репозитории внутрь проекта. Это позволяет отдельно обновлять WeSep,
DeepFilterNet и другие модели.

Перед запуском скопируйте шаблон окружения:

```bash
cp env.tse.example env.tse
source env.tse
```

Текущая TSE-обёртка - WeSep:

```bash
export WESEP_TSE_CMD='python scripts/run_wesep_tse.py --mixture {mixture} --reference {reference} --output {output} --sample-rate {sample_rate} --device {device}'
```

## Подстановки

В каждой шаблонной команде доступны:

- `{mixture}` - подготовленный входной микс;
- `{reference}` - образец голоса целевого спикера;
- `{output}` - путь, куда модель должна записать результат;
- `{sample_rate}` - рабочая частота дискретизации;
- `{device}` - устройство инференса, например `cuda:0`, `cuda:1` или `cpu`.

## WeSep

Установите WeSep из официального репозитория в `external/wesep`, подключите его
к основной `.venv` и установите WeSpeaker, потому что pretrained WeSep-модель
зависит от определений speaker encoder из WeSpeaker.

`scripts/run_wesep_tse.py` обрабатывает длинные файлы чанками:

```bash
--chunk-sec 25 --overlap-sec 4
```

Модель загружается один раз, каждый чанк извлекается с тем же образцом голоса, а
результат склеивается через overlap-add. Для длинных записей это обязательно:
если подать в WeSep 15 минут или больше одним тензором, легко получить OOM.
Чанки держат VRAM примерно пропорционально выбранному `chunk_sec`.

Рекомендации:

- `25 sec / 4 sec overlap` - текущий баланс качества, скорости и VRAM;
- `15 sec / 3 sec overlap` - режим для меньшей VRAM;
- `35-45 sec` - можно пробовать на больших GPU, но выигрыш по скорости небольшой.

## Шумовой residual-трек

Пайплайн сохраняет промежуточные стадии:

- `manager_speech_tse_raw.wav` - прямой вывод выбранной TSE-модели;
- `manager_speech_tse_aligned.wav` - TSE после delay alignment;
- `manager_speech_tse_gainmatched.wav` - TSE после loudness/gain matching;
- `manager_speech_clean_prefilter.wav` - речь до финального residual-guided denoise;
- `manager_speech_clean.wav` - финальная речь менеджера;
- `manager_noise_residual_subtract.wav` - контрольный прямой subtract;
- `manager_noise_residual_prefilter.wav` - residual после базового подавления менеджера;
- `manager_noise_residual.wav` - финальный шумовой трек.

По умолчанию громкость речи работает в режиме `input_matched`, а не фиксируется
в `-23 dBFS`. Если установлен `pyloudnorm`, активная громкость считается через
BS.1770/LUFS; иначе используется active RMS dBFS. Финальная речь дополнительно
проходит residual-guided маску, чтобы убрать фон, который просочился через TSE.

Финальный шумовой трек строится не только прямым subtract. Он подавляет
частотно-временные bins, похожие на очищенный голос менеджера, затем проходит
дополнительный manager-leak suppression и нормализуется примерно к `-45 dBFS`.

## DeepFilterNet

Если `deepFilter` доступен в `PATH`, `process_call.py` попробует использовать
его для улучшения речи. Если DeepFilterNet не установлен, пайплайн пишет warning
в `report.json` и использует мягкую встроенную постобработку, чтобы smoke-тесты
оставались рабочими.

Можно задать свою команду:

```bash
export DEEPFILTERNET_CMD='deepFilter {input} --output {output}'
```

Если нужен строгий режим без резервной обработки, передайте:

```bash
--require-deepfilternet
```

## Dual-input режим

Если есть общий микс и отдельная запись микрофона менеджера, запускайте
`process_dual_input.py` или ручку `/v1/jobs-dual`. Режим сначала выравнивает обе
записи, использует микрофон менеджера как опорную дорожку для удаления
менеджера из общего микса, затем удаляет грубую оценку клиента из микрофона
менеджера. После этого очищенный `manager_mic_no_client_leak.wav` отправляется
в обычную TSE-цепочку.

Настройка WeSep при этом не меняется: dual-input просто даёт TSE более чистый
вход.

## Smoke-тест WeSep

```bash
python process_call.py \
  --input input/manager_mic_mono.wav \
  --reference input/manager_reference_clean.wav \
  --outdir output_wesep_test \
  --device cuda:0 \
  --quality smoke \
  --models wesep \
  --disable-fallback \
  --chunk-sec 6 \
  --overlap-sec 1
```

Для длинных файлов используйте продакшен-настройки:

```bash
--processing-sample-rate 16000 --tse-chunk-sec 25 --tse-overlap-sec 4
```
