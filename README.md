# Deep Learning — домашние задания

Решения домашних заданий по курсу Deep Learning. Каждое задание — соревнование в Яндекс Контесте: модель обучается на выданных данных, предсказания для test сдаются CSV-файлом, решение — Jupyter-ноутбуком.

## Задания

| № | Задача | Метод | Результат |
|---|---|---|---|
| 1 | [Поиск аномалий в табличных данных](dl_hw1_competition_data/README.md) — 6 датчиков, метрика Average Precision | MLP (64, 32), GELU, Dropout; 10 моделей на всём train | CV AP 0.746, Public AP 0.604 (порог 0.2) |
| 2 | [Классификация МРТ-снимков](dl_hw2_competition_data_v5/README.md) — 5 классов, метрика Accuracy | своя ResNet с нуля, аугментации, TTA, псевдо-лейблинг на test, ансамбль 5 моделей | 5-fold CV 0.994, Public 1.000 (порог 0.85) |

Ограничения курса: только разрешённые в условии архитектуры, обученные с нуля; без предобученных весов, готовых моделей (`torchvision.models`, `timm`) и внешних данных.

## Структура

```
dl_hwN_*/
├── README.md              # условие задания
├── ...                    # выданные данные и учебный ноутбук (не меняются)
└── solution/
    ├── solution.ipynb     # итоговый ноутбук: данные → обучение → валидация → submission.csv
    ├── build_solution.py  # генератор solution.ipynb (nbformat)
    ├── experiments.py     # исследовательский скрипт: группы конфигов, CV
    ├── check_submission.py# проверка формата CSV перед отправкой
    ├── EXPERIMENTS.md     # журнал: эксперименты, диагностика, все посылки
    ├── runs/results.jsonl # итоги всех экспериментов
    └── submissions/       # архив отправленных CSV и ноутбуков
```

Подробности по каждому заданию — в его `solution/EXPERIMENTS.md`.

## Запуск

Python 3.12. Окружение в корне репозитория:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r dl_hw2_competition_data_v5/solution/requirements.txt
```

Итоговый ноутбук выполняется из папки `solution/` (путь к данным в нём относительный):

```bash
cd dl_hw2_competition_data_v5/solution
../../.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 solution.ipynb
../../.venv/bin/python check_submission.py submission.csv
```

Обучение ДЗ 2 рассчитано на GPU (CUDA или Apple `mps`); на CPU оно заметно дольше.
