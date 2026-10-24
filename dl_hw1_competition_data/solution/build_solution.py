import sys
import nbformat as nbf

out = sys.argv[1]
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Anomaly Detection Contest — решение (MLP)

Задача: по показаниям 6 датчиков выдать непрерывный `anomaly_score` для каждой строки `test.csv`. Метрика — Average Precision (AP).

Итоговое решение (только архитектуры из учебного ноутбука):
1. **Подготовка данных.** `id` не используется как признак. Хвосты обрезаются по 0.5% / 99.5% квантилям, затем `StandardScaler`. Обе операции обучаются **только на train-части фолда**.
2. **Модель.** Supervised MLP 6 → 64 → 32 → 1: GELU, Dropout 0.2, `BCEWithLogitsLoss(pos_weight=√(N−/N+))`, AdamW (lr=1e-3), batch 256.
3. **Валидация.** `StratifiedGroupKFold(5)` × 3 повтора. Группа — одинаковые строки признаков, чтобы дубли не попадали одновременно в train и val. Число эпох выбирается по средней по фолдам кривой validation AP, после чего OOF AP считается для этой фиксированной эпохи.
4. **Скор для test** — средний logit 10 моделей, обученных на всём train с числом эпох, выбранным по CV.

Используются только выданные `train.csv` и `test.csv`. Внешних данных, предобученных весов и разметки test нет.
""")

code("""
import hashlib
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

torch.set_num_threads(1)              # для воспроизводимости результатов на CPU
torch.use_deterministic_algorithms(True)
random.seed(0)
""")

md("## Загрузка данных")

code("""
DATA_DIR = ".."  # train.csv / test.csv лежат на уровень выше папки solution/

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

FEATURES = [f"sensor_{i}" for i in range(6)]  # id — не признак
X_all = train[FEATURES].to_numpy(np.float32)
y_all = train["anomaly"].to_numpy(np.float32)
X_test_raw = test[FEATURES].to_numpy(np.float32)

print("train:", X_all.shape, "test:", X_test_raw.shape)
print(f"аномалий: {int(y_all.sum())} ({y_all.mean():.2%})")
""")

md("""
## Краткий EDA

- Сильный дисбаланс (~2.3% аномалий): accuracy бессмысленна, используем AP и `pos_weight` в loss.
- ~29% строк train — одна и та же «холостая» строка; аномалий среди её копий почти нет. В test таких строк ~1040.
- У признаков тяжёлые хвосты (до ±20 при std≈1) — поэтому обрезаем их по квантилям train.
""")

code("""
values, counts = np.unique(X_all, axis=0, return_counts=True)
idle_row = values[counts.argmax()]
is_idle = (X_all == idle_row).all(axis=1)
print(f"холостая строка: {is_idle.sum()} копий в train, аномалий среди них {int(y_all[is_idle].sum())}")
print(f"в test: {(X_test_raw == idle_row).all(axis=1).sum()} копий")
display(train[FEATURES].describe().T.round(3))
""")

md("""
## Разбиение на фолды

Одинаковые строки признаков объединены в группы, чтобы копия не оказалась одновременно в train и в validation.
Исключение — холостая строка: каждая её копия — отдельная группа, так что она есть во всех фолдах, как и в test.
""")

code("""
keys = [hashlib.md5(row.tobytes()).hexdigest() for row in X_all]
groups = pd.factorize(pd.Series(keys))[0].astype(np.int64)
groups[is_idle] = groups.max() + 1 + np.arange(is_idle.sum())

N_FOLDS, REPEATS = 5, (0, 1, 2)


def get_splits(repeat):
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=repeat)
    return list(sgkf.split(X_all, y_all, groups))
""")

md("## Предобработка (обучается только на train-части фолда)")

code("""
def preprocess(X_tr, X_others):
    X_tr = X_tr.astype(np.float64)
    lo, hi = np.quantile(X_tr, [0.005, 0.995], axis=0)
    scaler = StandardScaler().fit(np.clip(X_tr, lo, hi))
    transform = lambda X: scaler.transform(np.clip(X.astype(np.float64), lo, hi)).astype(np.float32)
    return [transform(X_tr)] + [transform(X) for X in X_others]
""")

md("""
## Модель: MLP из учебного ноутбука

Linear → GELU → Dropout на каждом скрытом слое, на выходе один logit.
По сравнению с учебным бейзлайном изменены функция активации (GELU), Dropout и вес позитивного класса.
""")

code("""
CONFIG = dict(hidden=(64, 32), dropout=0.2, lr=1e-3, weight_decay=0.0, batch_size=256,
              pos_weight_power=0.5, max_epochs=400)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden=(64, 32), dropout=0.0):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def predict_proba(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(X)).numpy()


def train_fold(tr_idx, val_idx, seed, cfg=CONFIG):
    \"\"\"Обучает MLP на фолде; возвращает предсказания val и test после каждой эпохи.\"\"\"
    torch.manual_seed(seed)
    np.random.seed(seed)
    X_tr, X_val, X_te = preprocess(X_all[tr_idx], [X_all[val_idx], X_test_raw])
    y_tr = y_all[tr_idx]

    model = MLP(X_tr.shape[1], cfg["hidden"], cfg["dropout"])
    pos_weight = ((1 - y_tr).sum() / y_tr.sum()) ** cfg["pos_weight_power"]
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    Xt, yt = torch.tensor(X_tr), torch.tensor(y_tr)
    Xv, Xe = torch.tensor(X_val), torch.tensor(X_te)
    gen = torch.Generator().manual_seed(seed)
    val_hist, test_hist = [], []
    for _ in range(cfg["max_epochs"]):
        model.train()
        perm = torch.randperm(len(Xt), generator=gen)
        for i in range(0, len(Xt), cfg["batch_size"]):
            b = perm[i:i + cfg["batch_size"]]
            optimizer.zero_grad()
            loss = criterion(model(Xt[b]), yt[b])
            loss.backward()
            optimizer.step()
        val_hist.append(predict_proba(model, Xv))
        test_hist.append(predict_proba(model, Xe))
    return np.stack(val_hist), np.stack(test_hist)


def train_full(seed, n_epochs, cfg=CONFIG):
    \"\"\"Обучает MLP на всём train ровно n_epochs эпох; возвращает logit на test.\"\"\"
    torch.manual_seed(10_000 + seed)
    np.random.seed(10_000 + seed)
    X_tr, X_te = preprocess(X_all, [X_test_raw])

    model = MLP(X_tr.shape[1], cfg["hidden"], cfg["dropout"])
    pos_weight = ((1 - y_all).sum() / y_all.sum()) ** cfg["pos_weight_power"]
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    Xt, yt = torch.tensor(X_tr), torch.tensor(y_all)
    gen = torch.Generator().manual_seed(10_000 + seed)
    for _ in range(n_epochs):
        model.train()
        perm = torch.randperm(len(Xt), generator=gen)
        for i in range(0, len(Xt), cfg["batch_size"]):
            b = perm[i:i + cfg["batch_size"]]
            optimizer.zero_grad()
            loss = criterion(model(Xt[b]), yt[b])
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X_te)).numpy()
""")

md("""
## Обучение: 3 повтора × 5 фолдов

Для каждого фолда сохраняем предсказания на каждой эпохе. Так можно выбрать **одно** число эпох для всех фолдов
по средней кривой validation AP, а не подбирать эпоху под каждый фолд отдельно, что завысило бы оценку.
Занимает несколько минут на CPU.
""")

code("""
E = CONFIG["max_epochs"]
oof_by_repeat = []                      # [repeat] -> (E, N)
fold_curves = []                        # AP фолда по эпохам
test_sum = np.zeros((E, len(X_test_raw)))
n_models = 0

for rep in REPEATS:
    oof = np.zeros((E, len(y_all)))
    for fold, (tr_idx, val_idx) in enumerate(get_splits(rep)):
        val_hist, test_hist = train_fold(tr_idx, val_idx, seed=1000 * rep + fold)
        oof[:, val_idx] = val_hist
        fold_curves.append([average_precision_score(y_all[val_idx], val_hist[e]) for e in range(E)])
        test_sum += test_hist
        n_models += 1
    oof_by_repeat.append(oof)
    print(f"повтор {rep}: готово")

fold_curves = np.array(fold_curves)
mean_curve = fold_curves.mean(0)
smooth = np.convolve(np.pad(mean_curve, 2, mode="edge"), np.ones(5) / 5, mode="valid")
best_epoch = int(smooth.argmax())
print("выбранное число эпох:", best_epoch + 1)
""")

md("## Локальная validation AP")

code("""
oof_aps = [average_precision_score(y_all, o[best_epoch]) for o in oof_by_repeat]
fold_aps = fold_curves[:, best_epoch]
oof = np.mean([o[best_epoch] for o in oof_by_repeat], axis=0)

print(f"OOF AP по повторам:  {np.round(oof_aps, 4)}  -> среднее {np.mean(oof_aps):.4f}")
print(f"AP по 15 фолдам:     {fold_aps.mean():.4f} ± {fold_aps.std(ddof=1):.4f}")
print(f"OOF ROC-AUC:         {roc_auc_score(y_all, oof_by_repeat[0][best_epoch]):.4f}")
print(f"AP без холостых строк: {np.mean([average_precision_score(y_all[~is_idle], o[best_epoch][~is_idle]) for o in oof_by_repeat]):.4f}")
""")

md("""
### Сравнение вариантов (из рабочих экспериментов, та же схема CV)

| Вариант | OOF AP |
|---|---|
| Бейзлайн учебного ноутбука: MLP (64,32), ReLU, StandardScaler, pos_weight=N−/N+ | 0.68 |
| + обрезка хвостов по квантилям | 0.69 |
| + √pos_weight | 0.71 |
| + Dropout 0.2 | 0.73 |
| **+ GELU (итог)** | **0.75** |
| То же, слои (128,64) | 0.74 |
| Автоэнкодер на normal (лучший из 5 вариантов), скор = MSE реконструкции | 0.30 |
| Смесь MLP + AE по рангам | ≤ 0.69 |

Автоэнкодер не использует метки аномалий и заметно уступает supervised MLP; в смеси он только портит ранжирование, поэтому в итог не входит.
""")

md("""
## Финальные модели: обучение на всём train

Число эпох `best_epoch + 1` выбрано по CV. Финальные модели обучаются на всех 7828 строках train
(на 25% больше данных, чем фолд-модели, — кривая обучения ещё растёт) с `N_SEEDS` разными seed, их логиты усредняются.
Скор = средний logit: он монотонен вероятности, поэтому AP не меняется, а насыщения около 1 нет.
""")

code("""
from scipy.stats import spearmanr

N_SEEDS = 10
n_epochs_final = best_epoch + 1
test_logits = np.stack([train_full(seed, n_epochs_final) for seed in range(N_SEEDS)])
test_logit = test_logits.mean(0)

cv_test_pred = test_sum[best_epoch] / n_models   # прежний вариант: среднее вероятностей фолд-моделей
print(f"моделей: {N_SEEDS}, эпох: {n_epochs_final}")
print(f"Spearman(test_logit, CV-среднее вероятностей): {spearmanr(test_logit, cv_test_pred).statistic:.4f}")
""")

md("## Экспорт submission.csv")

code("""
submission = pd.DataFrame({"id": test["id"].to_numpy(), "anomaly_score": test_logit})

assert list(submission.columns) == ["id", "anomaly_score"]
assert len(submission) == len(test) and submission["id"].is_unique
assert set(submission["id"]) == set(test["id"])
assert np.isfinite(submission["anomaly_score"]).all()

submission.to_csv("submission.csv", index=False, float_format="%.10g")
print(submission.head())
print("размер файла:", os.path.getsize("submission.csv"), "байт")
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}
nbf.validate(nb)
nbf.write(nb, out)
print("written", out)
