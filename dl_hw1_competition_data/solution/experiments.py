# %% [markdown]
# # Эксперименты (рабочая копия, в Контест не идёт)
#
# Файл в формате `# %%`: в VS Code открывается как ноутбук (Run Cell), из терминала —
# `python experiments.py <группа>` прогоняет группу конфигов параллельно и дописывает итоги в `runs/results.jsonl`.
#
# Схема валидации (см. план):
# - StratifiedGroupKFold(5) × 3 повтора; группа = хеш строки признаков (мелкие дубли не разъезжаются
#   между train/val), холостые строки — каждая своя группа (они должны быть во всех фолдах, как в test);
# - модель учится фиксированное число эпох, на каждой эпохе сохраняем OOF и test предсказания;
#   число эпох выбираем по средней по фолдам кривой val AP — так OOF AP почти честная;
# - test = среднее вероятностей фолд-моделей на выбранной эпохе.

# %%
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
os.makedirs(RUNS, exist_ok=True)

FEATURES = [f"sensor_{i}" for i in range(6)]
train = pd.read_csv(os.path.join(HERE, "..", "train.csv"))
test = pd.read_csv(os.path.join(HERE, "..", "test.csv"))
X_all = train[FEATURES].to_numpy(np.float32)
y_all = train["anomaly"].to_numpy(np.float32)
X_test_raw = test[FEATURES].to_numpy(np.float32)

_vals, _cnt = np.unique(X_all, axis=0, return_counts=True)
IDLE_ROW = _vals[_cnt.argmax()]
idle_tr = (X_all == IDLE_ROW).all(1)
idle_te = (X_test_raw == IDLE_ROW).all(1)

N_FOLDS, REPEATS = 5, (0, 1, 2)


def make_groups():
    keys = [hashlib.md5(r.tobytes()).hexdigest() for r in X_all]
    groups = pd.factorize(pd.Series(keys))[0].astype(np.int64)
    groups[idle_tr] = groups.max() + 1 + np.arange(idle_tr.sum())
    return groups


GROUPS = make_groups()


def splits(repeat):
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=repeat)
    return list(sgkf.split(X_all, y_all, GROUPS))


# %% [markdown]
# ## Предобработка (fit только на train-части фолда)

# %%
def preprocess(kind, X_tr, X_others, add_idle):
    """kind: std | clip_std | robust | clip_robust | quantile."""
    X_tr = X_tr.astype(np.float64)
    X_others = [x.astype(np.float64) for x in X_others]
    if kind.startswith("clip"):
        lo, hi = np.quantile(X_tr, [0.005, 0.995], axis=0)
        X_tr = np.clip(X_tr, lo, hi)
        X_others = [np.clip(x, lo, hi) for x in X_others]
    if kind.endswith("std"):
        sc = StandardScaler()
    elif kind.endswith("robust"):
        sc = RobustScaler()
    elif kind == "quantile":
        sc = QuantileTransformer(output_distribution="normal", n_quantiles=1000, random_state=0)
    else:
        raise ValueError(kind)
    out = [sc.fit_transform(X_tr)] + [sc.transform(x) for x in X_others]
    return [o.astype(np.float32) for o in out]


def idle_col(X_raw):
    return (X_raw == IDLE_ROW).all(1).astype(np.float32)[:, None]


# %% [markdown]
# ## Модели: MLP и автоэнкодер (как в учебном ноутбуке, + выбор активации)

# %%
ACTS = {"relu": nn.ReLU, "leaky": lambda: nn.LeakyReLU(0.1), "gelu": nn.GELU}


def block(prev, h, act, bn, dropout):
    layers: list[nn.Module] = [nn.Linear(prev, h)]
    if bn:
        layers.append(nn.BatchNorm1d(h))
    layers.append(ACTS[act]())
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return layers


class MLP(nn.Module):
    def __init__(self, d, hidden=(64, 32), act="relu", bn=False, dropout=0.0):
        super().__init__()
        layers, prev = [], d
        for h in hidden:
            layers += block(prev, h, act, bn, dropout)
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


class Autoencoder(nn.Module):
    def __init__(self, d, hidden=(16, 8), latent=3, act="relu", bn=False, dropout=0.0):
        super().__init__()
        enc, prev = [], d
        for h in hidden:
            enc += block(prev, h, act, bn, dropout)
            prev = h
        enc.append(nn.Linear(prev, latent))
        dec, prev = [], latent
        for h in reversed(hidden):
            dec += block(prev, h, act, bn, dropout)
            prev = h
        dec.append(nn.Linear(prev, d))
        self.encoder, self.decoder = nn.Sequential(*enc), nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))


# %% [markdown]
# ## Функции потерь и признаки из автоэнкодера

# %%
def make_loss(cfg, pw):
    """bce — как в ноутбуке; focal — BCE с понижением веса лёгких примеров;
    pair — BCE + логистическая парная потеря «аномалия выше нормы» (ближе к ранжированию, т.е. к AP)."""
    kind = cfg.get("loss", "bce")
    pos_w = torch.tensor([pw], dtype=torch.float32)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    if kind == "bce":
        return bce
    if kind == "focal":
        gamma = cfg.get("gamma", 2.0)

        def focal(logits, y):
            per = nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_w, reduction="none")
            p = torch.sigmoid(logits)
            pt = torch.where(y > 0.5, p, 1 - p)
            return ((1 - pt) ** gamma * per).mean()
        return focal
    if kind == "pair":
        lam = cfg.get("pair_lambda", 1.0)

        def pair(logits, y):
            loss = bce(logits, y)
            pos, neg = logits[y > 0.5], logits[y < 0.5]
            if len(pos) and len(neg):
                loss = loss + lam * nn.functional.softplus(neg[None, :] - pos[:, None]).mean()
            return loss
        return pair
    raise ValueError(kind)


def add_ae_features(X_tr, X_others, y_tr, idle_mask, ae_epochs, seed):
    """Обучает маленький AE на normal train-части (холостая строка один раз) и добавляет
    поэлементные квадраты ошибки реконструкции как 6 новых входов MLP."""
    torch.manual_seed(seed + 7)
    Xn = X_tr[(y_tr == 0) & ~idle_mask]
    Xn = np.vstack([Xn, X_tr[(y_tr == 0) & idle_mask][:1]])
    ae = Autoencoder(X_tr.shape[1], (16, 8), 3, "relu")
    opt = torch.optim.AdamW(ae.parameters(), lr=1e-3)
    Xt = torch.tensor(Xn)
    g = torch.Generator().manual_seed(seed + 7)
    for _ in range(ae_epochs):
        ae.train()
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), 256):
            xb = Xt[perm[i:i + 256]]
            opt.zero_grad()
            nn.functional.mse_loss(ae(xb), xb).backward()
            opt.step()
    ae.eval()

    def feats(X):
        with torch.no_grad():
            x = torch.tensor(X)
            return np.hstack([X, ((ae(x) - x) ** 2).numpy()]).astype(np.float32)
    return [feats(X_tr)] + [feats(X) for X in X_others]


# %% [markdown]
# ## Обучение одного фолда: возвращает предсказания val и test на каждой эпохе

# %%
def run_fold(cfg, tr_idx, val_idx, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model_type = cfg.get("model", "mlp")

    X_tr, X_val, X_te = preprocess(cfg["prep"], X_all[tr_idx], [X_all[val_idx], X_test_raw], cfg.get("idle", False))
    if cfg.get("idle", False):
        X_tr = np.hstack([X_tr, idle_col(X_all[tr_idx])])
        X_val = np.hstack([X_val, idle_col(X_all[val_idx])])
        X_te = np.hstack([X_te, idle_col(X_test_raw)])
    y_tr = y_all[tr_idx]
    if cfg.get("ae_feat", 0):
        X_tr, X_val, X_te = add_ae_features(X_tr, [X_val, X_te], y_tr, idle_tr[tr_idx], cfg["ae_feat"], seed)
    d = X_tr.shape[1]

    if model_type == "mlp":
        model = MLP(d, cfg["hidden"], cfg.get("act", "relu"), cfg.get("bn", False), cfg.get("dropout", 0.0))
        pw = (1 - y_tr).sum() / y_tr.sum()
        pw = pw ** cfg.get("pw_power", 1.0)
        criterion = make_loss(cfg, pw)
        Xt, yt = torch.tensor(X_tr), torch.tensor(y_tr)
    else:
        model = Autoencoder(d, cfg["hidden"], cfg["latent"], cfg.get("act", "relu"), cfg.get("bn", False), cfg.get("dropout", 0.0))
        normal = y_tr == 0
        Xn = X_tr[normal]
        if cfg.get("dedup_idle", True):
            idle_n = idle_tr[tr_idx][normal]
            keep = ~idle_n
            keep[np.flatnonzero(idle_n)[:1]] = True
            Xn = Xn[keep]
        Xt, yt = torch.tensor(Xn), None
        criterion = nn.MSELoss()

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=cfg.get("wd", 0.0))
    bs, n = cfg.get("batch", 256), len(Xt)
    Xv, Xe = torch.tensor(X_val), torch.tensor(X_te)

    def score(X):
        model.eval()
        with torch.no_grad():
            if model_type == "mlp":
                return torch.sigmoid(model(X)).numpy()
            return ((model(X) - X) ** 2).mean(1).numpy()

    val_hist, te_hist = [], []
    g = torch.Generator().manual_seed(seed)
    for _ in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            if len(b) < 2 and cfg.get("bn", False):
                continue
            opt.zero_grad()
            if model_type == "mlp":
                xb = Xt[b]
                in_noise = cfg.get("in_noise", 0.0)
                if in_noise > 0:
                    xb = xb + in_noise * torch.randn_like(xb)
                loss = criterion(model(xb), yt[b])
            else:
                xb = Xt[b]
                noise = cfg.get("noise", 0.0)
                inp = xb + noise * torch.randn_like(xb) if noise > 0 else xb
                loss = criterion(model(inp), xb)
            loss.backward()
            opt.step()
        val_hist.append(score(Xv))
        te_hist.append(score(Xe))
    return np.stack(val_hist), np.stack(te_hist)


# %% [markdown]
# ## CV-прогон конфига

# %%
def run_config(cfg):
    torch.set_num_threads(1)
    t0 = time.time()
    E = cfg["epochs"]
    rep_oof = []        # [repeat] -> (E, N) OOF по эпохам
    fold_curves = []    # (E,) AP фолда по эпохам
    test_sum = np.zeros((E, len(X_test_raw)))
    n_models = 0
    for rep in REPEATS:
        oof = np.zeros((E, len(y_all)))
        for f, (tr_idx, val_idx) in enumerate(splits(rep)):
            vh, th = run_fold(cfg, tr_idx, val_idx, seed=1000 * rep + f)
            oof[:, val_idx] = vh
            fold_curves.append([average_precision_score(y_all[val_idx], vh[e]) for e in range(E)])
            test_sum += th
            n_models += 1
        rep_oof.append(oof)
    curves = np.array(fold_curves)                       # (folds*repeats, E)
    mean_curve = curves.mean(0)
    smooth = np.convolve(np.pad(mean_curve, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    e_best = int(smooth.argmax())

    oof_aps = [average_precision_score(y_all, o[e_best]) for o in rep_oof]
    nonidle = [average_precision_score(y_all[~idle_tr], o[e_best][~idle_tr]) for o in rep_oof]
    o = rep_oof[0][e_best]
    idle_pct = float((o < o[idle_tr].mean()).mean())
    fold_at = curves[:, e_best]
    res = {
        "name": cfg["name"], "cfg": cfg, "epoch": e_best + 1,
        "oof_ap": float(np.mean(oof_aps)), "oof_ap_reps": [round(a, 4) for a in oof_aps],
        "fold_ap_mean": float(fold_at.mean()), "fold_ap_se": float(fold_at.std(ddof=1) / np.sqrt(len(fold_at))),
        "nonidle_ap": float(np.mean(nonidle)), "idle_pct": idle_pct,
        "sec": round(time.time() - t0, 1),
    }
    np.save(os.path.join(RUNS, f"{cfg['name']}_oof.npy"), np.stack([r[e_best] for r in rep_oof]))
    np.save(os.path.join(RUNS, f"{cfg['name']}_test.npy"), test_sum[e_best] / n_models)
    return res


# %% [markdown]
# ## Группы конфигов

# %%
BASE = dict(model="mlp", prep="std", hidden=(64, 32), act="relu", bn=False, dropout=0.0,
            lr=1e-3, wd=0.0, batch=256, epochs=120)


def C(name, **kw):
    return {**BASE, "name": name, **kw}


GROUPS_CFG = {
    "prep": [
        C("p_std"),
        C("p_std_idle", idle=True),
        C("p_clipstd", prep="clip_std"),
        C("p_clipstd_idle", prep="clip_std", idle=True),
        C("p_robust_idle", prep="robust", idle=True),
        C("p_cliprobust_idle", prep="clip_robust", idle=True),
        C("p_quant", prep="quantile"),
        C("p_quant_idle", prep="quantile", idle=True),
    ],
    # предобработка зафиксирована: clip_std; эпох больше — в "prep" лучшая эпоха упиралась в потолок 120
    "arch": [
        C("a_base", prep="clip_std", epochs=300),
        C("a_128_64", prep="clip_std", epochs=300, hidden=(128, 64)),
        C("a_128_64_32", prep="clip_std", epochs=300, hidden=(128, 64, 32)),
        C("a_gelu", prep="clip_std", epochs=300, act="gelu"),
        C("a_leaky", prep="clip_std", epochs=300, act="leaky"),
        C("a_bn", prep="clip_std", epochs=300, bn=True),
        C("a_drop02", prep="clip_std", epochs=300, dropout=0.2),
        C("a_wd1e-3", prep="clip_std", epochs=300, wd=1e-3),
        C("a_pwsqrt", prep="clip_std", epochs=300, pw_power=0.5),
        C("a_batch128", prep="clip_std", epochs=300, batch=128),
    ],
    # сочетания лучших из "arch": √pos_weight, Dropout 0.2, GELU/LeakyReLU
    "combo": [
        C("c_pw_drop", prep="clip_std", epochs=400, pw_power=0.5, dropout=0.2),
        C("c_pw_gelu", prep="clip_std", epochs=400, pw_power=0.5, act="gelu"),
        C("c_pw_drop_gelu", prep="clip_std", epochs=400, pw_power=0.5, dropout=0.2, act="gelu"),
        C("c_pw_drop_leaky", prep="clip_std", epochs=400, pw_power=0.5, dropout=0.2, act="leaky"),
        C("c_pw_drop_gelu_128", prep="clip_std", epochs=400, pw_power=0.5, dropout=0.2, act="gelu", hidden=(128, 64)),
    ],
    # проверка «переподстройки»: маленькие и сильно регуляризованные сети
    "simple": [
        C("s_16", prep="clip_std", epochs=400, hidden=(16,), act="gelu", pw_power=0.5),
        C("s_32_16", prep="clip_std", epochs=400, hidden=(32, 16), act="gelu", pw_power=0.5, dropout=0.2),
        C("s_wd1e-2", prep="clip_std", epochs=400, act="gelu", pw_power=0.5, dropout=0.2, wd=1e-2),
        C("s_drop04", prep="clip_std", epochs=400, act="gelu", pw_power=0.5, dropout=0.4),
        C("s_lr3e-4", prep="clip_std", epochs=400, act="gelu", pw_power=0.5, dropout=0.2, lr=3e-4),
        C("s_std_noclip", prep="std", epochs=400, act="gelu", pw_power=0.5, dropout=0.2),
        C("s_bn_drop", prep="clip_std", epochs=400, act="gelu", pw_power=0.5, dropout=0.2, bn=True),
        C("s_256_128", prep="clip_std", epochs=400, act="gelu", pw_power=0.5, dropout=0.3, hidden=(256, 128)),
    ],
    # второй заход: шум на входе, другие loss, признаки из AE (база = 02)
    "more": [
        C("m_noise005", prep="clip_std", epochs=400, act="gelu", pw_power=0.5, dropout=0.2, in_noise=0.05),
        C("m_noise01", prep="clip_std", epochs=400, act="gelu", pw_power=0.5, dropout=0.2, in_noise=0.1),
        C("m_noise02", prep="clip_std", epochs=400, act="gelu", pw_power=0.5, dropout=0.2, in_noise=0.2),
        C("m_focal", prep="clip_std", epochs=400, act="gelu", pw_power=0.5, dropout=0.2, loss="focal", gamma=2.0),
        C("m_focal1", prep="clip_std", epochs=400, act="gelu", pw_power=0.5, dropout=0.2, loss="focal", gamma=1.0),
        C("m_pair", prep="clip_std", epochs=400, act="gelu", pw_power=0.5, dropout=0.2, loss="pair", pair_lambda=1.0),
        C("m_pair03", prep="clip_std", epochs=400, act="gelu", pw_power=0.5, dropout=0.2, loss="pair", pair_lambda=0.3),
        C("m_aefeat10", prep="clip_std", epochs=400, act="gelu", pw_power=0.5, dropout=0.2, ae_feat=10),
        C("m_aefeat50", prep="clip_std", epochs=400, act="gelu", pw_power=0.5, dropout=0.2, ae_feat=50),
        C("m_noise01_pair", prep="clip_std", epochs=400, act="gelu", pw_power=0.5, dropout=0.2, in_noise=0.1, loss="pair", pair_lambda=0.3),
    ],
    # автоэнкодер: обучение только на normal, холостая строка дедуплицирована, скор = MSE реконструкции
    "ae": [
        C("ae_16_8_l2", model="ae", prep="clip_std", hidden=(16, 8), latent=2, epochs=200),
        C("ae_16_8_l3", model="ae", prep="clip_std", hidden=(16, 8), latent=3, epochs=200),
        C("ae_32_16_l3", model="ae", prep="clip_std", hidden=(32, 16), latent=3, epochs=200),
        C("ae_16_8_l3_noise", model="ae", prep="clip_std", hidden=(16, 8), latent=3, epochs=200, noise=0.1),
        C("ae_16_8_l3_std", model="ae", prep="std", hidden=(16, 8), latent=3, epochs=200),
    ],
}


def log_results(results):
    with open(os.path.join(RUNS, "results.jsonl"), "a") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def show(results):
    df = pd.DataFrame(results)[["name", "oof_ap", "fold_ap_mean", "fold_ap_se", "nonidle_ap", "idle_pct", "epoch", "sec"]]
    print(df.sort_values("oof_ap", ascending=False).round(4).to_string(index=False))


# %%
if __name__ == "__main__" and len(sys.argv) > 1:
    cfgs = [c for g in sys.argv[1:] for c in GROUPS_CFG[g]]
    with Pool(min(len(cfgs), os.cpu_count() or 4)) as pool:
        results = pool.map(run_config, cfgs)
    log_results(results)
    show(results)
