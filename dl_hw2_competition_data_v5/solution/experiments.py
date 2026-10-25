# %% [markdown]
# # ДЗ 2: эксперименты (рабочая копия, в Контест не идёт)
#
# Файл в формате `# %%`: в VS Code открывается как ноутбук (Run Cell), из терминала —
# `python experiments.py <группа> [<группа> ...]` последовательно прогоняет конфиги групп
# (один процесс, MPS) и дописывает итоги в `runs/results.jsonl`.
# `python experiments.py ens name1 name2 ...` — ансамбль сохранённых val-предсказаний (только train-модели).
# `python experiments.py enstest out.csv name1 name2 ...` — CSV-сабмит из усреднённых test-предсказаний.
#
# Данные целиком в памяти, аугментации — батчем на устройстве. Только CNN, собранная вручную.

# %%
import contextlib
import glob
import hashlib
import json
import math
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import check_submission

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
WEIGHTS = os.path.join(HERE, "weights")
DATA = os.path.join(HERE, "..", "mri_classification_dataset")
os.makedirs(RUNS, exist_ok=True)

NUM_CLASSES = 5
UNIQUE_THR = 0.98
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# %% [markdown]
# ## Загрузка данных (кэш в runs/cache_*.pt)

# %%
def list_split(split):
    """Возвращает (пути, метки или None, id)."""
    if split == "test":
        paths = sorted(glob.glob(os.path.join(DATA, "test", "*.jpg")))
        labels = None
    else:
        paths, labels = [], []
        for k in range(NUM_CLASSES):
            ps = sorted(glob.glob(os.path.join(DATA, split, f"class_{k}", "*.jpg")))
            paths += ps
            labels += [k] * len(ps)
    ids = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    return paths, labels, ids


def load_split(split, S):
    """-> dict(x=uint8 [N,1,S,S], y=int64 [N] или None, ids, orig=исходный размер [N])."""
    cache = os.path.join(RUNS, f"cache_{split}_{S}_bilinear.pt")
    if os.path.exists(cache):
        return torch.load(cache, weights_only=True)
    paths, labels, ids = list_split(split)
    xs, orig = [], []
    for p in paths:
        im = Image.open(p).convert("L")
        orig.append(im.size[0])
        xs.append(np.asarray(im.resize((S, S), Image.BILINEAR), dtype=np.uint8))
    d = dict(
        x=torch.from_numpy(np.stack(xs))[:, None].contiguous(),
        y=None if labels is None else torch.tensor(labels, dtype=torch.long),
        ids=ids,
        orig=torch.tensor(orig),
    )
    torch.save(d, cache)
    return d


def small_vectors(split):
    paths, _, _ = list_split(split)
    v = np.stack([
        np.asarray(Image.open(p).convert("L").resize((16, 16), Image.BILINEAR), dtype=np.float32).ravel()
        for p in paths
    ])
    v -= v.mean(1, keepdims=True)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
    return v


def unique_masks():
    """val/test_unique: максимум корреляции 16x16-вектора с train < 0.98."""
    cache = os.path.join(RUNS, f"cache_unique_thr{UNIQUE_THR}.pt")
    if os.path.exists(cache):
        return torch.load(cache, weights_only=True)
    tr = small_vectors("train")
    out = {}
    for split in ("val", "test"):
        corr = small_vectors(split) @ tr.T
        out[split] = torch.from_numpy(corr.max(1) < UNIQUE_THR)
    torch.save(out, cache)
    return out


_DATA_CACHE = {}


def get_data(S):
    if S not in _DATA_CACHE:
        d = {s: load_split(s, S) for s in ("train", "val", "test")}
        m = unique_masks()
        d["val"]["unique"], d["test"]["unique"] = m["val"], m["test"]
        print(f"img={S}: train {len(d['train']['x'])}, val {len(d['val']['x'])} "
              f"(unique {m['val'].float().mean():.3f}), test {len(d['test']['x'])} "
              f"(unique {m['test'].float().mean():.3f})", flush=True)
        _DATA_CACHE[S] = d
    return _DATA_CACHE[S]


# %% [markdown]
# ## Аугментации на устройстве

# %%
def _affine(x, theta):
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def shift_images(x, dx, dy):
    """Сдвиг на доли размера (dx, dy — тензоры [B] долей ширины/высоты)."""
    B = x.shape[0]
    theta = torch.zeros(B, 2, 3, device=x.device)
    theta[:, 0, 0] = 1
    theta[:, 1, 1] = 1
    theta[:, 0, 2] = -2 * dx
    theta[:, 1, 2] = -2 * dy
    return _affine(x, theta)


AUG_COLS = 12  # колонок случайных чисел на объект: flip, ang, scale, shift(2), c, b, area, ratio, y0, x0, gate


def aug_params(n, gen):
    """Все случайные числа аугментаций одним тензором [n, AUG_COLS] (CPU-генератор -> device одним переносом)."""
    return torch.rand(n, AUG_COLS, generator=gen).to(DEVICE)


def augment(x, cfg, gen, r=None):
    """x: float [B,1,S,S] в [0,1] на device. cfg: dict(rot, scale, shift, bc, erase, flip). Сила 0 -> отключено.
    r: готовые случайные числа [B, AUG_COLS] (aug_params), иначе берутся из CPU-генератора gen."""
    B, _, H, W = x.shape
    dev = x.device
    if r is None:
        r = aug_params(B, gen)

    if cfg.get("flip", True):
        flip = (r[:, 0] < 0.5).float() * -2 + 1  # -1 -> отразить по горизонтали
    else:
        flip = torch.ones(B, device=dev)
    ang = (r[:, 1] * 2 - 1) * math.radians(cfg.get("rot", 0))
    sc = 1 + (r[:, 2] * 2 - 1) * cfg.get("scale", 0)
    sh = (r[:, 3:5] * 2 - 1) * cfg.get("shift", 0) * 2
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = torch.cos(ang) / sc * flip
    theta[:, 0, 1] = -torch.sin(ang) / sc
    theta[:, 1, 0] = torch.sin(ang) / sc * flip
    theta[:, 1, 1] = torch.cos(ang) / sc
    theta[:, :, 2] = sh
    x = _affine(x, theta)

    bc = cfg.get("bc", 0)
    if bc > 0:
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        c = 1 + (r[:, 5].view(B, 1, 1, 1) * 2 - 1) * bc
        b = (r[:, 6].view(B, 1, 1, 1) * 2 - 1) * bc * 0.5
        x = ((x - mean) * c + mean + b).clamp(0, 1)

    er = cfg.get("erase", 0)
    if er > 0:
        area = 0.02 + r[:, 7] * 0.13
        ratio = torch.exp((r[:, 8] * 2 - 1) * math.log(3))
        h = (area * ratio).sqrt().clamp(max=1) * H
        w = (area / ratio).sqrt().clamp(max=1) * W
        y0 = r[:, 9] * (H - h)
        x0 = r[:, 10] * (W - w)
        ys = torch.arange(H, device=dev).view(1, H, 1)
        xs = torch.arange(W, device=dev).view(1, 1, W)
        m = (ys >= y0.view(B, 1, 1)) & (ys < (y0 + h).view(B, 1, 1)) & \
            (xs >= x0.view(B, 1, 1)) & (xs < (x0 + w).view(B, 1, 1))
        m = m & (r[:, 11] < er).view(B, 1, 1)
        x = torch.where(m[:, None], x.mean(dim=(1, 2, 3), keepdim=True).expand_as(x), x)
    return x


# %% [markdown]
# ## Модели

# %%
class TutorialCNN(nn.Module):
    """Архитектура CNN из учебного ноутбука: 3x(conv3x3-ReLU-MaxPool) 32/64/128 + FC 512 -> classes.
    Адаптация: вход 1 канал, 5 классов; перед Flatten AdaptiveAvgPool2d((4,4)) вместо фиксированного
    размера 4x4 (в туториале 32x32 даёт 4x4 после трёх пулингов; при другом img размер подгоняется).
    Dropout в туториале нет (по умолчанию 0)."""

    def __init__(self, num_classes=NUM_CLASSES, dropout=0.0):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512), nn.ReLU(), nn.Dropout(dropout), nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return self.fc_layers(x.flatten(1))


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + (x if self.short is None else self.short(x)))


class ResNet(nn.Module):
    def __init__(self, width=32, blocks=(2, 2, 2, 2), stem_pool=False, dropout=0.2, num_classes=NUM_CLASSES):
        super().__init__()
        stem = [nn.Conv2d(1, width, 3, 2, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU()]
        if stem_pool:
            stem.append(nn.MaxPool2d(3, 2, 1))
        self.stem = nn.Sequential(*stem)
        layers, cin = [], width
        for i, n in enumerate(blocks):
            cout = width * 2 ** i
            for j in range(n):
                layers.append(BasicBlock(cin, cout, (1 if i == 0 else 2) if j == 0 else 1))
                cin = cout
        self.layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(cin, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.layers(self.stem(x))
        return self.fc(self.drop(self.pool(x).flatten(1)))


def build_model(cfg):
    if cfg["model"] == "tutorial":
        m = TutorialCNN(dropout=cfg["dropout"])
    else:
        m = ResNet(cfg["width"], tuple(cfg["blocks"]), cfg["stem_pool"], cfg["dropout"])
    # channels_last на MPS в ~3 раза медленнее (замер) — не используем
    return m.to(DEVICE)


# %% [markdown]
# ## Обучение

# %%
DEFAULTS = dict(
    model="resnet", width=32, blocks=[2, 2, 2, 2], stem_pool=False, dropout=0.2, img=224, epochs=60,
    bs=32, lr=2e-3, wd=5e-4, sched="onecycle", ls=0.1, mixup=0.0,
    aug=dict(rot=15, scale=0.15, shift=0.08, bc=0.2, erase=0.0),
    seed=0, train_on="train", tta=None, pseudo=None, save_weights=False,
    amp=False,  # False | "fp16" | "bf16": autocast на MPS медленнее fp32 (замер), по умолчанию выключен
    eval_every=5,  # val-оценка каждые eval_every эпох и на всех последних 5
)
# tta=None -> True, если есть аугментации (aug), иначе False. flip: cfg["aug"]["flip"]=False отключает hflip.
AMP_DTYPES = dict(fp16=torch.float16, bf16=torch.bfloat16)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)


@torch.no_grad()
def predict(model, x_u8, mean, std, tta=False, bs=64):
    """softmax [N,5] (numpy). tta: False | True (оригинал+hflip) | 'full' (+ сдвиги ±4%)."""
    model.eval()
    out = []
    for i in range(0, len(x_u8), bs):
        x = x_u8[i:i + bs].to(DEVICE).float() / 255
        views = [x]
        if tta:
            views.append(x.flip(3))
        if tta == "full":
            z = torch.zeros(len(x), device=DEVICE)
            for dx, dy in ((0.04, 0), (-0.04, 0), (0, 0.04), (0, -0.04)):
                views.append(shift_images(x, z + dx, z + dy))
        p = 0
        for v in views:
            v = (v - mean) / std
            p = p + F.softmax(model(v), 1)
        out.append((p / len(views)).cpu())
    return torch.cat(out).numpy()


def read_results():
    path = os.path.join(RUNS, "results.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def find_run(name):
    """Последняя запись results.jsonl с этим именем (или None)."""
    rows = [r for r in read_results() if r.get("name") == name]
    return rows[-1] if rows else None


def run_train_on(row):
    return row.get("train_on") or row["cfg"].get("train_on", "train")


def load_pseudo(spec, d, train_on="train"):
    """Псевдо-метки от источников -> (x_u8, y, info).

    Схема трансдуктивная: источник обучен ТОЛЬКО на train; val/test идут как неразмеченные с псевдо-метками,
    val accuracy по истинным меткам остаётся честной (как на test). Такие запуски помечаются pl_val=True.
    spec: source (имя или список), targets (по умолчанию ['test']), thr (0.9), agree (False)."""
    src = spec["source"]
    src = [s.strip() for s in src.split(",")] if isinstance(src, str) else list(src)
    for s_name in src:
        row = find_run(s_name)
        if row is None:
            raise ValueError(f"источник {s_name!r} не найден в results.jsonl — не могу проверить утечки")
        if run_train_on(row) != "train":
            raise ValueError(f"источник {s_name!r} обучен на {run_train_on(row)!r}, нужен train_on='train'")
        sp = row["cfg"].get("pseudo")
        if sp and "val" in sp.get("targets", ["test"]):
            raise ValueError(f"источник {s_name!r} сам обучен с псевдо-метками val")
    targets = list(spec.get("targets", ["test"]))
    if train_on == "trainval" and "val" in targets:
        print("  ВНИМАНИЕ: train_on='trainval' -> 'val' удалён из targets псевдо-меток (val уже с истинными метками)",
              flush=True)
        targets.remove("val")
    thr, agree = spec.get("thr", 0.9), spec.get("agree", False)
    xs, ys = [], []
    info = dict(pl_val="val" in targets, n_pseudo=0, pseudo_acc_val=None)
    for split in targets:
        ps = [np.load(os.path.join(RUNS, f"{s_name}_{split}.npy")) for s_name in src]
        p = np.mean(ps, axis=0)
        mx = p.max(1)
        q = np.quantile(mx, [0.1, 0.25, 0.5, 0.75, 0.9])
        sel = mx >= thr
        if agree:
            am = np.stack([x.argmax(1) for x in ps])
            sel &= (am == am[0]).all(0)
        sel = torch.from_numpy(sel)
        pl = torch.from_numpy(p.argmax(1))
        msg = (f"  pseudo[{split}]: отобрано {int(sel.sum())}/{len(p)} (thr={thr}, agree={agree}); "
               f"квантили max-prob 0.1/0.25/0.5/0.75/0.9 = " + "/".join(f"{v:.3f}" for v in q))
        if split == "val":
            info["pseudo_acc_val"] = float((pl[sel] == d["val"]["y"][sel]).float().mean()) if sel.any() else None
            msg += f", точность псевдо-меток {info['pseudo_acc_val']}"
        print(msg, flush=True)
        xs.append(d[split]["x"][sel])
        ys.append(pl[sel])
        info["n_pseudo"] += int(sel.sum())
    return torch.cat(xs), torch.cat(ys), info


def make_optimizer(params, cfg):
    try:
        return torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["wd"], fused=True)
    except (RuntimeError, ValueError):
        return torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["wd"])


def prep_cfg(cfg):
    cfg = {**DEFAULTS, **cfg}
    if cfg["tta"] is None:
        cfg["tta"] = bool(cfg["aug"])
    return cfg


def train_loop(cfg, x_tr, y_tr, mean, std, eval_fn=None, tag=None):
    """Общий цикл обучения (cfg уже прошёл prep_cfg). x_tr uint8 [N,1,S,S], y_tr long [N]; mean/std — float.
    eval_fn(model, ep) -> строка для лога или '' (вызывается каждую эпоху). Возвращает обученную модель."""
    name = tag or cfg["name"]
    set_seed(cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    nrng = np.random.RandomState(cfg["seed"])
    xtr_dev, ytr_dev = x_tr.to(DEVICE), y_tr.to(DEVICE)
    N, bs, epochs = len(x_tr), cfg["bs"], cfg["epochs"]
    steps_ep = N // bs

    model = build_model(cfg)
    opt = make_optimizer(model.parameters(), cfg)
    if cfg["sched"] == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=cfg["lr"], total_steps=steps_ep * epochs, pct_start=0.15)
    else:
        warm, total = 3 * steps_ep, steps_ep * epochs
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / warm if s < warm
            else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    crit = nn.CrossEntropyLoss(label_smoothing=cfg["ls"])
    amp_dtype = AMP_DTYPES[cfg["amp"]] if cfg["amp"] else None

    mean_t = torch.tensor(mean, device=DEVICE)
    std_t = torch.tensor(std, device=DEVICE)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(N, generator=gen).to(DEVICE)
        aug_r = aug_params(steps_ep * bs, gen).view(steps_ep, bs, AUG_COLS) if cfg["aug"] else None
        loss_sum = torch.zeros((), device=DEVICE)
        correct = torch.zeros((), device=DEVICE)
        for i in range(steps_ep):
            idx = perm[i * bs:(i + 1) * bs]
            x = xtr_dev[idx].float() / 255
            y = ytr_dev[idx]
            if cfg["aug"]:
                x = augment(x, cfg["aug"], gen, aug_r[i])
            x = (x - mean_t) / std_t
            y2, lam = None, 1.0
            if cfg["mixup"] > 0 and nrng.rand() < 0.5:
                lam = float(nrng.beta(cfg["mixup"], cfg["mixup"]))
                j = torch.randperm(len(x), generator=gen).to(DEVICE)
                x = lam * x + (1 - lam) * x[j]
                y2 = y[j]
            with (torch.autocast("mps", dtype=amp_dtype) if amp_dtype else contextlib.nullcontext()):
                out = model(x)
                loss = crit(out, y) if y2 is None else lam * crit(out, y) + (1 - lam) * crit(out, y2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.detach().float() * len(y)
            correct += (out.argmax(1) == y).sum()
        last = ep + 1 == epochs
        extra = eval_fn(model, ep + 1) if eval_fn else ""
        if (ep + 1) % 5 == 0 or ep == 0 or last:
            print(f"[{name}] ep {ep + 1}/{epochs} loss {loss_sum.item() / (steps_ep * bs):.4f} "
                  f"train_acc {correct.item() / (steps_ep * bs):.4f} " + extra + f"t {time.time() - t0:.0f}s",
                  flush=True)
    return model


def run_config(cfg):
    cfg = prep_cfg(cfg)
    name = cfg["name"]
    trainval = cfg["train_on"] == "trainval"
    S = cfg["img"]
    d = get_data(S)

    xtr, ytr = d["train"]["x"], d["train"]["y"]
    mean = (xtr.float() / 255).mean().item()
    std = (xtr.float() / 255).std().item()
    if trainval:
        xtr, ytr = torch.cat([xtr, d["val"]["x"]]), torch.cat([ytr, d["val"]["y"]])
    pl_info = dict(pl_val=False, n_pseudo=0, pseudo_acc_val=None)
    if cfg["pseudo"]:
        px, py, pl_info = load_pseudo(cfg["pseudo"], d, cfg["train_on"])
        xtr, ytr = torch.cat([xtr, px]), torch.cat([ytr, py])

    mean_t = torch.tensor(mean, device=DEVICE)
    std_t = torch.tensor(std, device=DEVICE)
    epochs = cfg["epochs"]
    xval, yval = d["val"]["x"], d["val"]["y"].numpy()
    uniq = d["val"]["unique"].numpy()
    val_hist, uval_hist = [], []  # пары (epoch, acc); для trainval пусто

    def eval_fn(model, ep):
        if trainval or not ((ep) % cfg["eval_every"] == 0 or ep > epochs - 5):
            return ""
        pv = predict(model, xval, mean_t, std_t).argmax(1)
        val_hist.append([ep, float((pv == yval).mean())])
        uval_hist.append([ep, float((pv == yval)[uniq].mean())])
        return f"val {val_hist[-1][1]:.4f} val_unique {uval_hist[-1][1]:.4f} "

    t0 = time.time()
    model = train_loop(cfg, xtr, ytr, mean, std, eval_fn)
    train_sec = time.time() - t0
    n_params = sum(p.numel() for p in model.parameters())

    # финал: последняя эпоха (не best-checkpoint)
    pt = predict(model, d["test"]["x"], mean_t, std_t, cfg["tta"])
    np.save(os.path.join(RUNS, f"{name}_test.npy"), pt)
    if cfg["save_weights"]:
        os.makedirs(WEIGHTS, exist_ok=True)
        torch.save(dict(state_dict=model.state_dict(), cfg=cfg, mean=mean, std=std),
                   os.path.join(WEIGHTS, f"{name}.pt"))
    res = dict(name=name, cfg=cfg, train_on=cfg["train_on"], **pl_info,
               sec=round(train_sec, 1), params=n_params)
    if trainval:
        # val в обучении -> val-метрик нет, {name}_val.npy не сохраняем
        return dict(res, val_acc=None, val_acc_tta=None, val_unique_acc_tta=None,
                    best_epoch=None, best_val=None, val_hist=None, val_unique_hist=None)
    pv_plain = predict(model, xval, mean_t, std_t)
    pv = predict(model, xval, mean_t, std_t, cfg["tta"]) if cfg["tta"] else pv_plain
    np.save(os.path.join(RUNS, f"{name}_val.npy"), pv)
    ok = pv.argmax(1) == yval
    best = int(np.argmax([h[1] for h in val_hist]))
    return dict(
        res,
        val_acc=float((pv_plain.argmax(1) == yval).mean()),
        val_acc_tta=float(ok.mean()),
        val_unique_acc_tta=float(ok[uniq].mean()),
        best_epoch=val_hist[best][0], best_val=val_hist[best][1],
        val_hist=[[e, round(v, 4)] for e, v in val_hist],
        val_unique_hist=[[e, round(v, 4)] for e, v in uval_hist],
    )


# %% [markdown]
# ## Ансамбль, журнал, вывод

# %%
def log_results(results):
    with open(os.path.join(RUNS, "results.jsonl"), "a") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def show(results):
    cvs = [r for r in results if r.get("kind") == "cv"]
    for r in cvs:
        print(f"{r['name']}: OOF acc {r['oof_acc']:.4f}, ошибок {r['n_err']}, nll {r['oof_nll']:.4f}, "
              f"{r['minutes']:.1f} мин")
    results = [r for r in results if r.get("kind") != "cv"]
    if not results:
        return
    df = pd.DataFrame(results)
    df["minutes"] = df["sec"] / 60
    cols = ["name", "train_on", "pl_val", "val_acc", "val_acc_tta", "val_unique_acc_tta", "best_epoch", "best_val",
            "minutes"]
    print(df[[c for c in cols if c in df]].round(4).to_string(index=False))


def ensemble(names):
    """Ансамбль по val: только модели, обученные на train (иначе val не честный)."""
    d = get_data(DEFAULTS["img"])
    y = d["val"]["y"].numpy()
    uniq = d["val"]["unique"].numpy()
    ps = []
    for n in names:
        row = find_run(n)
        if row is not None and run_train_on(row) != "train":
            raise ValueError(f"{n!r} обучен на {run_train_on(row)!r}: val-оценка ансамбля недействительна")
        path = os.path.join(RUNS, f"{n}_val.npy")
        if not os.path.exists(path):
            raise ValueError(f"нет {n}_val.npy (модель на trainval или не запускалась)")
        ps.append(np.load(path))
    p = np.mean(ps, axis=0)
    ok = p.argmax(1) == y
    print(f"ens {names}: val {ok.mean():.4f}, val_unique {ok[uniq].mean():.4f}")
    return dict(name="ens:" + "+".join(names), members=list(names), val_acc=None,
                val_acc_tta=float(ok.mean()), val_unique_acc_tta=float(ok[uniq].mean()), sec=0.0)


def ensemble_test(names, out_csv):
    """Среднее runs/{n}_test.npy (любые train_on) -> argmax -> CSV id,label; проверка check_submission."""
    ids = load_split("test", DEFAULTS["img"])["ids"]
    p = np.mean([np.load(os.path.join(RUNS, f"{n}_test.npy")) for n in names], axis=0)
    if len(p) != len(ids):
        raise ValueError(f"{len(p)} предсказаний, {len(ids)} id")
    pd.DataFrame({"id": ids, "label": p.argmax(1)}).to_csv(out_csv, index=False, lineterminator="\n")
    errors, sha, n = check_submission.check(out_csv)
    if errors:
        print("FAIL:", *errors, sep="\n  - ")
    else:
        print(f"OK: {n} строк, sha256={sha}")
    return errors, sha


# %% [markdown]
# ## Кросс-валидация по train+val (группы почти-дублей), CV-псевдолейблинг

# %%
BASELINE_CSV = os.path.join(HERE, "submissions", "02_res_aug_strong_trainval_5seeds.csv")


def cv_data(S):
    """train+val подряд: (x uint8, y, ids с префиксом сплита 'train/' / 'val/')."""
    d = get_data(S)
    x = torch.cat([d["train"]["x"], d["val"]["x"]])
    y = torch.cat([d["train"]["y"], d["val"]["y"]])
    ids = ["train/" + i for i in d["train"]["ids"]] + ["val/" + i for i in d["val"]["ids"]]
    return x, y, ids


def dup_groups():
    """Компоненты связности по корреляции 16x16-векторов > UNIQUE_THR среди всех train+val -> group id [N]."""
    cache = os.path.join(RUNS, "cache_cv_groups.npy")
    if os.path.exists(cache):
        return np.load(cache)
    v = np.concatenate([small_vectors("train"), small_vectors("val")])
    corr = v @ v.T
    n = len(v)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in zip(*np.where(np.triu(corr > UNIQUE_THR, 1))):
        parent[find(i)] = find(j)
    roots = np.array([find(i) for i in range(n)])
    g = np.unique(roots, return_inverse=True)[1]
    np.save(cache, g)
    return g


def cv_folds(S, n_folds=5, seed=0):
    """Номер фолда на объект train+val (StratifiedGroupKFold по группам почти-дублей), кэш runs/cv_folds_s{seed}.npy."""
    from sklearn.model_selection import StratifiedGroupKFold
    _, y, _ = cv_data(S)
    y = y.numpy()
    g = dup_groups()
    suffix = "" if n_folds == 5 else f"_k{n_folds}"
    cache = os.path.join(RUNS, f"cv_folds_s{seed}{suffix}.npy")
    if os.path.exists(cache):
        folds = np.load(cache)
    else:
        folds = np.full(len(y), -1)
        for k, (_, te) in enumerate(StratifiedGroupKFold(n_folds, shuffle=True, random_state=seed).split(y, y, g)):
            folds[te] = k
        np.save(cache, folds)
    sizes = np.bincount(g)
    print(f"cv_folds(seed={seed}): {len(y)} объектов, групп {len(sizes)}, групп с >1 объектом {int((sizes > 1).sum())} "
          f"(объектов в них {int(sizes[sizes > 1].sum())}, макс. группа {sizes.max()})", flush=True)
    for k in range(n_folds):
        m = folds == k
        print(f"  fold {k}: {int(m.sum())} объектов, классы {np.bincount(y[m], minlength=NUM_CLASSES).tolist()}",
              flush=True)
    return folds


def error_report(p, y, ids):
    """p [N,5], y [N] -> acc, n_err, nll, пары 't->p': n, id ошибок."""
    pred = p.argmax(1)
    wrong = np.where(pred != y)[0]
    pairs = {}
    for i in wrong:
        key = f"{y[i]}->{pred[i]}"
        pairs[key] = pairs.get(key, 0) + 1
    nll = float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())
    return dict(acc=float((pred == y).mean()), n_err=int(len(wrong)), nll=nll, pairs=pairs,
                err_ids=[ids[i] for i in wrong])


def cv_pseudo_fold(spec, k, fold_idx, y_all, S, folds_seed, n_folds):
    """Псевдо-метки для фолда k: неразмеченные = отложенный фолд k + test. -> (x_u8, y, info)."""
    src = spec["sources"]
    src = [s.strip() for s in src.split(",")] if isinstance(src, str) else list(src)
    thr, agree = spec.get("thr", 0.9), spec.get("agree", True)
    oofs, tests = [], []
    for s_name in src:
        row = find_run(s_name)
        if row is None or row.get("kind") != "cv":
            raise ValueError(f"источник {s_name!r} не найден или не kind='cv'")
        assert row.get("cv_seed") == folds_seed and row.get("n_folds") == n_folds, \
            f"источник {s_name!r} использовал другие фолды (seed {row.get('cv_seed')}, n_folds {row.get('n_folds')})"
        if row["cfg"].get("pseudo_cv"):
            print(f"  ВНИМАНИЕ: источник {s_name!r} сам обучен с pseudo_cv", flush=True)
        oof = np.load(os.path.join(RUNS, f"cv_{s_name}_oof.npy"))
        assert not np.isnan(oof[fold_idx]).any(), f"в OOF {s_name!r} нет предсказаний для фолда {k}"
        oofs.append(oof[fold_idx])
        tf_path = os.path.join(RUNS, f"cv_{s_name}_test_folds.npy")
        if not os.path.exists(tf_path):
            raise ValueError(f"у источника {s_name!r} нет cv_{s_name}_test_folds.npy — перезапустить его")
        tf = np.load(tf_path)
        assert not np.isnan(tf[k]).any(), f"в test_folds {s_name!r} нет предсказаний фолда {k}"
        tests.append(tf[k])  # только модель фолда k: она не видела фолд k
    d = get_data(S)
    x_all = cv_data(S)[0]
    xs, ys, info = [], [], dict(n_pseudo_fold=0, n_pseudo_test=0, pseudo_acc_fold=None)
    for tag, ps, x in (("fold", oofs, x_all[fold_idx]), ("test", tests, d["test"]["x"])):
        p = np.mean(ps, axis=0)
        sel = p.max(1) >= thr
        if agree:
            am = np.stack([q.argmax(1) for q in ps])
            sel &= (am == am[0]).all(0)
        pl = p.argmax(1)
        info[f"n_pseudo_{tag}"] = int(sel.sum())
        msg = f"  pseudo[{tag}]: отобрано {int(sel.sum())}/{len(p)} (thr={thr}, agree={agree})"
        if tag == "fold":
            if sel.any():
                info["pseudo_acc_fold"] = float((pl[sel] == y_all[fold_idx][sel]).mean())
            msg += f", точность псевдо-меток на фолде {info['pseudo_acc_fold']}"
        print(msg, flush=True)
        sel_t = torch.from_numpy(sel)
        xs.append(x[sel_t])
        ys.append(torch.from_numpy(pl[sel]))
    return torch.cat(xs), torch.cat(ys), info


def run_cv(cfg):
    cfg = prep_cfg(cfg)
    name, S = cfg["name"], cfg["img"]
    n_folds, folds_seed = cfg.get("n_folds", 5), cfg.get("cv_seed", 0)
    only = cfg.get("only_folds")
    x_all, y_all, ids = cv_data(S)
    y_np = y_all.numpy()
    folds = cv_folds(S, n_folds, folds_seed)
    test_x = get_data(S)["test"]["x"]
    oof = np.full((len(y_all), NUM_CLASSES), np.nan, dtype=np.float32)
    tests, fold_info = [], []
    test_folds = np.full((n_folds, len(test_x), NUM_CLASSES), np.nan, dtype=np.float32)
    t_all = time.time()
    for k in range(n_folds):
        if only is not None and k not in only:
            continue
        t0 = time.time()
        te_idx = np.where(folds == k)[0]
        tr_idx = np.where(folds != k)[0]
        xtr, ytr = x_all[tr_idx], y_all[tr_idx]
        mean = (xtr.float() / 255).mean().item()
        std = (xtr.float() / 255).std().item()
        info = {}
        if cfg.get("pseudo_cv"):
            px, py, info = cv_pseudo_fold(cfg["pseudo_cv"], k, te_idx, y_np, S, folds_seed, n_folds)
            xtr, ytr = torch.cat([xtr, px]), torch.cat([ytr, py])
        fcfg = {**cfg, "seed": cfg["seed"] + k}
        model = train_loop(fcfg, xtr, ytr, mean, std, tag=f"{name}/f{k}")
        mean_t = torch.tensor(mean, device=DEVICE)
        std_t = torch.tensor(std, device=DEVICE)
        oof[te_idx] = predict(model, x_all[te_idx], mean_t, std_t, cfg["tta"])
        tests.append(predict(model, test_x, mean_t, std_t, cfg["tta"]))
        test_folds[k] = tests[-1]
        nerr = int((oof[te_idx].argmax(1) != y_np[te_idx]).sum())
        sec = time.time() - t0
        fold_info.append(dict(fold=k, n=len(te_idx), n_err=nerr, sec=round(sec, 1), **info))
        print(f"[{name}] fold {k}: acc {1 - nerr / len(te_idx):.4f}, ошибок {nerr}/{len(te_idx)}, {sec / 60:.1f} мин",
              flush=True)
        np.save(os.path.join(RUNS, f"cv_{name}_oof_partial.npy"), oof)
        np.save(os.path.join(RUNS, f"cv_{name}_test_partial.npy"), np.mean(tests, axis=0))
        np.save(os.path.join(RUNS, f"cv_{name}_test_folds_partial.npy"), test_folds)
        del model
    done = ~np.isnan(oof).any(1)
    rep = error_report(oof[done], y_np[done], [i for i, m in zip(ids, done) if m])
    test_p = np.mean(tests, axis=0)
    np.save(os.path.join(RUNS, f"cv_{name}_oof.npy"), oof)
    np.save(os.path.join(RUNS, f"cv_{name}_test.npy"), test_p)
    np.save(os.path.join(RUNS, f"cv_{name}_test_folds.npy"), test_folds)
    for suf in ("oof", "test", "test_folds"):
        part = os.path.join(RUNS, f"cv_{name}_{suf}_partial.npy")
        if os.path.exists(part):
            os.remove(part)
    print(f"[{name}] OOF acc {rep['acc']:.4f}, ошибок {rep['n_err']}/{int(done.sum())}, nll {rep['nll']:.4f}, "
          f"пары {rep['pairs']}", flush=True)
    return dict(name=name, kind="cv", cfg=cfg, cv_seed=folds_seed, n_folds=n_folds, only_folds=only,
                oof_acc=rep["acc"], n_err=rep["n_err"], oof_nll=rep["nll"], folds=fold_info,
                err_pairs=rep["pairs"], err_ids=rep["err_ids"], minutes=round((time.time() - t_all) / 60, 2),
                sec=round(time.time() - t_all, 1))


def cv_ensemble(names):
    """Ансамбль OOF cv-прогонов + сравнение test-предсказания с baseline CSV."""
    rows = []
    for n in names:
        row = find_run(n)
        if row is None or row.get("kind") != "cv":
            raise ValueError(f"{n!r} не найден или не kind='cv'")
        rows.append(row)
    assert len({(r["cv_seed"], r["n_folds"]) for r in rows}) == 1, "разные фолды у членов ансамбля"
    _, y, ids = cv_data(DEFAULTS["img"])
    y = y.numpy()
    oof = np.mean([np.load(os.path.join(RUNS, f"cv_{n}_oof.npy")) for n in names], axis=0)
    if np.isnan(oof).any():
        raise ValueError("OOF содержит NaN (неполный прогон)")
    rep = error_report(oof, y, ids)
    print(f"cv_ens {names}: OOF acc {rep['acc']:.4f}, ошибок {rep['n_err']}, nll {rep['nll']:.4f}, "
          f"пары {rep['pairs']}\n  ошибки: {rep['err_ids']}", flush=True)
    tp = np.mean([np.load(os.path.join(RUNS, f"cv_{n}_test.npy")) for n in names], axis=0)
    test_ids = load_split("test", DEFAULTS["img"])["ids"]
    out = dict(name="cvens:" + "+".join(names), kind="cv_ens", members=list(names), oof_acc=rep["acc"],
               n_err=rep["n_err"], oof_nll=rep["nll"], err_pairs=rep["pairs"], err_ids=rep["err_ids"], sec=0.0)
    if os.path.exists(BASELINE_CSV):
        base = pd.read_csv(BASELINE_CSV).set_index("id")["label"].reindex(test_ids).to_numpy()
        diff = np.where(tp.argmax(1) != base)[0]
        print(f"  test: {len(diff)} объектов меняют argmax относительно {os.path.basename(BASELINE_CSV)}")
        for i in diff:
            print(f"    {test_ids[i]}: было {base[i]} -> {tp[i].argmax()}, p={[round(float(v), 3) for v in tp[i]]}")
        out["test_changed"] = [test_ids[i] for i in diff]
    else:
        print(f"  нет {BASELINE_CSV}")
    np.save(os.path.join(RUNS, "cvens_" + "+".join(names) + "_test.npy"), tp)
    return out


# %% [markdown]
# ## Группы конфигов (новые группы — добавлять в GROUPS_CFG)

# %%
def C(name, **kw):
    return {"name": name, **kw}


GROUPS_CFG = {
    "smoke": [C("smoke", epochs=2, img=128)],
    "base": [
        C("tut_noaug", model="tutorial", aug=None, epochs=40, img=224, ls=0.0, sched="cosine",
          lr=1e-3, wd=0.0, dropout=0.0, tta=False),
        C("res_noaug", aug=None, epochs=40, tta=False),
        C("res_aug", epochs=60),
    ],
    "aug": [
        C("res_aug_strong", aug=dict(rot=20, scale=0.2, shift=0.1, bc=0.3, erase=0.25)),
        C("res_aug_light", aug=dict(rot=10, scale=0.1, shift=0.05, bc=0.1, erase=0.0)),
    ],
    "arch": [
        C("res_w48", width=48),
        C("res_img256", img=256),
        C("res_img288_pool", img=288, stem_pool=True),
        C("res_e100", epochs=100),
    ],
    "reg": [
        C("res_mixup", mixup=0.2),
        C("res_ls0", ls=0.0),
        C("res_wd5e-3", wd=5e-3),
    ],
    "seeds": [
        C("res_aug_s1", seed=1),
        C("res_aug_s2", seed=2),
    ],
}

STRONG = dict(aug=dict(rot=20, scale=0.2, shift=0.1, bc=0.3, erase=0.25), epochs=60, img=224, ls=0.1, width=32,
              tta=True, seed=0)
GROUPS_CFG.update({
    "cv_smoke": [C("cv_smoke", **{**STRONG, "epochs": 2, "img": 128, "n_folds": 5})],
    "cv_main": [C("cv_strong", **STRONG), C("cv_w48", **{**STRONG, "width": 48})],
    "cv_pl": [C("cv_strong_pl", **STRONG,
                pseudo_cv=dict(sources=["cv_strong", "cv_w48"], thr=0.9, agree=True))],
    "cv_256": [C("cv_strong_256", **{**STRONG, "img": 256})],
})


# %%
if __name__ == "__main__" and len(sys.argv) > 1:
    print(f"device: {DEVICE}", flush=True)
    if sys.argv[1] == "ens":
        log_results([ensemble(sys.argv[2:])])
    elif sys.argv[1] == "cvens":
        log_results([cv_ensemble(sys.argv[2:])])
    elif sys.argv[1] == "enstest":
        ensemble_test(sys.argv[3:], sys.argv[2])
    else:
        results = []
        for g in sys.argv[1:]:
            for cfg in GROUPS_CFG[g]:
                r = run_cv(cfg) if g.startswith("cv") else run_config(cfg)
                log_results([r])  # пишем сразу: падение посреди группы не теряет сделанное
                results.append(r)
        show(results)
