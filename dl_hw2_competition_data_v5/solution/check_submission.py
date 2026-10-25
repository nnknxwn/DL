"""Проверка формата submission.csv (id,label; 5 классов).

Использование: python check_submission.py path/to/submission.csv
"""
import hashlib
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PATH = os.path.join(HERE, "..", "mri_classification_dataset", "sample_submission.csv")
MAX_BYTES = 1024 * 1024
EXPECTED_LINES = 301
LABEL_RE = re.compile(r"^[0-4]$")


def check(path):
    errors = []

    size = os.path.getsize(path)
    if size > MAX_BYTES:
        errors.append(f"размер {size} байт > 1 MiB")

    with open(path, "rb") as f:
        raw = f.read()
    header = ""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        errors.append("файл не в UTF-8")
        text = None
    if text is not None:
        # сырой текст: допускаются \n и \r\n; BOM не допускается (заголовок тогда не 'id,label')
        if text.endswith("\r\n"):
            text = text[:-2]
        elif text.endswith("\n"):
            text = text[:-1]
        lines = text.replace("\r\n", "\n").split("\n")
        header = lines[0]
        if len(lines) != EXPECTED_LINES:
            errors.append(f"строк {len(lines)}, ожидается {EXPECTED_LINES} (заголовок + 300)")
        if any(ln == "" for ln in lines):
            errors.append(f"пустые строки: {sum(ln == '' for ln in lines)}")
        bad_commas = sum(ln.count(",") != 1 for ln in lines[1:] if ln != "")
        if bad_commas:
            errors.append(f"строк данных не с одной запятой: {bad_commas}")
    if header != "id,label":
        errors.append(f"заголовок {header!r}, ожидается 'id,label'")

    # всё читаем как строки: ловим 2.0, NaN, inf, пустые значения
    try:
        sub = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8", encoding_errors="replace")
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        errors.append(f"не удалось разобрать CSV: {type(e).__name__}")
        return errors, hashlib.sha256(raw).hexdigest(), 0
    sample_ids = set(pd.read_csv(SAMPLE_PATH, dtype=str, keep_default_na=False)["id"])

    if list(sub.columns) != ["id", "label"]:
        errors.append(f"колонки {list(sub.columns)}")
    else:
        if (sub["id"] == "").any():
            errors.append("пустые id")
        if sub["id"].duplicated().any():
            errors.append(f"повторные id: {int(sub['id'].duplicated().sum())}")
        ids = set(sub["id"])
        if ids - sample_ids:
            errors.append(f"лишние id: {len(ids - sample_ids)}")
        if sample_ids - ids:
            errors.append(f"пропущенные id: {len(sample_ids - ids)}")

        bad = ~sub["label"].map(lambda v: bool(LABEL_RE.match(v)))
        if bad.any():
            ex = sub.loc[bad, "label"].unique()[:5].tolist()
            errors.append(f"label вне ^[0-4]$: {int(bad.sum())}, примеры {ex}")

    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()

    return errors, sha, len(sub)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "submission.csv"
    errors, sha, n = check(path)
    if errors:
        print("FAIL:")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    print(f"OK: {n} строк, sha256={sha}")
