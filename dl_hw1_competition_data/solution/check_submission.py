"""Проверка формата submission.csv по правилам задачи B.

Использование: python check_submission.py path/to/submission.csv
"""
import hashlib
import math
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TEST_PATH = os.path.join(HERE, "..", "test.csv")
MAX_BYTES = 1024 * 1024


def check(path):
    errors = []

    size = os.path.getsize(path)
    if size > MAX_BYTES:
        errors.append(f"размер {size} байт > 1 MiB")

    with open(path, encoding="utf-8") as f:
        header = f.readline().strip()
    if header != "id,anomaly_score":
        errors.append(f"заголовок {header!r}, ожидается 'id,anomaly_score'")

    sub = pd.read_csv(path)
    test_ids = set(pd.read_csv(TEST_PATH, usecols=["id"])["id"])

    if list(sub.columns) != ["id", "anomaly_score"]:
        errors.append(f"колонки {list(sub.columns)}")
    else:
        if sub["id"].isna().any() or not pd.api.types.is_integer_dtype(sub["id"]):
            errors.append("id не целые или есть пропуски")
        if sub["id"].duplicated().any():
            errors.append(f"повторные id: {sub['id'].duplicated().sum()}")
        ids = set(sub["id"])
        if ids - test_ids:
            errors.append(f"лишние id: {len(ids - test_ids)}")
        if test_ids - ids:
            errors.append(f"пропущенные id: {len(test_ids - ids)}")

        scores = pd.to_numeric(sub["anomaly_score"], errors="coerce")
        bad = scores.isna() | ~scores.map(lambda v: math.isfinite(v) if pd.notna(v) else False)
        if bad.any():
            errors.append(f"нечисловые/NaN/inf скоры: {int(bad.sum())}")
        elif scores.nunique() < 2:
            errors.append("скор константный — AP будет равна доле аномалий")

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
