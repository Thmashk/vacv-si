"""Подстановка значений только для точных копий молекул из трейна.

Смесь по соседям (src/neighbors.py) на публичной части ухудшила результат:
без выброса её корреляция с ответом 0.427 против 0.582 у opt_mix3. То есть
частичное сходство вводит в заблуждение чаще, чем помогает.

Но точная копия — другое дело. Если у молекулы теста сходство Tanimoto
не меньше 0.99 с обучающей молекулой и совпадает клеточная культура, это
то же вещество в том же эксперименте, и его SI известен.

Скрипт берёт готовый сабмит и заменяет значения только в таких строках,
ничего больше не трогая. Так гипотеза проверяется в чистом виде.

Запуск:
    python -m src.exact_copies submissions/opt_mix3.csv --name mix3_exact
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SUBMISSIONS, TARGET
from .data import load_test, load_train
from .neighbors import fp_matrix, tanimoto


def exact_matches(threshold: float = 0.99, require_culture: bool = True):
    """Индексы строк теста с точной копией в трейне и их известные SI."""
    train, test = load_train(), load_test()
    sim = tanimoto(fp_matrix(test["smiles_std"]), fp_matrix(train["smiles_std"]))
    best = sim.argmax(1)
    top = sim.max(1)

    cu_test = test["Culture"].astype(str).str.strip().str.lower().to_numpy()
    cu_train = train["Culture"].astype(str).str.strip().str.lower().to_numpy()
    same_culture = cu_test == cu_train[best]

    mask = top >= threshold
    if require_culture:
        mask &= same_culture
    values = train[TARGET].to_numpy(dtype=float)[best]
    return np.where(mask)[0], values, top, best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path, help="готовый сабмит, который правим")
    ap.add_argument("--threshold", type=float, default=0.99)
    ap.add_argument("--any-culture", action="store_true",
                    help="подставлять и при несовпадении культуры")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    sub = pd.read_csv(args.source)
    col = sub.columns[1]
    pred = np.array(sub[col].to_numpy(dtype=float), copy=True)

    idx, values, top, _best = exact_matches(args.threshold, not args.any_culture)
    before = pred.copy()
    pred[idx] = values[idx]

    changed = int((np.abs(pred - before) > 1e-6).sum())
    big = idx[values[idx] > 10000]
    print(f"точных копий (сходство >= {args.threshold}"
          f"{', культура совпадает' if not args.any_culture else ''}): {len(idx)}")
    print(f"реально изменено значений: {changed}")
    print(f"из них со значением выше 10 000: {len(big)}  {list(big)}")
    print(f"среднее {before.mean():,.1f} -> {pred.mean():,.1f}, "
          f"максимум {before.max():,.0f} -> {pred.max():,.0f}")

    name = args.name or (args.source.stem + "_exact")
    out = SUBMISSIONS / f"{name}.csv"
    pd.DataFrame({sub.columns[0]: sub.iloc[:, 0], col: np.clip(pred, 0, None)}).to_csv(
        out, index=False)
    print(f"записано: {out.name}")


if __name__ == "__main__":
    main()
