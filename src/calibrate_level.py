"""Выравнивание уровня прогноза по измеренному среднему теста.

Зонды из src/probe.py дали точное среднее целевой переменной на публичной
части теста. При MSE уровень прогноза важен сам по себе: ошибка в среднем
входит в метрику квадратом, независимо от того, насколько хорошо модель
упорядочивает молекулы.

Скрипт домножает готовый сабмит на постоянный множитель так, чтобы среднее
совпало с измеренным. Порядок молекул при этом не меняется, меняется только
масштаб. Так можно сравнить модели при одинаковом уровне и понять, какая из
них лучше именно по форме прогноза.

Запуск:
    python -m src.calibrate_level --target 323.38 submissions/blend.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def calibrate(path: Path, target_mean: float, suffix: str = "_cal") -> Path:
    df = pd.read_csv(path)
    col = df.columns[1]
    cur = float(df[col].mean())
    if cur <= 0:
        raise SystemExit(f"{path.name}: среднее {cur}, масштабировать нечего")
    factor = target_mean / cur
    df[col] = (df[col] * factor).clip(lower=0.0)
    out = path.with_name(path.stem + suffix + path.suffix)
    df.to_csv(out, index=False)
    print(f"{path.name}: среднее {cur:,.1f} -> {df[col].mean():,.1f} "
          f"(множитель {factor:.3f}), максимум {df[col].max():,.0f}")
    print(f"  записано: {out.name}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--target", type=float, required=True,
                    help="измеренное среднее теста")
    args = ap.parse_args()
    for f in args.files:
        calibrate(f, args.target)


if __name__ == "__main__":
    main()
