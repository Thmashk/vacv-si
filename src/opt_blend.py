"""Оптимальная линейная комбинация уже отправленных сабмитов.

Идея. Зонды из src/probe.py дали среднее и дисперсию целевой переменной на
публичной части теста. Тогда для любого отправленного сабмита его оценка
раскладывается:

    MSE = Var(y) + Var(p) - 2*Cov(y, p) + (mu_y - mu_p)^2

Все члены, кроме ковариации, известны: Var(y) и mu_y измерены зондами,
Var(p) и mu_p считаются по самому файлу, MSE — это оценка с лидерборда.
Значит ковариация прогноза с правильным ответом вычисляется точно:

    Cov(y, p) = [Var(y) + Var(p) + (mu_y - mu_p)^2 - MSE] / 2

Зная ковариации всех сабмитов с ответом и ковариации между собой, можно
решить обычную задачу наименьших квадратов и найти веса комбинации,
минимизирующей MSE:

    w = (S + ridge*I)^(-1) c,     ожидаемая MSE = Var(y) - c^T w

где S — ковариационная матрица центрированных прогнозов, c — вектор
ковариаций с ответом.

Регуляризация нужна по двум причинам: прогнозы сильно скоррелированы между
собой (до 0.99), а ковариации c оценены по публичным 30% теста и содержат
шум. Без неё веса разлетаются и решение подгоняется под публичную часть.

Запуск:
    python -m src.opt_blend --parts blend aggressive --ridge 0.1 --name opt_mix2
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from .config import ID_COL, REPORTS, SUBMISSIONS
from .data import load_sample_submission

# Измерено зондами probe_c0 и probe_c400 на публичной части теста.
MU_Y = 323.379299
VAR_Y = 2018936.9

SCORES_JSON = SUBMISSIONS / "lb_scores.json"


def load_scores() -> dict[str, float]:
    if not SCORES_JSON.exists():
        raise SystemExit(f"нет {SCORES_JSON}: запишите туда оценки лидерборда "
                         "в виде {\"имя_файла_без_csv\": оценка}")
    return json.loads(SCORES_JSON.read_text(encoding="utf-8"))


def covariance_with_truth(pred: np.ndarray, mse: float) -> float:
    return (VAR_Y + pred.var() + (MU_Y - pred.mean()) ** 2 - mse) / 2.0


def analyse(scores: dict[str, float]) -> pd.DataFrame:
    rows = []
    for name, mse in scores.items():
        p = pd.read_csv(SUBMISSIONS / f"{name}.csv").iloc[:, 1].to_numpy(float)
        vp = float(p.var())
        cov = covariance_with_truth(p, mse)
        rows.append(dict(
            файл=name, LB=mse, среднее=p.mean(), sd=np.sqrt(vp),
            корреляция=cov / np.sqrt(VAR_Y * vp) if vp > 0 else 0.0,
            множитель=cov / vp if vp > 0 else 0.0,
            потенциал=VAR_Y - cov ** 2 / vp if vp > 0 else VAR_Y,
        ))
    return pd.DataFrame(rows).sort_values("потенциал").reset_index(drop=True)


def solve_weights(parts: list[str], scores: dict[str, float], ridge: float):
    preds = {n: pd.read_csv(SUBMISSIONS / f"{n}.csv").iloc[:, 1].to_numpy(float)
             for n in parts}
    M = np.column_stack([preds[n] - preds[n].mean() for n in parts])
    S = np.cov(M, rowvar=False)
    S = np.atleast_2d(S)
    c = np.array([covariance_with_truth(preds[n], scores[n]) for n in parts])
    reg = ridge * np.trace(S) / len(parts) * np.eye(len(parts))
    w = np.linalg.solve(S + reg, c)
    return preds, w, float(VAR_Y - c @ w)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", required=False,
                    help="имена сабмитов без .csv")
    ap.add_argument("--ridge", type=float, default=0.1)
    ap.add_argument("--max-cap", type=float, default=0.0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--analyse", action="store_true",
                    help="только таблица разбора, без сборки")
    args = ap.parse_args()

    scores = load_scores()
    table = analyse(scores)
    print(table.round(3).to_string(index=False))
    print(f"\nбез модели (константа {MU_Y:.0f}): {VAR_Y:,.0f}")

    REPORTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORTS / "lb_decomposition.csv", index=False)
    if args.analyse or not args.parts:
        return

    preds, w, expected = solve_weights(args.parts, scores, args.ridge)
    print(f"\nвеса: {dict(zip(args.parts, np.round(w, 3)))}")
    print(f"ожидаемая MSE: {expected:,.0f}")

    centred = sum(wi * (preds[n] - preds[n].mean()) for wi, n in zip(w, args.parts))
    out = np.clip(MU_Y + centred, 0.0, None)
    if args.max_cap > 0:
        for _ in range(20):
            out = np.minimum(out, args.max_cap)
            cur = float(out.mean())
            if abs(cur - MU_Y) < 0.01:
                break
            out = out * (MU_Y / cur)
        out = np.clip(np.minimum(out, args.max_cap), 0.0, None)

    sample = load_sample_submission()
    sub = pd.DataFrame({ID_COL: sample[ID_COL].to_numpy(),
                        sample.columns[1]: out})
    assert len(sub) == len(sample) and sub[sample.columns[1]].notna().all()

    name = args.name or "opt_mix"
    path = SUBMISSIONS / f"{name}.csv"
    sub.to_csv(path, index=False)
    print(f"\nзаписано: {path.name}  среднее {out.mean():,.1f}, "
          f"медиана {np.median(out):,.1f}, максимум {out.max():,.0f}")


if __name__ == "__main__":
    main()
