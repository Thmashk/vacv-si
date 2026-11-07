"""Какие файлы выбирать финальными (R4). Монте-Карло по ключевым четырём строкам.

Строки 8b, 8c, 7e, 7l почти наверняка приватные, поэтому публичный счёт не
отличает файлы, различающиеся только на них. Чем отличаются файлы, решает
выбор ставки: если засчитывается один файл, нужен файл с наименьшей ожидаемой
ошибкой; если засчитывается лучший из двух выбранных, нужна пара с наименьшим
E[min]. Остальные строки у файлов общие, их вклад в сравнение сокращается.

Распределения берутся из аудита (reports/r4_audit/calibration.csv, квантили
q10/q50/q90 по спецификациям S2–S5 и двум структурам: «full» — стереоконфигурация
решает, «skel» — только каркас). Внутри структуры строки считаются независимыми.

    python -m src.exp_pair_choice
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .config import REPORTS, SUBMISSIONS, TARGET

ROWS = {160: "8b", 56: "8c", 233: "7e", 326: "7l"}
SPECS = (("S2", 0.864), ("S3", 0.864), ("S4", 0.502), ("S5", 0.502), ("S4", 0.40))
CAP = 40000.0


def file_values() -> dict[str, np.ndarray]:
    rows = list(ROWS)
    out = {f: pd.read_csv(SUBMISSIONS / f"{f}.csv")[TARGET].to_numpy()[rows]
           for f in ("fam2_bold", "fam2_mid", "fam2_gen", "fam_bet")}
    cal = pd.read_csv(REPORTS / "r4_audit" / "calibration_final.csv").set_index("row")
    out["audit_M"] = cal.loc[rows, "recommended_prediction"].to_numpy(float)
    return out


def draw(cal: pd.DataFrame, spec: str, struct: str, n: int, rng) -> np.ndarray:
    cols = []
    for r in ROWS:
        q = cal[(cal.row == r) & (cal.spec == spec) & (cal.structure == struct)].iloc[0]
        lo, md, hi = [np.log10(1 + max(float(q[k]), 0.0)) for k in ("SI_q10", "SI_q50", "SI_q90")]
        sd = max((hi - lo) / (2 * 1.2816), 0.15)
        cols.append(np.minimum(10 ** (md + sd * rng.standard_normal(n)) - 1, CAP))
    return np.column_stack(cols)


def main(n: int = 200_000, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    cal = pd.read_csv(REPORTS / "r4_audit" / "calibration.csv")
    files = file_values()
    names = list(files)
    P = np.array([files[k] for k in names], float)
    lines = ["# Выбор финальных файлов (R4)", "",
             "Ожидаемая ошибка на ключевых строках, тыс. единиц MSE (меньше — лучше). E — один файл, "
             "P(лучший) — доля сценариев, где файл лучше остальных, пары — E[min] при правиле «лучший из двух».", "",
             "Значения на ключевых строках (8b, 8c, 7e, 7l):", ""]
    lines += [f"* `{k}`: " + ", ".join(f"{ROWS[r]}={v:,.0f}" for r, v in zip(ROWS, files[k])) for k in names]
    for spec, pfull in SPECS:
        nf = int(n * pfull)
        Y = np.vstack([draw(cal, spec, "full", nf, rng), draw(cal, spec, "skel", n - nf, rng)])
        K = np.stack([((Y - P[i]) ** 2).sum(1) / 597.0 for i in range(len(names))], 1)
        best = np.bincount(K.argmin(1), minlength=len(names)) / len(K)
        lines += ["", f"## Спецификация {spec}, P(full)={pfull}",
                  f"P(8b>10k)={np.mean(Y[:, 0] > 1e4):.2f}, P(8c>10k)={np.mean(Y[:, 1] > 1e4):.2f}", "",
                  "| файл | E, тыс. | P(лучший) |", "|---|---|---|"]
        lines += [f"| {k} | {K[:, i].mean() / 1e3:.0f} | {best[i]:.2f} |" for i, k in enumerate(names)]
        pairs = sorted((K[:, [i, j]].min(1).mean(), names[i], names[j]) for i, j in itertools.combinations(range(len(names)), 2))
        lines += ["", "Лучшие пары: " + "; ".join(f"{a}+{b} = {v / 1e3:.0f}" for v, a, b in pairs[:3])]
    (REPORTS / "r4_pair_choice.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
