"""Эксперименты R4: наборы признаков и ядерная модель на той же CV, что и run_cv.

Запуск: python -m src.exp_r4        Результат: reports/r4_experiments.md
Решения принимаются по RMSE log и Spearman (MSE на сыром SI шумит из-за одной молекулы).
"""
from __future__ import annotations

import numpy as np

from .config import N_REPEATS, N_SPLITS, REPORTS, SEED, TARGET
from .cv import iter_splits, rmse_log, scaffold_groups, spearman
from .data import load_train
from .features import FeatureBuilder
from .models import ConditionalMeanModel, KernelLogModel, LogSmearingModel

SETS = {"база": (), "+maccs": ("maccs",), "+antiviral": ("antiviral",), "+maccs+antiviral": ("maccs", "antiviral")}


def main() -> None:
    df = load_train()
    y = df[TARGET].to_numpy(dtype=float)
    groups = scaffold_groups(df["smiles_std"])
    rows = []
    for sname, extras in SETS.items():
        fb = FeatureBuilder(extras=extras)
        X, names = fb.fit_transform(df)
        fp_idx = [i for i, n in enumerate(names) if n.startswith("fp_")]
        ctx_idx = [i for i, n in enumerate(names) if n.startswith(("cult_", "strain_"))]
        makers = {
            "лог+smearing": lambda: LogSmearingModel(),
            "условное среднее": lambda: ConditionalMeanModel(n_estimators=300),
            "ядро+культура": lambda: KernelLogModel(fp_idx, ctx_idx),
        }
        for mname, mk in makers.items():
            if mname == "ядро+культура" and extras:
                continue  # ядро использует только fp и культуру, дополнительные блоки ему не нужны
            oof = np.zeros((N_REPEATS, len(y)))
            for rep, fold, tr, va in iter_splits(y, groups, N_SPLITS, N_REPEATS, SEED):
                oof[rep, va] = mk().fit(X[tr], y[tr]).predict(X[va])
            m = oof.mean(axis=0)
            rows.append((sname, mname, rmse_log(y, m), spearman(y, m), float(np.mean((y - m) ** 2))))
            print(f"{sname:18s}{mname:20s} RMSE log {rows[-1][2]:.3f}  Spearman {rows[-1][3]:.3f}", flush=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    lines = ["# Эксперименты R4", "", "| признаки | модель | RMSE log | Spearman | MSE (шумит) |", "|---|---|---|---|---|"]
    lines += [f"| {a} | {b} | {c:.3f} | {d:.3f} | {e:,.0f} |" for a, b, c, d, e in sorted(rows, key=lambda r: r[2])]
    (REPORTS / "r4_experiments.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
