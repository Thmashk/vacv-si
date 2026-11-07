"""CV смеси «модель + соседи». Протоколы: A — внешние строки как есть (как в тесте),
B — из внешних убраны молекулы валидационного фолда. Запуск: python -m src.exp_neighbors"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .config import REPORTS, SEED, TARGET
from .cv import iter_splits, rmse_log, scaffold_groups, spearman
from .data import load_test, load_train, standardize_smiles
from .features import FeatureBuilder
from .models import LogSmearingModel
from .neighbors import NeighborBlend, fp_matrix, tanimoto


def main() -> None:
    tr = load_train()
    y = tr[TARGET].to_numpy(float)
    groups = scaffold_groups(tr["smiles_std"])
    import os
    path = "data/processed/vacv_external_usable.csv" if os.path.exists("data/processed/vacv_external_usable.csv") else "data/processed/vacv_external_dataset.csv"
    print("внешний набор:", path)
    ex = pd.read_csv(path)
    ex["smiles_std"] = ex["smiles"].map(standardize_smiles)
    ex = ex.dropna(subset=["smiles_std"]).reset_index(drop=True)
    X, _ = FeatureBuilder().fit_transform(tr)
    tfp, efp = fp_matrix(tr["smiles_std"]), fp_matrix(ex["smiles_std"])
    ex_vs_tr = tanimoto(efp, tfp)
    te = load_test()
    tf = fp_matrix(te["smiles_std"])
    nn = np.maximum(tanimoto(tf, tfp).max(1), tanimoto(tf, efp).max(1))
    print(f"покрытие теста (train + внешний): копия >=0.99: {(nn >= 0.99).sum()} из {len(te)}, >=0.7: {(nn >= 0.7).sum()}, >=0.5: {(nn >= 0.5).sum()}")
    grid = [dict(p=20, lam=lam, w_ext=we, culture_factor=0.3, exact_thr=et)
            for lam, we, et in itertools.product((0.3, 1.0), (0.5, 1.0), (None, 0.99))]
    big = int(np.argmax(y))
    store = {("база", "-"): np.zeros((2, len(y)))}
    for key in itertools.product(("A", "B"), range(len(grid))):
        store[key] = np.zeros((2, len(y)))
    for rep, fold, a, b in iter_splits(y, groups, 5, 2, SEED):
        base = LogSmearingModel().fit(X[a], y[a]).predict(X[b])
        store[("база", "-")][rep, b] = base
        keep_B = ex_vs_tr[:, b].max(1) < 0.99
        for proto in ("A", "B"):
            exm = ex if proto == "A" else ex[keep_B]
            efm = efp if proto == "A" else efp[keep_B]
            for gi, g in enumerate(grid):
                nb = NeighborBlend(**g).fit(tr.iloc[a], y[a], exm, ext_fp=efm, train_fp=tfp[a])
                store[(proto, gi)][rep, b] = nb.predict(tr.iloc[b], base, query_fp=tfp[b])
    def score(m):
        p = m.mean(0)
        ok = np.arange(len(y)) != big
        return rmse_log(y, p), spearman(y, p), float(np.mean((y - p) ** 2)), float(np.mean((y[ok] - p[ok]) ** 2))
    b0 = score(store[("база", "-")])
    rows = [("база", "-", *b0)]
    for (proto, gi), m in store.items():
        if proto in ("A", "B"):
            rows.append((proto, str(grid[gi]), *score(m)))
    rows.sort(key=lambda r: r[2])
    lines = ["# Смесь «модель + соседи» (R4)", "", "Протокол A — внешние как в тесте, B — без молекул валидации. Решать по RMSE log и Spearman.", "",
             "| протокол | параметры | RMSE log | Spearman | MSE | MSE без гиганта |", "|---|---|---|---|---|---|"]
    lines += [f"| {r[0]} | {r[1]} | {r[2]:.3f} | {r[3]:.3f} | {r[4]:,.0f} | {r[5]:,.0f} |" for r in rows[:14]]
    lines += ["", "База: " + f"RMSE log {b0[0]:.3f}, Spearman {b0[1]:.3f}, MSE {b0[2]:,.0f}, без гиганта {b0[3]:,.0f}"]
    for proto in ("A", "B"):
        best = min((r for r in rows if r[0] == proto), key=lambda r: r[2])
        lines.append(f"Лучший {proto}: {best[1]} -> RMSE log {best[2]:.3f}, Spearman {best[3]:.3f}")
    (REPORTS / "r4_neighbors.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
