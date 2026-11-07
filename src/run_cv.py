"""Кросс-валидация всех кандидатов и подбор весов смеси.

Запуск:  python -m src.run_cv
Результат: reports/cv_results.md, reports/oof.npz
"""
from __future__ import annotations

import json
import time

import numpy as np

from .config import N_REPEATS, N_SPLITS, REPORTS, SEED, TARGET
from .cv import (
    bootstrap_mse_ci,
    iter_splits,
    rmse_log,
    scaffold_groups,
    score_all,
    spearman,
)
from .data import load_train
from .features import FeatureBuilder
from .models import (
    ConditionalMeanModel,
    IsotonicCalibratedModel,
    LogSmearingModel,
    MeanBaseline,
    TailAwareForest,
    fit_blend_weights,
)


def build_candidates():
    return {
        "константа": MeanBaseline(),
        "лог+smearing": LogSmearingModel(),
        "условное среднее": ConditionalMeanModel(),
        "лог + изотоника": IsotonicCalibratedModel(),
        "лес на SI": TailAwareForest(),
    }


def main() -> None:
    t0 = time.time()
    REPORTS.mkdir(parents=True, exist_ok=True)

    df = load_train()
    y = df[TARGET].to_numpy(dtype=float)
    groups = scaffold_groups(df["smiles_std"])

    fb = FeatureBuilder()
    X, names = fb.fit_transform(df)

    print(f"трейн: {X.shape[0]} молекул, {X.shape[1]} признаков, "
          f"{len(set(groups))} скаффолдов")
    print(f"нулей в таргете: {(y == 0).sum()} ({100 * (y == 0).mean():.1f}%), "
          f"максимум SI: {y.max():,.0f}\n")

    cand = build_candidates()
    # OOF-предсказания: отдельная матрица на каждый повтор CV.
    oof = {k: np.zeros((N_REPEATS, len(y))) for k in cand}

    for rep, fold, tr, va in iter_splits(y, groups, N_SPLITS, N_REPEATS, SEED):
        for key, proto in cand.items():
            model = proto.__class__(**_params(proto))
            model.fit(X[tr], y[tr])
            oof[key][rep, va] = model.predict(X[va])
        print(f"  повтор {rep + 1}/{N_REPEATS}, фолд {fold + 1}/{N_SPLITS} готов",
              end="\r")

    print("\n")
    rows = []
    for key, mat in oof.items():
        per_rep = [np.mean((y - mat[r]) ** 2) for r in range(N_REPEATS)]
        lo, hi = bootstrap_mse_ci(y, mat.mean(axis=0))
        rows.append({
            "модель": key,
            "MSE": float(np.mean(per_rep)),
            "MSE_se": float(np.std(per_rep) / np.sqrt(N_REPEATS)),
            "MSE_ci": (lo, hi),
            "RMSE_log": rmse_log(y, mat.mean(axis=0)),
            "Spearman": spearman(y, mat.mean(axis=0)),
        })

    pooled = {k: v.mean(axis=0) for k, v in oof.items() if k != "константа"}
    weights = fit_blend_weights(pooled, y)
    blend_pred = sum(w * pooled[k] for k, w in weights.items())
    lo, hi = bootstrap_mse_ci(y, blend_pred)
    rows.append({
        "модель": "смесь",
        "MSE": float(np.mean((y - blend_pred) ** 2)),
        "MSE_se": float("nan"),
        "MSE_ci": (lo, hi),
        "RMSE_log": rmse_log(y, blend_pred),
        "Spearman": spearman(y, blend_pred),
    })

    rows.sort(key=lambda r: r["RMSE_log"])
    _print_table(rows)
    print("\nвеса смеси:", json.dumps(weights, ensure_ascii=False))
    print(f"\nготово за {time.time() - t0:.0f} c")

    np.savez(REPORTS / "oof.npz", y=y, **{k: v.mean(axis=0) for k, v in oof.items()})
    (REPORTS / "blend_weights.json").write_text(
        json.dumps(weights, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_report(rows, weights, X.shape, len(set(groups)))


def _params(proto) -> dict:
    """Повторная сборка модели того же класса с теми же гиперпараметрами."""
    keep = ("seed", "tail_scale", "n_estimators", "smearing", "inner_splits")
    return {k: v for k, v in vars(proto).items() if k in keep}


def _print_table(rows) -> None:
    head = f"{'модель':<22}{'MSE':>14}{'±':>12}{'RMSE log':>11}{'Spearman':>10}"
    print(head)
    print("-" * len(head))
    for r in rows:
        se = "" if r["MSE_se"] != r["MSE_se"] else f"{r['MSE_se']:,.0f}"
        print(f"{r['модель']:<22}{r['MSE']:>14,.0f}{se:>12}"
              f"{r['RMSE_log']:>11.3f}{r['Spearman']:>10.3f}")
    print("\n95% интервал MSE (бутстрэп):")
    for r in rows:
        lo, hi = r["MSE_ci"]
        print(f"  {r['модель']:<22}[{lo:>12,.0f} … {hi:>14,.0f}]")


def _write_report(rows, weights, shape, n_scaf) -> None:
    lines = [
        "# Результаты кросс-валидации", "",
        f"Трейн: {shape[0]} молекул, {shape[1]} признаков, {n_scaf} скаффолдов.",
        f"Схема: StratifiedGroupKFold по скаффолдам, {N_SPLITS} фолдов, "
        f"{N_REPEATS} повтора, сид {SEED}.", "",
        "| модель | MSE | RMSE log | Spearman | 95% интервал MSE |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lo, hi = r["MSE_ci"]
        lines.append(
            f"| {r['модель']} | {r['MSE']:,.0f} | {r['RMSE_log']:.3f} | "
            f"{r['Spearman']:.3f} | {lo:,.0f} … {hi:,.0f} |"
        )
    lines += [
        "", "## Веса смеси", "",
        "```json", json.dumps(weights, ensure_ascii=False, indent=2), "```", "",
        "## Как читать таблицу", "",
        "MSE на исходном SI определяется несколькими экстремальными молекулами, "
        "поэтому доверительный интервал очень широк и сравнивать модели по этому "
        "столбцу почти бессмысленно. Для выбора модели смотрите на RMSE log и "
        "Spearman: там сигнал есть. MSE нужна только для финального решения между "
        "двумя-тремя кандидатами.",
    ]
    (REPORTS / "cv_results.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
