"""Эксперимент: помогают ли внешние данные на кросс-валидации соревнования.

Валидация всегда только на строках train_v2. Внешние строки попадают
исключительно в обучающую часть фолда, в валидационную — никогда.

Два протокола:
  A «как в тесте» — внешние строки оставлены все, включая молекулы, которые
     совпадают с валидационными. В тесте такое пересечение есть (около 105
     молекул из 597), поэтому оценка честная по отношению к реальности;
  B «чистая»     — из обучающей части удалены внешние строки, чей скелет
     InChIKey совпадает с валидационной молекулой. Показывает обобщение
     на действительно новые структуры.

Запуск: python -m src.exp_external
Результат: reports/r4_external.md
"""
from __future__ import annotations

import argparse
import itertools
import os
import time

import numpy as np
import pandas as pd

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")

from .config import REPORTS, SEED, TARGET  # noqa: E402
from .cv import iter_splits, rmse_log, scaffold_groups, spearman  # noqa: E402
from .data import load_train  # noqa: E402
from .external import add_keys, build_model, load_external  # noqa: E402
from .features import FeatureBuilder  # noqa: E402

N_SPLITS, N_REPEATS = 5, 2
WEIGHTS = [0.0, 0.1, 0.3, 1.0]
RELIABLE = [True, False]
CLIPS = [None, 5000.0]
MODELS = ["log", "kernel"]
REPORT = REPORTS / "r4_external.md"


def prepare():
    """Матрица признаков строится один раз на объединении train и external."""
    train = add_keys(load_train())
    variants = {}
    for rel, clip in itertools.product(RELIABLE, CLIPS):
        variants[(rel, clip)] = load_external(reliable_only=rel, clip=clip)

    widest = variants[(False, None)]
    allrows = pd.concat([train, widest], ignore_index=True)

    fb = FeatureBuilder().fit(allrows)
    X_tr, names = fb.transform(train)
    fp_idx = [i for i, n in enumerate(names) if n.startswith("fp_")]
    ctx_idx = [i for i, n in enumerate(names)
               if n.startswith("cult_") or n.startswith("strain_")]

    ext_X = {}
    for key, df in variants.items():
        ext_X[key] = fb.transform(df)[0]

    return train, X_tr, variants, ext_X, fp_idx, ctx_idx, names


def run_config(train, X_tr, ext_df, ext_X, fp_idx, ctx_idx,
               model_kind, weight, protocol):
    """OOF-предсказания для одной конфигурации."""
    y = train[TARGET].to_numpy(dtype=float)
    groups = scaffold_groups(train["smiles_std"])
    oof = np.zeros((N_REPEATS, len(y)))

    ext_y = ext_df[TARGET].to_numpy(dtype=float) if ext_df is not None else None
    ext_keys = ext_df["ik_block"].to_numpy() if ext_df is not None else None
    val_keys_all = train["ik_block"].to_numpy()

    for rep, _fold, tr, va in iter_splits(y, groups, N_SPLITS, N_REPEATS, SEED):
        Xs, ys, ws = [X_tr[tr]], [y[tr]], [np.ones(len(tr))]

        if weight > 0 and ext_df is not None:
            mask = np.ones(len(ext_df), dtype=bool)
            if protocol == "B":
                blocked = set(val_keys_all[va]) - {None}
                mask = ~np.array([k in blocked for k in ext_keys])
            if mask.any():
                Xs.append(ext_X[mask])
                ys.append(ext_y[mask])
                ws.append(np.full(int(mask.sum()), weight))

        model = build_model(model_kind, fp_idx=fp_idx, ctx_idx=ctx_idx)
        model.fit(np.vstack(Xs), np.concatenate(ys),
                  sample_weight=np.concatenate(ws))
        oof[rep, va] = model.predict(X_tr[va])

    return oof.mean(axis=0)


def paired_bootstrap(y, pred_a, pred_b, n_boot: int = 2000, seed: int = SEED):
    """Интервал для разности RMSE log между вариантом и базой.

    Отрицательные значения означают, что вариант лучше базы. Если ноль
    попадает внутрь интервала, разница неотличима от шума.
    """
    rng = np.random.default_rng(seed)
    la = np.log10(np.clip(pred_a, 0, None) + 1.0)
    lb = np.log10(np.clip(pred_b, 0, None) + 1.0)
    ly = np.log10(np.clip(y, 0, None) + 1.0)
    ea, eb = (la - ly) ** 2, (lb - ly) ** 2
    n = len(y)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = np.sqrt(ea[idx].mean()) - np.sqrt(eb[idx].mean())
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="только модель log")
    args = ap.parse_args()

    t0 = time.time()
    REPORTS.mkdir(parents=True, exist_ok=True)

    train, X_tr, variants, ext_X, fp_idx, ctx_idx, _names = prepare()
    y = train[TARGET].to_numpy(dtype=float)
    print(f"train: {len(train)} молекул, признаков {X_tr.shape[1]}")
    for (rel, clip), df in variants.items():
        print(f"  внешних строк (reliable={rel}, clip={clip}): {len(df)}")

    # Подмножества для раздельного разбора.
    ext_keys_all = set(variants[(False, None)]["ik_block"]) - {None}
    in_ext = train["ik_block"].isin(ext_keys_all).to_numpy()
    top10 = np.zeros(len(y), dtype=bool)
    top10[np.argsort(y)[-10:]] = True
    print(f"  валидационных молекул, есть во внешнем наборе: {in_ext.sum()}")

    models = ["log"] if args.quick else MODELS
    rows, base_pred = [], {}

    for model_kind in models:
        base = run_config(train, X_tr, None, None, fp_idx, ctx_idx,
                          model_kind, 0.0, "A")
        base_pred[model_kind] = base
        rows.append(dict(model=model_kind, protocol="—", w=0.0, reliable="—",
                         clip="—", spearman=spearman(y, base),
                         rmse=rmse_log(y, base), lo=0.0, hi=0.0,
                         rmse_in=rmse_log(y[in_ext], base[in_ext]),
                         rmse_out=rmse_log(y[~in_ext], base[~in_ext]),
                         rmse_top=rmse_log(y[top10], base[top10])))
        print(f"\n{model_kind}: база w=0 -> Spearman {rows[-1]['spearman']:.3f}, "
              f"RMSE log {rows[-1]['rmse']:.3f}")

        for protocol, w, rel, clip in itertools.product(
                ["A", "B"], [x for x in WEIGHTS if x > 0], RELIABLE, CLIPS):
            ext_df = variants[(rel, clip)]
            pred = run_config(train, X_tr, ext_df, ext_X[(rel, clip)],
                              fp_idx, ctx_idx, model_kind, w, protocol)
            lo, hi = paired_bootstrap(y, pred, base)
            rows.append(dict(
                model=model_kind, protocol=protocol, w=w,
                reliable="только надёжные" if rel else "все",
                clip="нет" if clip is None else f"{clip:g}",
                spearman=spearman(y, pred), rmse=rmse_log(y, pred),
                lo=lo, hi=hi,
                rmse_in=rmse_log(y[in_ext], pred[in_ext]),
                rmse_out=rmse_log(y[~in_ext], pred[~in_ext]),
                rmse_top=rmse_log(y[top10], pred[top10])))
            mark = "лучше" if hi < 0 else ("хуже" if lo > 0 else "в шуме")
            print(f"  {protocol} w={w:<4} {rows[-1]['reliable']:<16} "
                  f"clip={rows[-1]['clip']:<5} RMSE {rows[-1]['rmse']:.3f} "
                  f"({lo:+.3f}…{hi:+.3f}) {mark}")

    df = pd.DataFrame(rows)
    df.to_csv(REPORTS / "r4_external.csv", index=False)
    write_report(df, train, in_ext, variants)
    print(f"\nготово за {time.time() - t0:.0f} c, отчёт: {REPORT}")


def write_report(df: pd.DataFrame, train, in_ext, variants) -> None:
    L = ["# Внешние данные: помогают ли они на CV", "",
         f"Схема: {N_SPLITS} фолдов по скаффолдам, {N_REPEATS} повтора, сид {SEED}. "
         "Валидация только на строках train_v2, внешние строки всегда в обучающей части.",
         "",
         f"Трейн: {len(train)} молекул, из них {int(in_ext.sum())} встречаются "
         "во внешнем наборе.", "",
         "Размер внешнего набора по вариантам фильтрации:", ""]
    for (rel, clip), v in variants.items():
        L.append(f"* reliable={rel}, clip={clip}: {len(v)} строк")
    L += ["", "## Результаты", "",
          "Интервал — парный бутстрэп разности RMSE log с базой w=0. "
          "Отрицательные значения означают улучшение; если ноль внутри "
          "интервала, разница неотличима от шума.", "",
          "| модель | прот. | w | строки | clip | Spearman | RMSE log | разность с базой | вердикт |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in df.itertuples(index=False):
        if r.protocol == "—":
            verdict, interval = "база", "—"
        else:
            verdict = "лучше" if r.hi < 0 else ("хуже" if r.lo > 0 else "в шуме")
            interval = f"{r.lo:+.3f} … {r.hi:+.3f}"
        L.append(f"| {r.model} | {r.protocol} | {r.w:g} | {r.reliable} | {r.clip} "
                 f"| {r.spearman:.3f} | {r.rmse:.3f} | {interval} | {verdict} |")

    L += ["", "## Разбор по подмножествам", "",
          "`RMSE in` — молекулы валидации, которые есть во внешнем наборе; "
          "`RMSE out` — остальные; `RMSE top10` — десять самых селективных "
          "молекул трейна.", "",
          "| модель | прот. | w | строки | clip | RMSE in | RMSE out | RMSE top10 |",
          "|---|---|---|---|---|---|---|---|"]
    for r in df.itertuples(index=False):
        L.append(f"| {r.model} | {r.protocol} | {r.w:g} | {r.reliable} | {r.clip} "
                 f"| {r.rmse_in:.3f} | {r.rmse_out:.3f} | {r.rmse_top:.3f} |")

    REPORT.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
