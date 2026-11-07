"""Сравнение двух внешних наборов на одной схеме валидации.

Набор из PR #1 (мой) и набор R2 различаются фильтрацией. В первом 301
строка, где IC50 в точности равен CC50: это потолок концентрации,
записанный дважды, и их отношение механически даёт SI = 1. Во втором таких
строк нет. Проверяем, что это значит для качества модели.

Сетка узкая: вес 1.0 и обрезка 5000 уже выбраны в src/exp_external.py.

Запуск: python -m src.exp_external_r2
Результат: reports/r4_external_r2.md
"""
from __future__ import annotations

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
from .exp_external import paired_bootstrap  # noqa: E402
from .external import add_keys, build_model, load_any  # noqa: E402
from .features import FeatureBuilder  # noqa: E402

N_SPLITS, N_REPEATS = 5, 2
CLIP = 5000.0
WEIGHT = 1.0
REPORT = REPORTS / "r4_external_r2.md"


def main() -> None:
    t0 = time.time()
    REPORTS.mkdir(parents=True, exist_ok=True)

    train = add_keys(load_train())
    y = train[TARGET].to_numpy(dtype=float)
    groups = scaffold_groups(train["smiles_std"])

    sets = {}
    for which in ("mine", "r2"):
        for rel in (True, False):
            sets[(which, rel)] = load_any(which, reliable_only=rel, clip=CLIP)

    allrows = pd.concat([train] + list(sets.values()), ignore_index=True)
    fb = FeatureBuilder().fit(allrows)
    X_tr, names = fb.transform(train)
    fp_idx = [i for i, n in enumerate(names) if n.startswith("fp_")]
    ctx_idx = [i for i, n in enumerate(names)
               if n.startswith("cult_") or n.startswith("strain_")]
    X_ext = {k: fb.transform(v)[0] for k, v in sets.items()}

    print(f"трейн {len(train)}, признаков {X_tr.shape[1]}")
    for k, v in sets.items():
        print(f"  {k}: {len(v)} строк")

    ext_keys_all = set(sets[("r2", False)]["ik_block"]) | set(sets[("mine", False)]["ik_block"])
    in_ext = train["ik_block"].isin(ext_keys_all - {None}).to_numpy()
    top10 = np.zeros(len(y), dtype=bool)
    top10[np.argsort(y)[-10:]] = True

    rows, base_pred = [], {}
    for model_kind in ("log", "kernel"):
        configs = [("база", None, None, "—")]
        for which in ("mine", "r2"):
            for rel in (True, False):
                for protocol in ("A", "B"):
                    configs.append((which, rel, (which, rel), protocol))

        for label, rel, key, protocol in configs:
            oof = np.zeros((N_REPEATS, len(y)))
            for rep, _f, tr, va in iter_splits(y, groups, N_SPLITS, N_REPEATS, SEED):
                Xs, ys, ws = [X_tr[tr]], [y[tr]], [np.ones(len(tr))]
                if key is not None:
                    ext = sets[key]
                    mask = np.ones(len(ext), dtype=bool)
                    if protocol == "B":
                        blocked = set(train["ik_block"].to_numpy()[va]) - {None}
                        mask = ~np.array([k in blocked for k in ext["ik_block"]])
                    if mask.any():
                        Xs.append(X_ext[key][mask])
                        ys.append(ext[TARGET].to_numpy(float)[mask])
                        ws.append(np.full(int(mask.sum()), WEIGHT))
                m = build_model(model_kind, fp_idx=fp_idx, ctx_idx=ctx_idx)
                m.fit(np.vstack(Xs), np.concatenate(ys),
                      sample_weight=np.concatenate(ws))
                oof[rep, va] = m.predict(X_tr[va])
            pred = oof.mean(axis=0)

            if key is None:
                base_pred[model_kind] = pred
                lo = hi = 0.0
            else:
                lo, hi = paired_bootstrap(y, pred, base_pred[model_kind])

            rows.append(dict(
                model=model_kind,
                dataset={"mine": "PR #1", "r2": "R2", "база": "—"}[label],
                reliable="—" if rel is None else ("надёжные" if rel else "все"),
                protocol=protocol,
                spearman=spearman(y, pred), rmse=rmse_log(y, pred), lo=lo, hi=hi,
                rmse_in=rmse_log(y[in_ext], pred[in_ext]),
                rmse_out=rmse_log(y[~in_ext], pred[~in_ext]),
                rmse_top=rmse_log(y[top10], pred[top10]))),
            r = rows[-1]
            mark = "—" if key is None else (
                "лучше" if hi < 0 else ("хуже" if lo > 0 else "в шуме"))
            print(f"  {model_kind:<7}{r['dataset']:<6}{r['reliable']:<10}"
                  f"{protocol:<3}Spearman {r['spearman']:.3f}  "
                  f"RMSE {r['rmse']:.3f}  {mark}")

    df = pd.DataFrame(rows)
    df.to_csv(REPORTS / "r4_external_r2.csv", index=False)
    write(df)
    print(f"\nготово за {time.time() - t0:.0f} c, отчёт: {REPORT}")


def write(df: pd.DataFrame) -> None:
    L = ["# Какой внешний набор лучше", "",
         f"Схема: {N_SPLITS} фолдов по скаффолдам, {N_REPEATS} повтора, сид {SEED}. "
         f"Вес внешних строк {WEIGHT}, обрезка SI на {CLIP:g}.", "",
         "Протокол A оставляет внешние строки целиком, включая молекулы из "
         "валидационного фолда; B их удаляет.", "",
         "| модель | набор | строки | прот. | Spearman | RMSE log | разность с базой | вердикт |",
         "|---|---|---|---|---|---|---|---|"]
    for r in df.itertuples(index=False):
        if r.dataset == "—":
            verdict, interval = "база", "—"
        else:
            verdict = "лучше" if r.hi < 0 else ("хуже" if r.lo > 0 else "в шуме")
            interval = f"{r.lo:+.3f} … {r.hi:+.3f}"
        L.append(f"| {r.model} | {r.dataset} | {r.reliable} | {r.protocol} | "
                 f"{r.spearman:.3f} | {r.rmse:.3f} | {interval} | {verdict} |")
    L += ["", "| модель | набор | строки | прот. | RMSE in | RMSE out | RMSE top10 |",
          "|---|---|---|---|---|---|---|"]
    for r in df.itertuples(index=False):
        L.append(f"| {r.model} | {r.dataset} | {r.reliable} | {r.protocol} | "
                 f"{r.rmse_in:.3f} | {r.rmse_out:.3f} | {r.rmse_top:.3f} |")
    REPORT.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
