"""Сабмит итоговой модели R4 (v2, см. src/final_model.py). Не меняет make_submission.

Запуск:  python -m src.make_v2_submission --name r4_v2 --twin-cap 18000
         python -m src.make_v2_submission --name r4_v2_diag --twin-cap 5000   # диагностический: чистый публичный замер

twin-cap — потолок прогноза для строк с копией известной молекулы. В тесте есть строка-копия гиганта трейна (SI=36067, Vero).
Из публичных фактов (MSE константы 1.9M на 179 строках) следует, что в публичной части нет значений выше ~18000, поэтому эта
строка либо в приватной части, либо её значение не воспроизводится. Диагностический файл с малым потолком не даёт этой одной
строке исказить публичный замер остальной модели; итоговый файл ставит значение, близкое к условному среднему.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os

import numpy as np
import pandas as pd

from .config import ID_COL, SUBMISSIONS, TARGET
from .data import load_sample_submission, load_test, load_train, standardize_smiles
from .features import FeatureBuilder
from .final_model import FinalModel
from .make_submission import git_commit, unify_duplicates, validate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="r4_v2")
    ap.add_argument("--twin-cap", type=float, default=18000.0)
    ap.add_argument("--ext", default="data/processed/vacv_external_usable.csv")
    ap.add_argument("--comment", default="")
    args = ap.parse_args()
    if not os.path.exists(args.ext):
        raise SystemExit(f"нет внешнего набора {args.ext}")

    train, test, sample = load_train(), load_test(), load_sample_submission()
    y = train[TARGET].to_numpy(float)
    fb = FeatureBuilder()
    X_tr, names = fb.fit_transform(train)
    X_te, _ = fb.transform(test)
    ctx_idx = [i for i, n in enumerate(names) if n.startswith(("cult_", "strain_"))]

    ext = pd.read_csv(args.ext)
    ext["smiles_std"] = ext["smiles"].map(standardize_smiles)
    ext = ext.dropna(subset=["smiles_std"]).reset_index(drop=True)

    model = FinalModel(ctx_idx, twin_cap=args.twin_cap).fit(train, X_tr, y, ext)
    pred = np.clip(unify_duplicates(test, model.predict(test, X_te)), 0.0, None)

    sub = pd.DataFrame({ID_COL: test[ID_COL].to_numpy(), sample.columns[1]: pred})
    validate(sub, sample)
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    path = SUBMISSIONS / f"{args.name}.csv"
    sub.to_csv(path, index=False)
    print(f"записано {path}: среднее {pred.mean():,.2f}, макс {pred.max():,.2f}, >1000: {(pred > 1000).sum()}, "
          f">5000: {(pred > 5000).sum()}, нулей {(pred == 0).sum()}")

    log = SUBMISSIONS / "log.csv"
    new = not log.exists()
    with log.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["время", "файл", "коммит", "модель", "tail_scale", "известные", "среднее", "макс", "LB", "комментарий"])
        w.writerow([f"{dt.datetime.now():%Y-%m-%d %H:%M}", path.name, git_commit(), "r4_v2", "", "",
                    round(float(pred.mean()), 2), round(float(pred.max()), 2), "",
                    args.comment or f"KRR+ET -> соседи -> двойники, потолок двойника {args.twin_cap:g}"])


if __name__ == "__main__":
    main()
