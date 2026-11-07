"""Сабмит со смесью «лог-модель + ядерная модель + соседи» (R4), не трогая make_submission.

Запуск: python -m src.make_nb_submission --name nb_logker [--scale 1.0] [--comment "..."]
CV на метрике, близкой к лидерборду (SI обрезан на 10000): MSE на отложенной половине -32% к константе,
у одной лог-модели около -2%. Журнал: строка в submissions/log.csv.
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
from .make_submission import git_commit, unify_duplicates, validate
from .models import KernelLogModel, LogSmearingModel
from .neighbors import NeighborBlend, fp_matrix, tanimoto


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="nb_logker")
    ap.add_argument("--scale", type=float, default=1.0, help="общий множитель прогноза (уровень хвоста)")
    ap.add_argument("--max-pred", type=float, default=12000.0,
                    help="потолок прогноза: одна копия гиганта в тесте иначе даёт ~29000, а ошибка там стоит миллионы MSE")
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
    fp_idx = [i for i, n in enumerate(names) if n.startswith("fp_")]
    ctx_idx = [i for i, n in enumerate(names) if n.startswith(("cult_", "strain_"))]

    base = (LogSmearingModel().fit(X_tr, y).predict(X_te)
            + KernelLogModel(fp_idx, ctx_idx).fit(X_tr, y).predict(X_te)) / 2

    ext = pd.read_csv(args.ext)
    ext["smiles_std"] = ext["smiles"].map(standardize_smiles)
    ext = ext.dropna(subset=["smiles_std"]).reset_index(drop=True)
    nb = NeighborBlend().fit(train, y, ext)
    pred = nb.predict(test, base)

    q = fp_matrix(test["smiles_std"])
    known = int((np.maximum(tanimoto(q, fp_matrix(train["smiles_std"])).max(1),
                            tanimoto(q, fp_matrix(ext["smiles_std"])).max(1)) >= 0.99).sum())
    pred = np.clip(unify_duplicates(test, pred * args.scale), 0.0, args.max_pred)

    sub = pd.DataFrame({ID_COL: test[ID_COL].to_numpy(), sample.columns[1]: pred})
    validate(sub, sample)
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    path = SUBMISSIONS / f"{args.name}.csv"
    sub.to_csv(path, index=False)
    print(f"записано {path}: среднее {pred.mean():,.2f}, макс {pred.max():,.2f}, нулей {(pred == 0).sum()}, "
          f"копий в train/внешнем наборе: {known}")

    log = SUBMISSIONS / "log.csv"
    new = not log.exists()
    with log.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["время", "файл", "коммит", "модель", "tail_scale", "известные", "среднее", "макс", "LB", "комментарий"])
        w.writerow([f"{dt.datetime.now():%Y-%m-%d %H:%M}", path.name, git_commit(), "nb_log+kernel", "",
                    known, round(float(pred.mean()), 2), round(float(pred.max()), 2), "",
                    args.comment or f"смесь соседей, множитель {args.scale}, потолок {args.max_pred:g}"])


if __name__ == "__main__":
    main()
