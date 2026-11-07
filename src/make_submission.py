"""Обучение на всём трейне и сборка файла для Kaggle.

Запуск:
    python -m src.make_submission --model blend --name blend_v1
    python -m src.make_submission --model condmean --tail-scale 1.6 --name aggressive

Что делает сверх предсказания модели:
  * подставляет известные ответы для молекул, которые есть и в трейне, и в тесте;
  * выдаёт одинаковый прогноз одинаковым молекулам внутри теста;
  * проверяет формат файла по sample_submit.csv;
  * дописывает строку в submissions/log.csv.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import subprocess

import numpy as np
import pandas as pd

from .config import ID_COL, REPORTS, SUBMISSIONS, TARGET
from .data import load_sample_submission, load_test, load_train
from .features import FeatureBuilder
from .models import (
    Blend,
    ConditionalMeanModel,
    IsotonicCalibratedModel,
    LogSmearingModel,
    MeanBaseline,
    TailAwareForest,
)


def build_model(kind: str, tail_scale: float):
    if kind == "mean":
        return MeanBaseline()
    if kind == "log":
        return LogSmearingModel()
    if kind == "condmean":
        return ConditionalMeanModel(tail_scale=tail_scale)
    if kind == "isotonic":
        return IsotonicCalibratedModel()
    if kind == "forest":
        return TailAwareForest()
    if kind == "blend":
        path = REPORTS / "blend_weights.json"
        if not path.exists():
            raise SystemExit("нет reports/blend_weights.json — сначала python -m src.run_cv")
        w = json.loads(path.read_text(encoding="utf-8"))
        registry = {
            "лог+smearing": LogSmearingModel(),
            "условное среднее": ConditionalMeanModel(tail_scale=tail_scale),
            "лог + изотоника": IsotonicCalibratedModel(),
            "лес на SI": TailAwareForest(),
        }
        models, weights = [], []
        for key, weight in w.items():
            if weight > 0 and key in registry:
                models.append(registry[key])
                weights.append(weight)
        total = sum(weights)
        return Blend(models, [x / total for x in weights], name="смесь")
    raise SystemExit(f"неизвестная модель: {kind}")


def apply_known_answers(test: pd.DataFrame, pred: np.ndarray,
                        train: pd.DataFrame) -> tuple[np.ndarray, int]:
    """Для молекул, присутствующих в трейне, берём известный SI."""
    pred = pred.copy()
    exact = (train.groupby(["smiles_std", "Culture"])[TARGET].mean().to_dict())
    loose = train.groupby("smiles_std")[TARGET].mean().to_dict()
    hits = 0
    for i, row in enumerate(test.itertuples(index=False)):
        key = (row.smiles_std, row.Culture)
        if key in exact:
            pred[i] = exact[key]
            hits += 1
        elif row.smiles_std in loose:
            pred[i] = loose[row.smiles_std]
            hits += 1
    return pred, hits


def unify_duplicates(test: pd.DataFrame, pred: np.ndarray) -> np.ndarray:
    """Одинаковым молекулам в тесте — одинаковый прогноз."""
    s = pd.Series(pred, index=test.index)
    return s.groupby(test["smiles_std"].to_numpy()).transform("mean").to_numpy()


def validate(sub: pd.DataFrame, sample: pd.DataFrame) -> None:
    assert list(sub.columns) == list(sample.columns), (
        f"колонки {list(sub.columns)} != {list(sample.columns)}")
    assert len(sub) == len(sample), f"строк {len(sub)} != {len(sample)}"
    assert sub[ID_COL].tolist() == sample[ID_COL].tolist(), "индексы не совпадают"
    vals = sub[sample.columns[1]].to_numpy(dtype=float)
    assert np.isfinite(vals).all(), "в прогнозе есть NaN или бесконечность"
    assert (vals >= 0).all(), "в прогнозе есть отрицательные значения"


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "нет-коммита"
    except Exception:
        return "нет-коммита"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="blend",
                    choices=["mean", "log", "condmean", "isotonic", "forest", "blend"])
    ap.add_argument("--tail-scale", type=float, default=1.0,
                    help="множитель верхнего бина: >1 агрессивнее на хвосте")
    ap.add_argument("--name", default=None)
    ap.add_argument("--no-known", action="store_true",
                    help="не подставлять ответы для известных молекул")
    ap.add_argument("--comment", default="")
    args = ap.parse_args()

    SUBMISSIONS.mkdir(parents=True, exist_ok=True)

    train, test, sample = load_train(), load_test(), load_sample_submission()
    fb = FeatureBuilder()
    X_tr, _ = fb.fit_transform(train)
    X_te, _ = fb.transform(test)
    y = train[TARGET].to_numpy(dtype=float)

    model = build_model(args.model, args.tail_scale)
    model.fit(X_tr, y)
    pred = np.asarray(model.predict(X_te), dtype=float)
    print(f"модель: {getattr(model, 'name', args.model)}")
    print(f"сырой прогноз: мин {pred.min():,.2f}, медиана {np.median(pred):,.2f}, "
          f"среднее {pred.mean():,.2f}, макс {pred.max():,.2f}")

    pred = unify_duplicates(test, pred)
    hits = 0
    if not args.no_known:
        pred, hits = apply_known_answers(test, pred, train)
        print(f"подставлено известных ответов: {hits}")

    pred = np.clip(pred, 0.0, None)
    target_col = sample.columns[1]
    sub = pd.DataFrame({ID_COL: test[ID_COL].to_numpy(), target_col: pred})
    validate(sub, sample)

    name = args.name or f"{args.model}_{dt.datetime.now():%m%d_%H%M}"
    path = SUBMISSIONS / f"{name}.csv"
    sub.to_csv(path, index=False)
    print(f"записано: {path}")
    print(f"итог: среднее {pred.mean():,.2f}, макс {pred.max():,.2f}, "
          f"нулей {(pred == 0).sum()}")

    log = SUBMISSIONS / "log.csv"
    new = not log.exists()
    with log.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["время", "файл", "коммит", "модель", "tail_scale",
                        "известные", "среднее", "макс", "LB", "комментарий"])
        w.writerow([f"{dt.datetime.now():%Y-%m-%d %H:%M}", path.name, git_commit(),
                    args.model, args.tail_scale, hits, round(float(pred.mean()), 2),
                    round(float(pred.max()), 2), "", args.comment])
    print(f"журнал обновлён: {log}")


if __name__ == "__main__":
    main()
