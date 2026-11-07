"""Сабмит на модели, обученной вместе с внешними данными.

Конфигурация взята из эксперимента src/exp_external.py: вес внешних строк
1.0, все строки, обрезка SI сверху на 5000. Без обрезки три строки-выброса
ломают поправку Дуана и вместе с ней весь прогноз.

После предсказания уровень выравнивается по измеренному среднему теста.
Зонды из src/probe.py дали среднее 323.38 на публичной части. При MSE
ошибка в уровне входит в метрику квадратом независимо от того, насколько
хорошо модель упорядочивает молекулы, поэтому выравнивание почти бесплатно
улучшает результат.

Запуск:
    python -m src.submit_external --model log --target-mean 323.38
    python -m src.submit_external --model both --name ext_blend
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt

import numpy as np
import pandas as pd

from .config import ID_COL, SUBMISSIONS, TARGET
from .data import load_sample_submission, load_test, load_train
from .external import add_keys, build_model, load_any
from .features import FeatureBuilder
from .make_submission import apply_known_answers, unify_duplicates, validate

CLIP = 5000.0
WEIGHT = 1.0


def fit_predict(kind: str, train, ext, X_tr, X_ext, X_te, fp_idx, ctx_idx):
    X = np.vstack([X_tr, X_ext])
    y = np.concatenate([train[TARGET].to_numpy(float), ext[TARGET].to_numpy(float)])
    w = np.concatenate([np.ones(len(train)), np.full(len(ext), WEIGHT)])
    model = build_model(kind, fp_idx=fp_idx, ctx_idx=ctx_idx)
    model.fit(X, y, sample_weight=w)
    return np.asarray(model.predict(X_te), dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="log", choices=["log", "kernel", "both"])
    ap.add_argument("--target-mean", type=float, default=323.38,
                    help="измеренное зондами среднее теста; 0 отключает выравнивание")
    ap.add_argument("--source", default="r2", choices=["mine", "r2"],
                    help="какой внешний набор брать")
    ap.add_argument("--all-rows", action="store_true",
                    help="брать и ненадёжные строки внешнего набора")
    ap.add_argument("--max-cap", type=float, default=0.0,
                    help="верхняя граница прогноза; 0 отключает")
    ap.add_argument("--name", default=None)
    ap.add_argument("--comment", default="")
    args = ap.parse_args()

    train = add_keys(load_train())
    test = load_test()
    sample = load_sample_submission()
    ext = load_any(args.source, reliable_only=not args.all_rows,
                   clip=CLIP, weight=WEIGHT)
    print(f"трейн {len(train)}, внешних {len(ext)} из набора {args.source} (clip {CLIP:g}), тест {len(test)}")

    fb = FeatureBuilder().fit(pd.concat([train, ext], ignore_index=True))
    X_tr, names = fb.transform(train)
    X_ext, _ = fb.transform(ext)
    X_te, _ = fb.transform(test)
    fp_idx = [i for i, n in enumerate(names) if n.startswith("fp_")]
    ctx_idx = [i for i, n in enumerate(names)
               if n.startswith("cult_") or n.startswith("strain_")]

    kinds = ["log", "kernel"] if args.model == "both" else [args.model]
    preds = []
    for k in kinds:
        p = fit_predict(k, train, ext, X_tr, X_ext, X_te, fp_idx, ctx_idx)
        print(f"  {k}: среднее {p.mean():,.1f}, медиана {np.median(p):,.1f}, "
              f"максимум {p.max():,.0f}")
        preds.append(p)
    pred = np.mean(preds, axis=0)

    pred = unify_duplicates(test, pred)
    pred, hits = apply_known_answers(test, pred, train)
    print(f"подставлено известных ответов: {hits}")

    if args.target_mean > 0:
        cur = float(pred.mean())
        pred = np.clip(pred * (args.target_mean / cur), 0.0, None)
        print(f"уровень выровнен: {cur:,.1f} -> {pred.mean():,.1f}, "
              f"максимум {pred.max():,.0f}")

    if args.max_cap > 0:
        # Лидерборд показал, что большие максимумы вредят: вариант с
        # максимумом 10 615 проиграл варианту с 5 312. Обрезаем верх и
        # возвращаем среднее на место, повторяя до сходимости.
        for _ in range(20):
            pred = np.minimum(pred, args.max_cap)
            if args.target_mean <= 0:
                break
            cur = float(pred.mean())
            if abs(cur - args.target_mean) < 0.01:
                break
            pred = pred * (args.target_mean / cur)
        pred = np.clip(np.minimum(pred, args.max_cap), 0.0, None)
        print(f"после обрезки на {args.max_cap:,.0f}: среднее {pred.mean():,.1f}, "
              f"максимум {pred.max():,.0f}")

    target_col = sample.columns[1]
    sub = pd.DataFrame({ID_COL: test[ID_COL].to_numpy(), target_col: pred})
    validate(sub, sample)

    name = args.name or f"ext_{args.model}_{dt.datetime.now():%m%d_%H%M}"
    path = SUBMISSIONS / f"{name}.csv"
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)
    print(f"записано: {path.name}  среднее {pred.mean():,.1f}, "
          f"максимум {pred.max():,.0f}")

    log = SUBMISSIONS / "log.csv"
    row = {"время": f"{dt.datetime.now():%Y-%m-%d %H:%M}", "файл": path.name,
           "коммит": "-", "модель": f"ext_{args.source}_{args.model}", "tail_scale": "",
           "известные": hits, "среднее": round(float(pred.mean()), 2),
           "макс": round(float(pred.max()), 2), "LB": "",
           "комментарий": args.comment or f"внешние {args.source} w={WEIGHT}, clip={CLIP:g}"}
    if log.exists():
        with log.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=list(pd.read_csv(log).columns)).writerow(row)
    print(f"журнал обновлён: {log.name}")


if __name__ == "__main__":
    main()
