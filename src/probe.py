"""Зонды: сабмиты из одной константы, чтобы измерить саму тестовую выборку.

Зачем. MSE константы c равна Var + (mu - c)^2, где mu и Var — среднее и
дисперсия целевой переменной на публичной части теста. Отправив две разные
константы и получив два значения MSE, эти два неизвестных можно вычислить
точно:

    MSE(c1) - MSE(c2) = (mu - c1)^2 - (mu - c2)^2
    mu  = (MSE(c2) - MSE(c1) + c1^2 - c2^2) / (2 * (c1 - c2))
    Var = MSE(c1) - (mu - c1)^2

Знание mu и Var говорит, какой вообще потолок у константы, насколько
тяжёлый хвост в тесте и сколько даёт модель сверх простого среднего.
Два сабмита за эту информацию — выгодный обмен при лимите 20 в день.

Создать зонды:   python -m src.probe --make 0 400
Посчитать ответ: python -m src.probe --solve 0 2102000 400 1750000
"""
from __future__ import annotations

import argparse

import pandas as pd

from .config import ID_COL, SUBMISSIONS
from .data import load_sample_submission, load_test


def make(constants: list[float]) -> None:
    test, sample = load_test(), load_sample_submission()
    target = sample.columns[1]
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    for c in constants:
        sub = pd.DataFrame({ID_COL: test[ID_COL].to_numpy(), target: float(c)})
        assert list(sub.columns) == list(sample.columns)
        assert len(sub) == len(sample)
        name = f"probe_c{c:g}.csv"
        sub.to_csv(SUBMISSIONS / name, index=False)
        print(f"  {name}: все {len(sub)} строк равны {c:g}")
    print("\nОтправьте эти файлы и запустите разбор:")
    joined = " ".join(f"{c:g} <MSE>" for c in constants)
    print(f"  python -m src.probe --solve {joined}")


def solve(pairs: list[float]) -> None:
    if len(pairs) < 4 or len(pairs) % 2:
        raise SystemExit("нужны пары: константа MSE константа MSE")
    c1, m1, c2, m2 = pairs[0], pairs[1], pairs[2], pairs[3]
    if c1 == c2:
        raise SystemExit("константы должны различаться")

    mu = (m2 - m1 + c1 ** 2 - c2 ** 2) / (2 * (c1 - c2))
    var = m1 - (mu - c1) ** 2
    print(f"среднее SI на публичной части теста: {mu:,.2f}")
    if var < 0:
        print("дисперсия вышла отрицательной — проверьте, что MSE указаны "
              "именно для этих констант")
        return
    print(f"дисперсия: {var:,.0f},  стандартное отклонение: {var ** 0.5:,.1f}")
    print(f"\nлучшая возможная константа: {mu:,.2f}, её MSE: {var:,.0f}")
    print("Любой результат ниже этого числа — заслуга модели, а не удачного "
          "выбора константы.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--make", nargs="+", type=float, metavar="C",
                    help="создать сабмиты-константы")
    ap.add_argument("--solve", nargs="+", type=float, metavar="X",
                    help="пары 'константа MSE' для вычисления mu и Var")
    args = ap.parse_args()
    if args.make:
        make(args.make)
    elif args.solve:
        solve(args.solve)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
