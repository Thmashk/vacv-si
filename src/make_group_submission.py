"""Поправка по группе культуры: сжатие прогнозов группы к её среднему в трейне (R4).

Берёт готовый файл, находит тестовые строки, у которых нормализованное имя
культуры начинается с --culture (например bsc40: 'BSC-40', 'BSC-40 cells', 'BSC40'),
считает среднее SI этой группы в трейне и тянет прогнозы группы к нему:
    новое = lam*база + (1-lam)*среднее_группы,
с потолком по максимуму группы в трейне (--cap auto) и, при --down-only,
только вниз. Строки, молекула которых точно есть в трейне (известные ответы),
получают "двойниковую" поправку (см. --twin-q). Правило применяется ко всем строкам группы одинаково.

Двойники. Если молекула тестовой строки точно есть в трейне в той же группе культур
(например 'BSC-40' и 'BSC-40 cells'), берём её SI из трейна (T). Среди 8 групп дублей
в трейне SI совпал точно в 5 (62%), поэтому прогноз = q*T + (1-q)*прогноз_после_сжатия, q=0.6.

    python -m src.make_group_submission --base submissions/fam_bet.csv \
        --culture bsc40 --lam 0.5 --down-only --name fam_bet_bsc
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import subprocess

import numpy as np
import pandas as pd

from .config import ID_COL, SAMPLE_CSV, SUBMISSIONS, TARGET
from .data import load_test, load_train


def _norm(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--culture", required=True, help="префикс нормализованного имени культуры")
    ap.add_argument("--lam", type=float, default=0.5, help="вес прежнего прогноза")
    ap.add_argument("--down-only", action="store_true")
    ap.add_argument("--cap", default="auto", help="число или auto (= максимум группы в трейне)")
    ap.add_argument("--twin-q", type=float, default=0.0, help="вес SI двойника из трейна (0 = не использовать)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--comment", default="")
    a = ap.parse_args()

    tr, te = load_train(), load_test()
    base = pd.read_csv(a.base)
    sample = pd.read_csv(SAMPLE_CSV)
    assert list(base.columns) == [ID_COL, TARGET] and len(base) == len(te)
    assert (base[ID_COL].to_numpy() == te[ID_COL].to_numpy()).all()

    mtr = tr["Culture"].map(_norm).str.startswith(_norm(a.culture)).to_numpy()
    mte = te["Culture"].map(_norm).str.startswith(_norm(a.culture)).to_numpy()
    if not mtr.any():
        raise SystemExit("в трейне нет строк такой культуры")
    y = tr.loc[mtr, TARGET].to_numpy(float)
    mean_g = float(y.mean())
    cap = float(y.max()) if a.cap == "auto" else float(a.cap)
    trg = tr[mtr]
    twin_si = {i: float(trg.loc[trg["smiles_std"] == te["smiles_std"][i], TARGET].mean())
               for i in np.where(mte)[0] if (trg["smiles_std"] == te["smiles_std"][i]).any()}
    use = mte.copy()

    old = base[TARGET].to_numpy(float)
    tgt = a.lam * old + (1 - a.lam) * mean_g
    if a.down_only:
        tgt = np.minimum(old, tgt)
    tgt = np.clip(np.minimum(tgt, cap), 0.0, None)
    if a.twin_q > 0:
        for i, t in twin_si.items():
            tgt[i] = a.twin_q * t + (1 - a.twin_q) * tgt[i]
    new = old.copy()
    new[use] = tgt[use]

    out = pd.DataFrame({ID_COL: te[ID_COL], TARGET: new})
    assert list(out[ID_COL]) == list(sample[ID_COL]) and np.isfinite(new).all()
    path = SUBMISSIONS / f"{a.name}.csv"
    out.to_csv(path, index=False)

    ch = np.where(use & (np.abs(new - old) > 1))[0]
    print(f"группа {a.culture}: трейн n={mtr.sum()}, среднее {mean_g:.0f}, максимум {y.max():.0f}; "
          f"тест {int(mte.sum())} строк, двойников в трейне {len(twin_si)}, изменено {len(ch)}; потолок {cap:.0f}")
    for i in ch[np.argsort(-old[ch])]:
        tw = f"  (двойник в трейне: {twin_si[i]:.0f})" if i in twin_si else ""
        print(f"  TE{i:<4d} {str(te['Title'][i])[:16]:16s} {old[i]:8.0f} -> {new[i]:8.0f}{tw}")
    print(f"среднее файла {old.mean():.1f} -> {new.mean():.1f}; сумма по группе {old[mte].sum():.0f} -> {new[mte].sum():.0f}")

    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        sha = ""
    with open(SUBMISSIONS / "log.csv", "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            dt.datetime.now().strftime("%Y-%m-%d %H:%M"), f"{a.name}.csv", sha,
            f"{a.base.split('/')[-1]}+группа({a.culture})", "", "", round(float(new.mean()), 2),
            round(float(new.max()), 2), "",
            a.comment or f"lam {a.lam}, {'только вниз, ' if a.down_only else ''}потолок {cap:.0f}, среднее группы {mean_g:.0f}, двойники q={a.twin_q}, строк {len(ch)}"])


if __name__ == "__main__":
    main()
