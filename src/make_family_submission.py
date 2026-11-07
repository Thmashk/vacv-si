"""Файл сабмита = готовая база + поправка по ряду амидов камфана/фенхана (R4).

База (по умолчанию submissions/opt_mix3.csv, LB 1 335 532) не меняется нигде,
кроме тестовых строк из ряда, для которых FamilyModel (src/family_model.py)
даёт прогноз. Там: новое = w·FamilyModel + (1−w)·база.

Режимы:
  bet   — поправка применяется ко всем строкам ряда (вверх и вниз).
  safe  — только вниз: новое = min(база, смесь). Гигантскую ставку не делает.

Строки, чья молекула точно есть в трейне (известные ответы), не трогаются.

    python -m src.make_family_submission --mode bet  --name fam_bet
    python -m src.make_family_submission --mode safe --name fam_safe
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import subprocess

import numpy as np
import pandas as pd

from .config import ID_COL, SAMPLE_CSV, SUBMISSIONS, TARGET
from .data import load_test, load_train
from .family_model import FamilyModel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(SUBMISSIONS / "opt_mix3.csv"))
    ap.add_argument("--mode", choices=["bet", "safe"], default="bet")
    ap.add_argument("--w", type=float, default=0.5, help="вес FamilyModel")
    ap.add_argument("--name", required=True)
    ap.add_argument("--comment", default="")
    a = ap.parse_args()

    tr, te = load_train(), load_test()
    base = pd.read_csv(a.base)
    sample = pd.read_csv(SAMPLE_CSV)
    assert list(base.columns) == [ID_COL, TARGET] and len(base) == len(te)
    assert (base[ID_COL].to_numpy() == te[ID_COL].to_numpy()).all()

    fm = FamilyModel().fit(tr["smiles_std"], tr[TARGET])
    fam = fm.predict(list(te["smiles_std"]))
    known = te["smiles_std"].isin(set(tr["smiles_std"])).to_numpy()
    use = ~np.isnan(fam) & ~known

    old = base[TARGET].to_numpy(float)
    mix = a.w * fam + (1 - a.w) * old
    new = old.copy()
    new[use] = mix[use] if a.mode == "bet" else np.minimum(old[use], mix[use])
    new = np.clip(new, 0.0, None)

    out = pd.DataFrame({ID_COL: te[ID_COL], TARGET: new})
    assert list(out[ID_COL]) == list(sample[ID_COL]) and np.isfinite(new).all()
    path = SUBMISSIONS / f"{a.name}.csv"
    out.to_csv(path, index=False)

    ch = np.where(use & (np.abs(new - old) > 100))[0]
    print(f"ряд: {int(use.sum())} строк в тесте, поправка >100 у {len(ch)}; обучено на {fm.n_} строках трейна")
    for i in ch[np.argsort(-np.abs(new[ch] - old[ch]))]:
        print(f"  TE{i:<4d} {str(te['Title'][i])[:16]:16s} база {old[i]:8.0f} -> {new[i]:8.0f}  (модель ряда {fam[i]:8.0f})")
    print(f"среднее {old.mean():.1f} -> {new.mean():.1f}, максимум {new.max():.0f}")

    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        sha = ""
    log = SUBMISSIONS / "log.csv"
    with open(log, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            dt.datetime.now().strftime("%Y-%m-%d %H:%M"), f"{a.name}.csv", sha,
            f"opt_mix3+ряд({a.mode})", "", "", round(float(new.mean()), 2), round(float(new.max()), 2), "",
            a.comment or f"база {a.base.split('/')[-1]}; вес ряда {a.w}; строк с поправкой {len(ch)}"])


if __name__ == "__main__":
    main()
