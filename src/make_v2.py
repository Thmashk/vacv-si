"""Сборка файлов v2 (R4): ряд камфана/фенхана по исправленной FamilyModel
плюс правило группы BSC-40 (src/doc_gate.py).

База — submissions/opt_mix3.csv. Меняются только:
  * строки теста из ряда (кроме молекул, точно известных по трейну):
    новое = w·FamilyModel + (1−w)·база; на «ключевых» строках вес w_key;
  * строки группы BSC-40 с записью во внешнем документе (режим ext)
    или с двойником в трейне (режим twin).

«Ключевые» строки — те, где решается ставка на стереоконфигурацию: амиды
борнана RRS (конфигурация гиганта 8e) и их C2-эпимеры RRR, у которых в трейне
есть партнёр RRS. Значения на них не проверяются публичной частью лидерборда
(их там почти наверняка нет), поэтому три варианта w_key различаются только
выбором ставки:
  w_key = 1.0  «bold»  — полная вера модели ряда;
  w_key = 0.5  «mid»   — как fam_bet;
  w_key = 0.0  «gen»   — как общая модель (opt_mix3).

    python -m src.make_v2 --name fam2_mid_doc  --w-key 0.5 --bsc ext
    python -m src.make_v2 --name fam2_bold_doc --w-key 1.0 --bsc ext
    python -m src.make_v2 --name fam2_gen_doc  --w-key 0.0 --bsc ext
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import subprocess

import numpy as np
import pandas as pd
from rdkit import Chem

from .config import ID_COL, SAMPLE_CSV, SUBMISSIONS, TARGET
from .data import load_test, load_train
from .doc_gate import group_override, load_external
from .family_model import FamilyModel, describe


def _flat(smiles: str) -> str:
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=False)


def key_rows(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Булева маска ключевых строк теста (см. docstring модуля)."""
    partners: dict[str, set] = {}
    for s in train["smiles_std"]:
        d = describe(s)
        if d is not None:
            partners.setdefault(_flat(s), set()).add(d[0])
    mask = np.zeros(len(test), bool)
    for i, s in enumerate(test["smiles_std"]):
        d = describe(s)
        if d is None:
            continue
        mask[i] = d[0] == "born_RRS" or (d[0] == "born_RRR" and "born_RRS" in partners.get(_flat(s), set()))
    return mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(SUBMISSIONS / "opt_mix3.csv"))
    ap.add_argument("--w", type=float, default=0.5, help="вес FamilyModel на обычных строках ряда")
    ap.add_argument("--w-key", type=float, default=0.5, help="вес FamilyModel на ключевых строках")
    ap.add_argument("--bsc", choices=["ext", "twin", "none"], default="ext")
    ap.add_argument("--name", required=True)
    ap.add_argument("--comment", default="")
    a = ap.parse_args()

    tr, te = load_train(), load_test()
    base = pd.read_csv(a.base)
    sample = pd.read_csv(SAMPLE_CSV)
    assert list(base.columns) == [ID_COL, TARGET] and len(base) == len(te)
    assert (base[ID_COL].to_numpy() == te[ID_COL].to_numpy()).all()
    assert (te[ID_COL].to_numpy() == np.arange(len(te))).all()

    fm = FamilyModel().fit(tr["smiles_std"], tr[TARGET])
    fam = fm.predict(list(te["smiles_std"]))
    in_family = np.array([describe(s) is not None for s in te["smiles_std"]])
    assert np.isfinite(fam[in_family]).all(), "для части строк ряда прогноз модели не посчитан"
    known = te["smiles_std"].isin(set(tr["smiles_std"])).to_numpy()
    use = in_family & ~known
    key = key_rows(tr, te) & use
    assert key.sum() == 4, f"ожидалось 4 ключевые строки, найдено {key.sum()}"

    old = base[TARGET].to_numpy(float)
    new = old.copy()
    w = np.where(key, a.w_key, a.w)
    new[use] = (w * fam + (1 - w) * old)[use]

    n_bsc = 0
    if a.bsc != "none":
        P, info = group_override(tr, te, load_external() if a.bsc == "ext" else None, a.bsc)
        assert not in_family[P.test_row.to_numpy()].any(), "правило BSC-40 задело строки ряда"
        new[P.test_row.to_numpy()] = P.proposal.to_numpy()
        n_bsc = len(P)
        print(f"BSC-40 ({a.bsc}): {n_bsc} строк; {({k: info[k] for k in info if k in ('document', 'q', 'molecules_agree', 'molecules_matched')})}")
    new = np.clip(new, 0.0, None)

    out = pd.DataFrame({ID_COL: te[ID_COL], TARGET: new})
    assert list(out[ID_COL]) == list(sample[ID_COL]) and np.isfinite(new).all()
    out.to_csv(SUBMISSIONS / f"{a.name}.csv", index=False)

    print(f"ряд: {int(use.sum())} строк, ключевые {np.where(key)[0].tolist()}; вес {a.w}, ключевые {a.w_key}")
    for i in np.where(key)[0]:
        print(f"  TE{i:<4d} {str(te['Title'][i])[:14]:14s} база {old[i]:8.0f} -> {new[i]:8.0f}  (модель {fam[i]:8.0f})")
    print(f"среднее {old.mean():.1f} -> {new.mean():.1f}, максимум {new.max():.0f}, изменено строк {int((np.abs(new - old) > 1e-9).sum())}")

    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        sha = ""
    with open(SUBMISSIONS / "log.csv", "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            dt.datetime.now().strftime("%Y-%m-%d %H:%M"), f"{a.name}.csv", sha,
            f"opt_mix3+ряд(w={a.w},ключ={a.w_key})+BSC({a.bsc})", "", "", round(float(new.mean()), 2),
            round(float(new.max()), 2), "",
            a.comment or f"исправленные метки конфигурации; BSC-40 {a.bsc}: {n_bsc} строк"])


if __name__ == "__main__":
    main()
