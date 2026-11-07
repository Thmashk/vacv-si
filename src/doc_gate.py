"""Правило «документ-ворота» для группы культуры (R4).

Внешний набор (data/processed/vacv_external_usable.csv, записи ChEMBL по вирусу
осповакцины, собраны скриптами r2) для культуры BSC-40 содержит документ
PMID17335190, записи которого воспроизводят ответы трейна на 15 строках из 15
(11 молекул из 11, отклонение не больше 1e-5). Правило, без ручных значений:

 1. Молекула теста совпала с записью набора: стереонезависимые счётные отпечатки
    Моргана, Tanimoto >= 0.99 после канонизации таутомера.
 2. «Проверенный документ» группы выбирается скриптом по трейну: его записи
    воспроизводят ответы (в пределах 2%) не менее чем в 90% строк группы, строк
    не меньше 8.
 3. Строка теста группы с записью в проверенном документе получает
        q * медиана(SI записей) + (1 - q) * m_G,
    где q = (k + 1) / (n + 2) — доля согласия по молекулам трейна (Лаплас),
    m_G — среднее ответов трейна по молекулам группы.
 4. Нет записи — прогноз не трогаем.

Режим `twin` не использует внешних данных: строка теста, чья молекула (после
канонизации таутомера) есть в трейне в той же группе, получает ответ трейна.

Политика. Внешняя запись здесь возвращает ответ тестовой строки; это тот же
механизм, что у NeighborBlend (точные копии), но он требует подтверждения
капитана и записи в SOURCES.md / DATASET_CARD.md. Режим `twin` от внешних
данных не зависит.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from .config import PROCESSED, TARGET
from .data import standardize_smiles
from .neighbors import fp_matrix, tanimoto

EXTERNAL_CSV = PROCESSED / "vacv_external_usable.csv"
SIM, TOL, GATE_N, GATE_AGREE = 0.99, 0.02, 8, 0.9
GROUP = "BSC-40"

_TE = rdMolStandardize.TautomerEnumerator()


def tautomer_smiles(smiles: str) -> str:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return smiles
    try:
        return Chem.MolToSmiles(_TE.Canonicalize(m))
    except Exception:
        return smiles


def load_external(path=EXTERNAL_CSV) -> pd.DataFrame:
    ex = pd.read_csv(path)
    ex["smiles_std"] = ex["smiles"].map(standardize_smiles)
    return ex.dropna(subset=["smiles_std"]).reset_index(drop=True)


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= TOL * max(abs(a), abs(b), 1e-9) + 1e-6


def group_override(train: pd.DataFrame, test: pd.DataFrame, ext: pd.DataFrame | None,
                   mode: str = "ext", culture: str = GROUP):
    """Прогнозы для строк теста группы `culture`.

    mode="ext":  проверенный документ внешнего набора (см. модуль).
    mode="twin": ответ трейна для той же молекулы в той же группе (без внешних данных).
    Возвращает (DataFrame: test_row, Title, Culture, value, proposal, n_records), сведения.
    Строки без записи/двойника в таблицу не попадают.
    """
    assert mode in ("ext", "twin")
    y = train[TARGET].to_numpy(float)
    gtr = np.where(train["Culture"].astype(str).str.contains(culture, regex=False))[0]
    gte = np.where(test["Culture"].astype(str).str.contains(culture, regex=False))[0]
    trt = [tautomer_smiles(train["smiles_std"].iat[i]) for i in gtr]
    tet = [tautomer_smiles(test["smiles_std"].iat[j]) for j in gte]
    smi = train["smiles_std"].to_numpy()
    mol_y = {s: y[gtr][smi[gtr] == s].mean() for s in pd.unique(smi[gtr])}
    m_g = float(np.mean(list(mol_y.values())))
    info = dict(culture=culture, train_rows=len(gtr), train_molecules=len(mol_y), group_mean=m_g, test_rows=len(gte))
    rows = []

    if mode == "twin":
        by_taut = {}
        for t, i in zip(trt, gtr):
            by_taut.setdefault(t, []).append(y[i])
        for t, j in zip(tet, gte):
            if t in by_taut:
                v = float(np.mean(by_taut[t]))
                rows.append(dict(test_row=int(j), value=v, proposal=v, n_records=len(by_taut[t])))
    else:
        ext_t = [tautomer_smiles(s) for s in ext["smiles_std"]]
        docs, si = ext["document"].astype(str).to_numpy(), ext["SI"].to_numpy(float)
        efp = fp_matrix(ext_t)
        s_tr, s_te = tanimoto(fp_matrix(trt), efp), tanimoto(fp_matrix(tet), efp)
        cells: dict[str, list[bool]] = {}
        for a, i in enumerate(gtr):
            recs = np.where(s_tr[a] >= SIM)[0]
            for d in set(docs[recs]):
                cells.setdefault(d, []).append(_close(np.median(si[[r for r in recs if docs[r] == d]]), y[i]))
        cand = sorted([(np.mean(v), len(v), d) for d, v in cells.items()
                       if len(v) >= GATE_N and np.mean(v) >= GATE_AGREE], reverse=True)
        if not cand:
            info.update(document=None)
            return pd.DataFrame(columns=["test_row", "Title", "Culture", "value", "proposal", "n_records"]), info
        doc = cand[0][2]
        k = n = 0
        pos = {i: a for a, i in enumerate(gtr)}
        for s in mol_y:
            ii = gtr[smi[gtr] == s]
            recs = [r for r in np.where(s_tr[pos[ii[0]]] >= SIM)[0] if docs[r] == doc]
            if recs:
                n += 1
                k += int(_close(np.median(si[recs]), y[ii[0]]))
        q = (k + 1) / (n + 2)
        info.update(document=doc, rows_agree=float(cand[0][0]), rows_checked=int(cand[0][1]),
                    molecules_matched=n, molecules_agree=k, q=q)
        for a, j in enumerate(gte):
            recs = [r for r in np.where(s_te[a] >= SIM)[0] if docs[r] == doc]
            if recs:
                v = float(np.median(si[recs]))
                rows.append(dict(test_row=int(j), value=v, proposal=q * v + (1 - q) * m_g, n_records=len(recs)))

    out = pd.DataFrame(rows, columns=["test_row", "value", "proposal", "n_records"])
    out.insert(1, "Title", [test["Title"].iat[j] for j in out.test_row])
    out.insert(2, "Culture", [test["Culture"].iat[j] for j in out.test_row])
    return out.sort_values("test_row").reset_index(drop=True), info
