"""Построение матрицы признаков.

Три блока:
  1. счётный фингерпринт Morgan — какие фрагменты есть в молекуле;
  2. дескрипторы RDKit — масса, липофильность, доноры и акцепторы и т. д.;
  3. протокол эксперимента: клеточная культура и штамм вируса.

Третий блок разрешён правилами и даёт много сигнала: средний SI в культуре
BSC-40 на два порядка выше, чем в HeLa.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator

from .config import MORGAN_BITS, MORGAN_RADIUS

_GEN = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS, fpSize=MORGAN_BITS
)

# Короткий, устойчивый набор дескрипторов. Полный descList содержит
# экзотические величины, которые часто падают или дают inf.
_DESC_NAMES = [
    "MolWt", "MolLogP", "MolMR", "TPSA", "NumHAcceptors", "NumHDonors",
    "NumRotatableBonds", "RingCount", "NumAromaticRings", "FractionCSP3",
    "HeavyAtomCount", "NHOHCount", "NOCount", "NumAliphaticRings",
    "NumSaturatedRings", "HallKierAlpha", "BalabanJ", "BertzCT",
    "Chi0v", "Chi1v", "Chi2v", "Kappa1", "Kappa2", "Kappa3",
    "LabuteASA", "MaxPartialCharge", "MinPartialCharge",
    "NumHeteroatoms", "qed",
]
_DESC_FUNCS = [(n, f) for n, f in Descriptors.descList if n in _DESC_NAMES]

# Подстроки штамма. Разные штаммы дают разную чувствительность к веществу.
_STRAIN_KEYS = {
    "strain_copenhagen": "copenhagen",
    "strain_wr": "western reserve",
    "strain_cowpox": "cowpox",
    "strain_ectromelia": "ectromelia",
    "strain_monkeypox": "monkeypox",
    "strain_lister": "lister",
    "strain_ihd": "ihd",
    "strain_variola": "variola",
}


def _mol(smiles: str):
    return Chem.MolFromSmiles(smiles)


def morgan_matrix(smiles: list[str]) -> np.ndarray:
    out = np.zeros((len(smiles), MORGAN_BITS), dtype=np.float32)
    for i, s in enumerate(smiles):
        m = _mol(s)
        if m is not None:
            out[i] = _GEN.GetCountFingerprintAsNumPy(m)
    return out


def descriptor_matrix(smiles: list[str]) -> tuple[np.ndarray, list[str]]:
    rows = []
    for s in smiles:
        m = _mol(s)
        if m is None:
            rows.append([np.nan] * len(_DESC_FUNCS))
            continue
        vals = []
        for _, fn in _DESC_FUNCS:
            try:
                vals.append(float(fn(m)))
            except Exception:
                vals.append(np.nan)
        rows.append(vals)
    arr = np.asarray(rows, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr, [f"desc_{n}" for n, _ in _DESC_FUNCS]


def _norm_culture(x) -> str:
    return str(x).strip().lower()


class FeatureBuilder:
    """Фитится на train, применяется к train и test одинаково.

    Категории культур фиксируются по трейну: так в тесте не появится
    колонка, которой модель не видела при обучении.
    """

    def __init__(self, min_culture_count: int = 3):
        self.min_culture_count = min_culture_count
        self.cultures_: list[str] = []
        self.names_: list[str] = []

    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        counts = df["Culture"].map(_norm_culture).value_counts()
        self.cultures_ = sorted(counts[counts >= self.min_culture_count].index)
        return self

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        smiles = df["smiles_std"].tolist()

        fp = morgan_matrix(smiles)
        fp_names = [f"fp_{i}" for i in range(fp.shape[1])]

        desc, desc_names = descriptor_matrix(smiles)

        cult = df["Culture"].map(_norm_culture)
        cult_mat = np.zeros((len(df), len(self.cultures_) + 1), dtype=np.float32)
        for j, c in enumerate(self.cultures_):
            cult_mat[:, j] = (cult == c).to_numpy(dtype=np.float32)
        cult_mat[:, -1] = (~cult.isin(self.cultures_)).to_numpy(dtype=np.float32)
        cult_names = [f"cult_{c}" for c in self.cultures_] + ["cult_прочее"]

        strain = df["strain"].astype(str).str.lower()
        strain_mat = np.column_stack(
            [strain.str.contains(key, regex=False).to_numpy(dtype=np.float32)
             for key in _STRAIN_KEYS.values()]
        )
        strain_names = list(_STRAIN_KEYS)

        X = np.hstack([fp, desc, cult_mat, strain_mat]).astype(np.float32)
        self.names_ = fp_names + desc_names + cult_names + strain_names
        return X, self.names_

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        return self.fit(df).transform(df)
