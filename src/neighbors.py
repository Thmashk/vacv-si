"""Смесь «модель + похожие молекулы» (оценка Надарая–Уотсона с априором от модели).

pred = (sum_j w_j * v_j + lam * base) / (sum_j w_j + lam),  w_j = sim_j**p * множители.
Для точной копии (sim=1) вес ~1 и прогноз близок к известному значению, для далёких молекул
сумма весов ~0 и прогноз возвращается к базовой модели. Внешние значения ограничены сверху.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fp_matrix(smiles) -> np.ndarray:
    out = np.zeros((len(smiles), 2048), dtype=np.float32)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        if m is not None:
            out[i] = _GEN.GetFingerprintAsNumPy(m)
    return out


def tanimoto(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    inter = A @ B.T
    return inter / np.maximum(A.sum(1)[:, None] + B.sum(1)[None, :] - inter, 1e-9)


class NeighborBlend:
    def __init__(self, p: float = 20.0, lam: float = 0.3, w_ext: float = 0.5,
                 culture_factor: float = 0.3, ext_cap: float = 5000.0):
        self.p, self.lam, self.w_ext = p, lam, w_ext
        self.culture_factor, self.ext_cap = culture_factor, ext_cap

    def fit(self, train: pd.DataFrame, y, ext: pd.DataFrame | None = None, ext_fp=None, train_fp=None):
        self.fp_ = [fp_matrix(train["smiles_std"]) if train_fp is None else train_fp]
        self.val_ = [np.asarray(y, float)]
        self.cult_ = [train["Culture"].astype(str).str.strip().str.lower().to_numpy()]
        self.isext_ = [np.zeros(len(train), bool)]
        if ext is not None and len(ext):
            self.fp_.append(fp_matrix(ext["smiles_std"]) if ext_fp is None else ext_fp)
            self.val_.append(np.minimum(ext["SI"].to_numpy(float), self.ext_cap))
            self.cult_.append(np.full(len(ext), "__external__"))
            self.isext_.append(np.ones(len(ext), bool))
        self.fp_, self.val_ = np.vstack(self.fp_), np.concatenate(self.val_)
        self.cult_, self.isext_ = np.concatenate(self.cult_), np.concatenate(self.isext_)
        return self

    def predict(self, query: pd.DataFrame, base: np.ndarray, query_fp=None) -> np.ndarray:
        q = fp_matrix(query["smiles_std"]) if query_fp is None else query_fp
        w = tanimoto(q, self.fp_) ** self.p
        qc = query["Culture"].astype(str).str.strip().str.lower().to_numpy()
        same = (qc[:, None] == self.cult_[None, :]) | self.isext_[None, :]
        w = w * np.where(same, 1.0, self.culture_factor) * np.where(self.isext_, self.w_ext, 1.0)[None, :]
        return (w @ self.val_ + self.lam * np.asarray(base, float)) / (w.sum(1) + self.lam)
