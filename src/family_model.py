"""Структурная модель для ряда амидов камфана/фенхана (R4).

Идея. В тренировочной части лежат амиды борниламина и фенхиламина (Vero,
штамм Copenhagen + cowpox + ectromelia), собранные в факториальную схему:
каркас с фиксированной стереоконфигурацией × ароильный заместитель.
Парные молекулы разнесены по трейну и тесту, а отпечатки Моргана
стереоконфигурацию не видят (7e и 8e для них одна молекула).

Модель: log10(1+SI) = среднее + эффект каркаса(остов+CIP-коды)
+ эффект класса заместителя + эффект тонкого заместителя + шум.
Гауссовы априоры, замкнутое решение (байесовская гребневая регрессия).
Выход — апостериорное предсказательное среднее на сырой шкале с потолком
SI (`cap`), чтобы логнормальный хвост не уходил в бесконечность.

Проверка leave-one-out по 18 строкам трейна: RMSE в log10 0.63 против 1.03
у общей модели (ядро + лог), гигант 8e по остальным строкам предсказан как
~2200 вместо ~120 у общей модели.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

_BORN = Chem.MolFromSmarts("[CX4]1([CH3])([CH3])[CX4]2[CH2][CH2][CX4]1([CH3])[CX4]([NX3,OX2])[CH2]2")
_FENCH = Chem.MolFromSmarts("[CX4]1([CH3])([CH3])[CX4]2[CH2][CH2][CX4]([CH3])([CH2]2)[CX4]1[NX3,OX2]")

# класс и тонкий заместитель по каноническому SMILES ацильной части
ACYL = {
    "NC(=O)c1ccc(Br)cc1": ("halo", "Br"), "NC(=O)c1ccc(Cl)cc1": ("halo", "Cl"),
    "NC(=O)c1ccc(F)cc1": ("halo", "F"),
    "NC(=O)c1ccc(C(F)(F)F)cc1": ("ewg", "CF3"),
    "NC(=O)c1ccc([N+](=O)[O-])cc1": ("ewg", "NO2"),
    "COc1ccc(C(N)=O)cc1": ("edg", "OMe"), "Cc1ccc(C(N)=O)cc1": ("edg", "Me"),
    "NC(=O)c1ccccc1": ("ph", "Ph"),
    "NC(=O)c1cccnc1": ("het", "3Py"), "NC(=O)c1ccncc1": ("het", "4Py"),
    "NC(=O)c1cnccn1": ("het", "Pyz"),
    "NC(=O)C1CCCCC1": ("alk", "cHex"), "NC(=O)C1CCC(F)(F)CC1": ("alk", "F2cHex"),
    "NC(=O)C12C[C@H]3C[C@@H](C1)C[C@@H](C2)C3": ("alk", "Ad"),
}


def describe(smiles: str):
    """(каркас+стерео, класс, заместитель) или None, если молекула не из ряда."""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    for name, patt, i2, ihet in (("born", _BORN, 8, 9), ("fench", _FENCH, 9, 10)):
        if not m.HasSubstructMatch(patt):
            continue
        rdCIPLabeler.AssignCIPLabels(m)
        match = m.GetSubstructMatch(patt)
        labs = [(i, m.GetAtomWithIdx(i).GetProp("_CIPCode")) for i in match
                if m.GetAtomWithIdx(i).HasProp("_CIPCode")]
        cfg = name + "_" + "".join(l for _, l in sorted(labs))
        em = Chem.RWMol(m)
        em.RemoveBond(match[i2], match[ihet])
        keep = [f for f in Chem.GetMolFrags(em) if match[ihet] in f][0]
        for a in sorted([i for i in range(em.GetNumAtoms()) if i not in keep], reverse=True):
            em.RemoveAtom(a)
        cls = ACYL.get(Chem.MolToSmiles(em))
        return (cfg, cls[0], cls[1]) if cls else None
    return None


class FamilyModel:
    def __init__(self, tau_cfg=2.0, tau_cls=1.0, tau_fine=1.0, sigma=0.6, cap=40000.0):
        self.tau = (tau_cfg, tau_cls, tau_fine)
        self.sigma, self.cap = sigma, cap

    def _cols(self, desc):
        return ([("cfg", c) for c in sorted({d[0] for d in desc})]
                + [("cls", c) for c in sorted({d[1] for d in desc})]
                + [("fine", c) for c in sorted({d[2] for d in desc})])

    def _x(self, d):
        x = np.zeros(len(self.cols_))
        for key in (("cfg", d[0]), ("cls", d[1]), ("fine", d[2])):
            if key in self.index_:
                x[self.index_[key]] = 1.0
        return x

    def fit(self, smiles, si):
        desc, y = [], []
        for s, v in zip(smiles, si):
            d = describe(s)
            if d is not None:
                desc.append(d); y.append(np.log10(1.0 + float(v)))
        self.n_ = len(y)
        if self.n_ == 0:
            raise ValueError("в трейне нет строк ряда")
        y = np.asarray(y)
        self.cols_ = self._cols(desc)
        self.index_ = {c: i for i, c in enumerate(self.cols_)}
        X = np.vstack([self._x(d) for d in desc])
        prior = np.array([self.tau[("cfg", "cls", "fine").index(c[0])] for c in self.cols_])
        A = X.T @ X / self.sigma ** 2 + np.diag(1.0 / prior ** 2)
        self.cov_ = np.linalg.inv(A)
        self.mu0_ = float(y.mean())
        self.coef_ = self.cov_ @ (X.T @ (y - self.mu0_)) / self.sigma ** 2
        return self

    def predict_log(self, smiles):
        """(среднее, sd) в log10(1+SI); NaN для молекул вне ряда."""
        mu, sd = np.full(len(smiles), np.nan), np.full(len(smiles), np.nan)
        for i, s in enumerate(smiles):
            d = describe(s)
            if d is None:
                continue
            x = self._x(d)
            mu[i] = self.mu0_ + x @ self.coef_
            sd[i] = np.sqrt(self.sigma ** 2 + x @ self.cov_ @ x)
        return mu, sd

    def predict(self, smiles):
        """Апостериорное среднее SI на сырой шкале (NaN вне ряда)."""
        mu, sd = self.predict_log(smiles)
        gx, gw = np.polynomial.hermite_e.hermegauss(60)
        gw = gw / gw.sum()
        out = np.full(len(mu), np.nan)
        for i in np.where(~np.isnan(mu))[0]:
            out[i] = float(np.sum(gw * np.minimum(10.0 ** (mu[i] + sd[i] * gx) - 1.0, self.cap)))
        return out
