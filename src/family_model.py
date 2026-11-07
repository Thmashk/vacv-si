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

Проверка leave-one-out по 18 строкам трейна (после исправления разметки
конфигурации, см. ниже): RMSE в log10 0.77 против 1.03 у общей модели
(ядро + лог) и 1.33 у константы. Это честная цифра: первая версия давала 0.63
из-за ошибки порядка CIP-кодов, аудит её нашёл (reports/r4_family.md).

Известные ограничения. Модель обучена на 18 строках, структура подобрана
после просмотра этих строк. Разность 7-й и 8-й серий (RRR против RRS) опирается
на две строки и аддитивность. Потолок `cap` — допущение, а не оценка: при
cap 20 000 апостериорное среднее 8b вдвое ниже, при 100 000 вдвое выше.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

# Позиции в SMARTS: атомы 0,1,2 — мостик с gem-диметилом, 3 — мостиковая голова
# C4, 6 — мостиковая голова C1 с метилом. Стереоцентры шаблона: 3, 6 и атом с
# гетероатомом (C2). Коды берём по ПОЗИЦИИ в шаблоне, а не по номеру атома в
# молекуле: номер атома зависит от порядка записи SMILES.
_BORN = Chem.MolFromSmarts("[CX4]1([CH3])([CH3])[CX4]2[CH2][CH2][CX4]1([CH3])[CX4]([NX3,OX2])[CH2]2")
_FENCH = Chem.MolFromSmarts("[CX4]1([CH3])([CH3])[CX4]2[CH2][CH2][CX4]([CH3])([CH2]2)[CX4]1[NX3,OX2]")
_SERIES = (("born", _BORN, (3, 6, 8), 8, 9), ("fench", _FENCH, (3, 6, 9), 9, 10))


def _canon(smiles: str) -> str:
    # без стереохимии: адамантан и т.п. не должны зависеть от версии RDKit
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=False)


# класс и тонкий заместитель по каноническому SMILES ацильной части
_ACYL_RAW = {
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
ACYL = {_canon(k): v for k, v in _ACYL_RAW.items()}


def describe(smiles):
    """(каркас+стерео, класс, заместитель) или None, если молекула не из ряда."""
    if not isinstance(smiles, str):
        return None
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    for name, patt, stereo_pos, i2, ihet in _SERIES:
        if not m.HasSubstructMatch(patt):
            continue
        rdCIPLabeler.AssignCIPLabels(m)
        match = m.GetSubstructMatch(patt)
        atoms = [m.GetAtomWithIdx(match[p]) for p in stereo_pos]
        cfg = name + "_" + "".join(a.GetProp("_CIPCode") if a.HasProp("_CIPCode") else "?" for a in atoms)
        em = Chem.RWMol(m)
        em.RemoveBond(match[i2], match[ihet])
        keep = [f for f in Chem.GetMolFrags(em) if match[ihet] in f][0]
        for a in sorted([i for i in range(em.GetNumAtoms()) if i not in keep], reverse=True):
            em.RemoveAtom(a)
        cls = ACYL.get(Chem.MolToSmiles(em, isomericSmiles=False))
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
        """(вектор признаков, дополнительная дисперсия). Уровень, которого не было
        в трейне, эффекта не получает, но его априорная дисперсия tau^2 попадает
        в неопределённость прогноза: иначе sd занижен, а среднее смещено вниз."""
        x = np.zeros(len(self.cols_))
        extra = 0.0
        for kind, tau, level in zip(("cfg", "cls", "fine"), self.tau, d):
            if (kind, level) in self.index_:
                x[self.index_[(kind, level)]] = 1.0
            else:
                extra += tau ** 2
        return x, extra

    def fit(self, smiles, si):
        desc, y = [], []
        for s, v in zip(smiles, si):
            d = describe(s)
            if d is not None:
                if not np.isfinite(float(v)) or float(v) < 0:
                    raise ValueError(f"в строке ряда SI = {v}: ожидается конечное число >= 0")
                desc.append(d); y.append(np.log10(1.0 + float(v)))
        self.n_ = len(y)
        if self.n_ == 0:
            raise ValueError("в трейне нет строк ряда")
        y = np.asarray(y)
        self.cols_ = self._cols(desc)
        self.index_ = {c: i for i, c in enumerate(self.cols_)}
        X = np.vstack([self._x(d)[0] for d in desc])
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
            x, extra = self._x(d)
            mu[i] = self.mu0_ + x @ self.coef_
            sd[i] = np.sqrt(self.sigma ** 2 + x @ self.cov_ @ x + extra)
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
