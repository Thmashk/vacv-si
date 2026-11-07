"""Итоговая модель R4 (v2): ядерная регрессия + ExtraTrees -> смесь соседей -> наложение двойников.

Собрана из трёх проверенных на CV улучшений (метрика, близкая к лидерборду: SI обрезан на 10000):
  A. база = 0.7 * KRR (среднее 4 MinMax-ядер по счётным отпечаткам + линейное ядро культуры/штамма, alpha=0.05)
           + 0.3 * ExtraTrees; обе части на log10(1+SI) с поправкой Дуана;
  B. NeighborBlend(lam=0.15, w_ext=1.0), внешние значения обрезаны на 5000;
  C. наложение двойников: если у молекулы есть копия (Tanimoto >= 0.99) в трейне (та же культура) или во внешнем наборе,
     T = среднее значений копий, при T >= 10 прогноз = 0.7 * T * kappa(T) + 0.3 * прогноз смеси, kappa убывает от 0.95 (T<=5000)
     до 0.65 (T>=20000) и затем не обрезается выше twin_cap; остальные строки: x1.2 с потолком 12000.
Ожидаемый эффект скромный (hold_ratio -0.31 -> -0.37 на сидах подбора, -0.30 -> -0.33 на подтверждающих), см. reports/r4_final.md.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator as G
from scipy.spatial.distance import cdist
from sklearn.ensemble import ExtraTreesRegressor

from .neighbors import NeighborBlend, fp_matrix, tanimoto


def _counts(smiles, gen):
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        out.append(gen.GetCountFingerprintAsNumPy(m).astype(np.float64) if m is not None else None)
    n = next((o.shape[0] for o in out if o is not None), 2048)
    return np.array([o if o is not None else np.zeros(n) for o in out])


def _minmax(A, B):
    l1 = cdist(A, B, "cityblock")
    s = A.sum(1)[:, None] + B.sum(1)[None, :]
    return (s - l1) / np.maximum(s + l1, 1e-9)


def _canon_culture(series):
    c = series.astype(str).str.strip().str.lower().str.replace(" cells", "", regex=False).to_numpy()
    return np.array([{"cells hel": "hela", "cells bsc1": "bsc-1"}.get(x, x) for x in c])


def kappa(T):
    return np.interp(T, [5000.0, 20000.0], [0.95, 0.65])


class FinalModel:
    def __init__(self, ctx_idx, alpha=0.05, w_et=0.3, w_ctx=1.0, lam=0.15, w_ext=1.0, ext_cap=5000.0,
                 twin_alpha=0.7, twin_thr=10.0, sim=0.99, mult=1.2, cap=12000.0, twin_cap=18000.0, seed=0):
        self.ctx_idx = list(ctx_idx)
        self.alpha, self.w_et, self.w_ctx, self.lam, self.w_ext, self.ext_cap = alpha, w_et, w_ctx, lam, w_ext, ext_cap
        self.twin_alpha, self.twin_thr, self.sim, self.mult, self.cap, self.twin_cap = twin_alpha, twin_thr, sim, mult, cap, twin_cap
        self.seed = seed
        self._gens = [G.GetMorganGenerator(radius=2, fpSize=2048),
                      G.GetMorganGenerator(radius=2, fpSize=2048, atomInvariantsGenerator=G.GetMorganFeatureAtomInvGen()),
                      G.GetRDKitFPGenerator(maxPath=5, fpSize=2048),
                      G.GetTopologicalTorsionGenerator(fpSize=2048)]

    def _cnt(self, smiles):
        return [_counts(smiles, g) for g in self._gens]

    def _kernel(self, cnt_a, ctx_a, cnt_b, ctx_b):
        kf = sum(_minmax(a, b) for a, b in zip(cnt_a, cnt_b)) / len(cnt_a)
        return kf + self.w_ctx * (ctx_a @ ctx_b.T) / max(ctx_a.shape[1], 1)

    def fit(self, train, X, y, ext):
        """train: DataFrame (smiles_std, Culture); X: матрица FeatureBuilder; y: SI; ext: DataFrame (smiles_std, SI)."""
        self.y_ = np.asarray(y, float)
        self.train_, self.ext_ = train, ext
        self.cnt_ = self._cnt(train["smiles_std"])
        self.ctx_ = np.asarray(X)[:, self.ctx_idx].astype(float)
        t = np.log10(self.y_ + 1.0)
        self.mu_ = float(t.mean())
        K = self._kernel(self.cnt_, self.ctx_, self.cnt_, self.ctx_)
        self.coef_ = np.linalg.solve(K + self.alpha * np.eye(len(K)), t - self.mu_)
        self.krr_smear_ = float(np.mean(10 ** (t - (self.mu_ + K @ self.coef_))))
        et = ExtraTreesRegressor(n_estimators=300, max_features=0.3, min_samples_leaf=1, bootstrap=True,
                                 oob_score=True, random_state=self.seed, n_jobs=2).fit(X, t)
        self.et_, self.et_smear_ = et, float(np.mean(10 ** (t - et.oob_prediction_)))
        self.tfp_ = fp_matrix(train["smiles_std"])
        self.efp_ = fp_matrix(ext["smiles_std"])
        self.nb_ = NeighborBlend(lam=self.lam, w_ext=self.w_ext, ext_cap=self.ext_cap).fit(
            train, self.y_, ext, ext_fp=self.efp_, train_fp=self.tfp_)
        self.tcult_ = _canon_culture(train["Culture"])
        return self

    def base(self, df, X):
        cnt = self._cnt(df["smiles_std"])
        ctx = np.asarray(X)[:, self.ctx_idx].astype(float)
        krr = np.clip(10 ** (self.mu_ + self._kernel(cnt, ctx, self.cnt_, self.ctx_) @ self.coef_) * self.krr_smear_ - 1.0, 0.0, None)
        et = np.clip(10 ** self.et_.predict(X) * self.et_smear_ - 1.0, 0.0, None)
        return (1 - self.w_et) * krr + self.w_et * et

    def predict(self, df, X):
        qfp = fp_matrix(df["smiles_std"])
        pred = self.nb_.predict(df, self.base(df, X), query_fp=qfp)
        s_tr, s_ex = tanimoto(qfp, self.tfp_), tanimoto(qfp, self.efp_)
        qc = _canon_culture(df["Culture"])
        exv = np.minimum(self.ext_["SI"].to_numpy(float), self.ext_cap)
        out = pred.copy()
        for i in range(len(out)):
            vals = list(self.y_[(s_tr[i] >= self.sim) & (self.tcult_ == qc[i])]) + list(exv[s_ex[i] >= self.sim])
            T = float(np.mean(vals)) if vals else 0.0
            if vals and T >= self.twin_thr:
                out[i] = min(self.twin_alpha * T * kappa(T) + (1 - self.twin_alpha) * out[i], self.twin_cap)
            else:
                out[i] = min(out[i] * self.mult, self.cap)
        return out
