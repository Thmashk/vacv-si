"""Внешние данные как дополнительные обучающие строки.

Внешний набор data/processed/vacv_external_dataset.csv собран из ChEMBL и
PubChem. В нём нет колонок Culture и strain, которые ждёт FeatureBuilder,
зато есть organism. Поэтому здесь строки приводятся к той же форме, что и
строки соревнования:

    Culture = "__external__"   — отдельная категория, модель видит,
                                 что строка пришла из другого источника;
    strain  = organism         — текст организма, из него FeatureBuilder
                                 достаёт флаги штаммов (Copenhagen, WR,
                                 cowpox и так далее).

Здесь же лежат обёртки моделей, умеющие принимать веса строк: сами модели
в src/models.py весов не принимают, а править чужой файл нельзя.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from .config import ROOT, SEED, TARGET
from .data import standardize_smiles
from .models import KernelLogModel, LogSmearingModel

RDLogger.DisableLog("rdApp.*")

EXTERNAL_CSV = ROOT / "data" / "processed" / "vacv_external_dataset.csv"
# Набор R2: отдельно собранный и тщательнее отфильтрованный. В нём нет строк,
# где IC50 в точности равен CC50 (потолок концентрации, записанный дважды,
# даёт мнимый SI = 1); в наборе выше таких строк 301.
EXTERNAL_R2_CSV = ROOT / "data" / "processed" / "vacv_external_usable.csv"
EXTERNAL_CULTURE = "__external__"


def inchikey_block(smiles: str) -> str | None:
    """Первый блок InChIKey: скелет молекулы без стереохимии и соли."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None
    return Chem.MolToInchiKey(mol).split("-")[0]


def load_external(reliable_only: bool = True,
                  clip: float | None = None,
                  weight: float = 1.0) -> pd.DataFrame:
    """Внешние строки в формате, который понимает FeatureBuilder.

    reliable_only — оставить только si_reliable. Ненадёжные записи это пары,
    где цензурированы обе величины («>100» и «>100»), их отношение
    механически равно единице и настоящим измерением не является.
    clip — верхняя граница SI. Внешний максимум 222857 на два порядка выше
    трейна, и такой выброс может перекосить обучение.
    """
    if not EXTERNAL_CSV.exists():
        raise SystemExit(f"нет {EXTERNAL_CSV} — сначала make data")

    df = pd.read_csv(EXTERNAL_CSV)
    if reliable_only and "si_reliable" in df.columns:
        df = df[df["si_reliable"].astype(bool)]
    df = df[df["SI"].notna() & (df["SI"] > 0)].copy()

    si = df["SI"].to_numpy(dtype=float)
    if clip is not None:
        si = np.minimum(si, float(clip))

    out = pd.DataFrame({
        "SMILES": df["smiles"].to_numpy(),
        "smiles_std": [standardize_smiles(s) or s for s in df["smiles"]],
        "Culture": EXTERNAL_CULTURE,
        "strain": df["organism"].fillna("").astype(str).to_numpy(),
        TARGET: si,
        "weight": float(weight),
        "is_external": True,
    })
    out["ik_block"] = [inchikey_block(s) for s in out["smiles_std"]]
    return out[out["smiles_std"].notna()].reset_index(drop=True)


def load_external_r2(reliable_only: bool = True,
                     clip: float | None = None,
                     weight: float = 1.0) -> pd.DataFrame:
    """Набор R2 из data/processed/vacv_external_usable.csv.

    Схема другая: вместо pairing_quality колонка quality, вместо organism
    ничего подходящего нет, поэтому strain остаётся пустым и флаги штаммов
    для внешних строк не поднимаются. Надёжной считается запись с
    quality == "exact" и без цензуры в обеих величинах.
    """
    if not EXTERNAL_R2_CSV.exists():
        raise SystemExit(f"нет {EXTERNAL_R2_CSV}")

    df = pd.read_csv(EXTERNAL_R2_CSV)
    if reliable_only:
        ok = pd.Series(True, index=df.index)
        if "quality" in df.columns:
            ok &= df["quality"].astype(str).eq("exact")
        for c in ("ic50_censored", "cc50_censored"):
            if c in df.columns:
                ok &= ~df[c].fillna(False).astype(bool)
        df = df[ok]
    df = df[df["SI"].notna() & (df["SI"] > 0)].copy()

    si = df["SI"].to_numpy(dtype=float)
    if clip is not None:
        si = np.minimum(si, float(clip))

    out = pd.DataFrame({
        "SMILES": df["smiles"].to_numpy(),
        "smiles_std": [standardize_smiles(s) or s for s in df["smiles"]],
        "Culture": EXTERNAL_CULTURE,
        "strain": "",
        TARGET: si,
        "weight": float(weight),
        "is_external": True,
    })
    if "skeleton" in df.columns:
        out["ik_block"] = df["skeleton"].to_numpy()
    else:
        out["ik_block"] = [inchikey_block(s) for s in out["smiles_std"]]
    return out[out["smiles_std"].notna()].reset_index(drop=True)


def load_any(which: str, **kw) -> pd.DataFrame:
    """which: 'mine' — набор из PR #1, 'r2' — набор коллеги."""
    return load_external_r2(**kw) if which == "r2" else load_external(**kw)


def add_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет ik_block к строкам соревнования."""
    df = df.copy()
    df["ik_block"] = [inchikey_block(s) for s in df["smiles_std"]]
    df["is_external"] = False
    df["weight"] = 1.0
    return df


# ---------------------------------------------------------------- обёртки


class WeightedLogSmearing(LogSmearingModel):
    """LogSmearingModel, принимающий веса строк.

    Бустинг scikit-learn умеет sample_weight, но базовый класс его не
    пробрасывает. Поправка Дуана тоже считается с весами, иначе внешние
    строки исказили бы множитель обратного преобразования.
    """

    name = "лог+smearing с весами"

    def fit(self, X, y, sample_weight=None):
        from sklearn.ensemble import HistGradientBoostingRegressor

        y = np.asarray(y, dtype=float)
        t = np.log10(y + 1.0)
        self.model_ = HistGradientBoostingRegressor(
            max_depth=4, max_iter=400, learning_rate=0.06,
            min_samples_leaf=5, l2_regularization=1.0, random_state=self.seed,
        )
        self.model_.fit(X, t, sample_weight=sample_weight)
        resid = t - self.model_.predict(X)
        if self.smearing:
            w = np.ones(len(resid)) if sample_weight is None else np.asarray(sample_weight, float)
            self.smear_ = float(np.average(10.0 ** resid, weights=w))
        else:
            self.smear_ = 1.0
        return self


class WeightedKernelLog(KernelLogModel):
    """Ядерная гребневая регрессия с весами строк.

    Взвешенная задача сводится к замене alpha * I на alpha * diag(1/w):
    строка с меньшим весом сильнее штрафуется за невязку, то есть влияет
    на решение слабее.
    """

    name = "ядро MinMax с весами"

    def fit(self, X, y, sample_weight=None):
        t = np.log10(np.asarray(y, dtype=float) + 1.0)
        self.X_, self.mu_ = X, float(t.mean())
        K = self._k(X, X)
        n = len(K)
        if sample_weight is None:
            reg = self.alpha * np.eye(n)
        else:
            w = np.clip(np.asarray(sample_weight, dtype=float), 1e-6, None)
            reg = self.alpha * np.diag(1.0 / w)
        self.a_ = np.linalg.solve(K + reg, t - self.mu_)
        resid = t - (self.mu_ + K @ self.a_)
        ww = np.ones(n) if sample_weight is None else np.asarray(sample_weight, float)
        self.smear_ = float(np.average(10.0 ** resid, weights=ww))
        return self


def build_model(kind: str, fp_idx=None, ctx_idx=None, seed: int = SEED):
    if kind == "log":
        return WeightedLogSmearing(seed=seed)
    if kind == "kernel":
        return WeightedKernelLog(fp_idx=fp_idx, ctx_idx=ctx_idx, seed=seed)
    raise SystemExit(f"неизвестная модель: {kind}")
