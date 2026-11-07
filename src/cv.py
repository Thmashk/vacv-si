"""Схема валидации и метрики.

Разбиение идёт по скаффолдам Бемиса–Мурко: молекулы с одинаковым остовом
попадают в один фолд. Без этого серии аналогов оказываются по обе стороны
разбиения и оценка получается завышенной.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedGroupKFold

from .config import BIN_EDGES, N_REPEATS, N_SPLITS, SEED


def murcko_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        s = Chem.MolToSmiles(scaf)
        return s if s else smiles
    except Exception:
        return smiles


def scaffold_groups(smiles: pd.Series) -> np.ndarray:
    return smiles.map(murcko_scaffold).to_numpy()


def target_bins(y: np.ndarray) -> np.ndarray:
    """Номер бина целевой переменной. Используется и для страт, и для модели."""
    return np.digitize(np.asarray(y, dtype=float), BIN_EDGES[1:], right=True)


def iter_splits(y: np.ndarray, groups: np.ndarray,
                n_splits: int = N_SPLITS, n_repeats: int = N_REPEATS,
                seed: int = SEED):
    """Повторяемое стратифицированное разбиение по группам."""
    strata = target_bins(y)
    for r in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for fold, (tr, va) in enumerate(cv.split(np.zeros(len(y)), strata, groups)):
            yield r, fold, tr, va


def mse_raw(y_true, y_pred) -> float:
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def rmse_log(y_true, y_pred) -> float:
    a = np.log10(np.clip(np.asarray(y_true, dtype=float), 0, None) + 1.0)
    b = np.log10(np.clip(np.asarray(y_pred, dtype=float), 0, None) + 1.0)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def spearman(y_true, y_pred) -> float:
    if len(np.unique(y_pred)) < 2:
        return 0.0
    return float(spearmanr(y_true, y_pred).statistic)


def score_all(y_true, y_pred) -> dict[str, float]:
    return {
        "mse_raw": mse_raw(y_true, y_pred),
        "rmse_log": rmse_log(y_true, y_pred),
        "spearman": spearman(y_true, y_pred),
    }


def bootstrap_mse_ci(y_true, y_pred, n_boot: int = 2000, seed: int = SEED):
    """Доверительный интервал для MSE. При тяжёлом хвосте он очень широкий,
    и это само по себе важный результат: разницу в 5% измерить нельзя."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[i] = np.mean((y_true[idx] - y_pred[idx]) ** 2)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
