"""Загрузка соревновательных файлов.

Главное здесь — load_competition: единственная разрешённая точка входа для
данных Kaggle. Она физически удаляет запрещённые колонки, поэтому ни одна
модель не может их увидеть даже по ошибке.
"""
from __future__ import annotations

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import SaltRemover

from ..config import FORBIDDEN, SAMPLE_CSV, SMILES_COL, TARGET, TEST_CSV, TRAIN_CSV

RDLogger.DisableLog("rdApp.*")
_SALT_REMOVER = SaltRemover.SaltRemover()


def load_competition(path) -> pd.DataFrame:
    """Читает CSV Kaggle и сразу выбрасывает запрещённые колонки."""
    df = pd.read_csv(path)
    drop = [c for c in df.columns if c in FORBIDDEN]
    return df.drop(columns=drop)


def load_competition_unsafe(path) -> pd.DataFrame:
    """Читает файл целиком, вместе с запрещёнными колонками.

    Только для EDA и отчётов. Ничто в src/features.py и src/models.py этого
    не импортирует, и тест tests/test_guard.py это проверяет.
    """
    return pd.read_csv(path)


def standardize_smiles(smiles: str) -> str | None:
    """Канонический SMILES: крупнейший органический фрагмент без солей."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = _SALT_REMOVER.StripMol(mol, dontRemoveEverything=True)
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if not frags:
        return None
    mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return Chem.MolToSmiles(mol)


def _add_std(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["smiles_std"] = df[SMILES_COL].map(standardize_smiles)
    df["smiles_ok"] = df["smiles_std"].notna()
    df["smiles_std"] = df["smiles_std"].fillna(df[SMILES_COL])
    return df


def load_train() -> pd.DataFrame:
    df = load_competition(TRAIN_CSV)
    if TARGET not in df.columns:
        raise ValueError(f"в {TRAIN_CSV} нет колонки {TARGET!r}")
    df = df[df[TARGET].notna()].reset_index(drop=True)
    return _add_std(df)


def load_test() -> pd.DataFrame:
    return _add_std(load_competition(TEST_CSV))


def load_sample_submission() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_CSV)
