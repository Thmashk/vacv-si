"""Проверки собранного внешнего датасета.

Если датасета ещё нет, тесты пропускаются: коллекторы ходят в сеть и в
обычном прогоне `make test` не запускаются.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "processed" / "vacv_external_dataset.csv"
REQUIRED_FIRST = ["smiles", "IC50", "CC50", "SI", "doi"]
DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")

pytestmark = pytest.mark.skipif(
    not DATASET.exists(),
    reason="нет data/processed/vacv_external_dataset.csv, сначала make data",
)


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return pd.read_csv(DATASET)


def test_required_columns_come_first(df):
    assert list(df.columns[:5]) == REQUIRED_FIRST


def test_values_are_positive(df):
    for col in ["IC50", "CC50", "SI"]:
        assert df[col].notna().all(), f"пропуски в {col}"
        assert (df[col] > 0).all(), f"неположительные значения в {col}"


def test_si_equals_ratio(df):
    calc = df["CC50"] / df["IC50"]
    rel = (calc - df["SI"]).abs() / calc.abs().clip(lower=1e-12)
    assert (rel < 0.01).all(), "SI не совпадает с CC50 / IC50"


def test_doi_format(df):
    has = df["doi"].notna()
    assert has.any(), "ни у одной строки нет DOI"
    assert df.loc[has, "doi"].astype(str).str.match(DOI_RE).all()


def test_all_smiles_parse(df):
    bad = [s for s in df["smiles"] if Chem.MolFromSmiles(str(s)) is None]
    assert not bad, f"RDKit не разобрал {len(bad)} SMILES"


def test_one_row_per_molecule(df):
    keys = [Chem.MolToInchiKey(Chem.MolFromSmiles(s)) for s in df["smiles"]]
    assert len(keys) == len(set(keys)), "есть повторяющиеся молекулы"


def test_quality_flags_present(df):
    for col in ["si_reliable", "pairing_quality", "source"]:
        assert col in df.columns, f"нет служебной колонки {col}"
    assert df["si_reliable"].any(), "нет ни одного надёжного измерения"
