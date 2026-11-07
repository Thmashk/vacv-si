"""Проверки объединённого внешнего датасета.

Повторяют tests/test_dataset.py, но для data/processed/vacv_external_merged.csv —
файла, который берут для обучения. Собирается скриптом
src/data/merge_external.py, см. reports/r2_merge_report.md.

Если файла нет, тесты пропускаются: он собирается из двух выгрузок, которые
в обычном прогоне make test не запускаются.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "processed" / "vacv_external_merged.csv"
REQUIRED_FIRST = ["smiles", "IC50", "CC50", "SI", "doi"]
DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")

pytestmark = pytest.mark.skipif(
    not DATASET.exists(),
    reason="нет data/processed/vacv_external_merged.csv, "
           "сначала python src/data/merge_external.py",
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
    """Каждая строка собрана из одной публикации, поэтому формула точна.

    Медиана SI по нескольким публикациям здесь не используется именно
    поэтому: она ломала это равенство на 15 молекулах.
    """
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


def test_skeleton_matches_rdkit(df):
    """Колонка skeleton должна быть первыми 14 символами ключа от RDKit.

    По ней идёт дедупликация, и если она разойдётся с реальной структурой,
    молекулы начнут склеиваться или двоиться незаметно.
    """
    bad = []
    for smiles, skeleton in zip(df["smiles"], df["skeleton"]):
        key = Chem.MolToInchiKey(Chem.MolFromSmiles(str(smiles)))
        if key[:14] != str(skeleton):
            bad.append((smiles, skeleton, key[:14]))
    assert not bad, f"скелет не совпал с RDKit на {len(bad)} строках: {bad[:3]}"


def test_quality_flags_present(df):
    for col in ["si_reliable", "pairing_quality", "source", "origin", "quality"]:
        assert col in df.columns, f"нет служебной колонки {col}"
    assert df["si_reliable"].any(), "нет ни одного надёжного измерения"


def test_no_censored_values_marked_reliable(df):
    """Усечённое измерение не может считаться надёжным.

    Главный урок сравнения с набором из PR #1: там si_reliable ставился в
    False только когда усечены обе величины, и 139 записей с IC50 равным
    CC50 проходили фильтр, давая SI ровно 1.
    """
    reliable = df[df["si_reliable"]]
    assert not (reliable["IC50"] == reliable["CC50"]).any(), (
        "среди надёжных есть записи, где IC50 в точности равен CC50")
    for col in ["ic50_censored", "cc50_censored"]:
        assert not reliable[col].any(), f"среди надёжных есть усечённые по {col}"
