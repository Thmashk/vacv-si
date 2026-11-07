"""Защитные тесты. Без зелёных тестов ветки не сливаем.

Главный из них — test_forbidden_columns_never_reach_features. Правила
соревнования запрещают использовать CC50_uM и IC50_uM_VACV: из них SI
получается делением, и случайное попадание их в модель означает
дисквалификацию.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import data, features, models
from src.config import FORBIDDEN, TARGET
from src.data import load_competition, load_test, load_train
from src.make_submission import unify_duplicates, validate

ROOT = Path(__file__).resolve().parent.parent


def test_loader_drops_forbidden_columns():
    for path in (ROOT / "data/raw/train_v2.csv", ROOT / "data/raw/test.csv"):
        df = load_competition(path)
        assert not FORBIDDEN & set(df.columns), f"запрещённые колонки в {path.name}"


def test_forbidden_columns_never_reach_features():
    train = load_train()
    fb = features.FeatureBuilder()
    X, names = fb.fit_transform(train)
    for bad in FORBIDDEN:
        assert not any(bad.lower() in n.lower() for n in names)
    assert X.shape[0] == len(train)
    assert np.isfinite(X).all(), "в признаках есть NaN или бесконечность"


def test_unsafe_loader_is_not_imported_by_model_code():
    """load_competition_unsafe разрешён только для EDA."""
    for module in (features, models):
        src = inspect.getsource(module)
        assert "load_competition_unsafe" not in src, (
            f"{module.__name__} обращается к небезопасному загрузчику")


def test_target_zero_means_missing_ic50():
    """Фиксируем свойство данных: SI = 0 стоит там, где IC50 не измерен.

    Если организаторы обновят файл, тест упадёт и это нужно будет заметить.
    """
    raw = data.load_competition_unsafe(ROOT / "data/raw/train_v2.csv")
    no_ic50 = raw["IC50_uM_VACV"].isna()
    assert (raw.loc[no_ic50, TARGET] == 0).all(), "заглушка перестала быть нулём"
    assert no_ic50.sum() > 0


def test_si_formula_holds():
    raw = data.load_competition_unsafe(ROOT / "data/raw/train_v2.csv")
    ok = raw["IC50_uM_VACV"].notna() & raw["CC50_uM"].notna() & (raw["IC50_uM_VACV"] > 0)
    calc = raw.loc[ok, "CC50_uM"] / raw.loc[ok, "IC50_uM_VACV"]
    rel = (calc - raw.loc[ok, TARGET]).abs() / calc.abs().clip(lower=1e-9)
    assert (rel < 0.01).all(), "SI больше не равен CC50 / IC50"


def test_submission_format():
    sample = pd.read_csv(ROOT / "data/raw/sample_submit.csv")
    sub = sample.copy()
    sub[sample.columns[1]] = 1.0
    validate(sub, sample)

    bad = sub.copy()
    bad.loc[0, sample.columns[1]] = np.nan
    with pytest.raises(AssertionError):
        validate(bad, sample)


def test_duplicates_get_equal_predictions():
    test = load_test()
    pred = np.arange(len(test), dtype=float)
    unified = unify_duplicates(test, pred)
    grouped = pd.Series(unified).groupby(test["smiles_std"].to_numpy()).nunique()
    assert (grouped == 1).all(), "одинаковым молекулам даны разные прогнозы"


def test_feature_builder_is_deterministic():
    train = load_train()
    a, _ = features.FeatureBuilder().fit_transform(train)
    b, _ = features.FeatureBuilder().fit_transform(train)
    assert np.array_equal(a, b)


def test_model_is_deterministic():
    train = load_train().head(120)
    X, _ = features.FeatureBuilder().fit_transform(train)
    y = train[TARGET].to_numpy(dtype=float)
    p1 = models.ConditionalMeanModel(n_estimators=60).fit(X, y).predict(X)
    p2 = models.ConditionalMeanModel(n_estimators=60).fit(X, y).predict(X)
    assert np.allclose(p1, p2)
