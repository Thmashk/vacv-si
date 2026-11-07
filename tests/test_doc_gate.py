import inspect

import numpy as np
import pandas as pd
import pytest

from src import doc_gate, make_v2
from src.config import FORBIDDEN, SAMPLE_CSV, SUBMISSIONS
from src.data import load_test, load_train
from src.doc_gate import group_override, load_external
from src.make_submission import validate


@pytest.fixture(scope="module")
def data():
    return load_train(), load_test(), load_external()


def test_new_code_never_names_forbidden_columns():
    for module in (doc_gate, make_v2):
        src = inspect.getsource(module)
        for bad in FORBIDDEN:
            assert bad not in src, f"{module.__name__} упоминает {bad}"


def test_bsc_document_is_found_by_train_agreement(data):
    tr, te, ex = data
    P, info = group_override(tr, te, ex, "ext")
    assert info["document"] == "PMID17335190"
    assert info["rows_checked"] == 15 and info["rows_agree"] == 1.0
    assert info["molecules_matched"] == info["molecules_agree"] == 11
    assert info["q"] == pytest.approx(12 / 13)
    assert len(P) == 18 and (P.proposal > 0).all()
    assert np.allclose(P.proposal, info["q"] * P.value + (1 - info["q"]) * info["group_mean"])
    assert P.loc[P.Title == "tecovirimat", "value"].iloc[0] == 2500.0


def test_twin_mode_returns_train_labels_without_external_data(data):
    tr, te, _ = data
    P, info = group_override(tr, te, None, "twin")
    assert P.test_row.tolist() == [62, 294, 297, 320, 407, 551]
    assert P.value.to_numpy() == pytest.approx([5000.0, 33.333, 2000.0, 2500.0, 5.0, 11.948], rel=1e-3)
    assert (P.value == P.proposal).all()


def test_key_rows_are_the_four_stereo_bet_rows(data):
    tr, te, _ = data
    assert np.where(make_v2.key_rows(tr, te))[0].tolist() == [56, 160, 233, 326]


@pytest.mark.parametrize("name", ["fam2_mid_doc", "fam2_bold_doc", "fam2_gen_doc"])
def test_v2_files_are_valid_submissions(name):
    path = SUBMISSIONS / f"{name}.csv"
    if not path.exists():
        pytest.skip("файл ещё не собран: python -m src.make_v2")
    validate(pd.read_csv(path), pd.read_csv(SAMPLE_CSV))
