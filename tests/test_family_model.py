import numpy as np

from src.family_model import FamilyModel, describe

NITRO_8 = "CC1(C)[C@@H]2CC[C@@]1(C)[C@@H](NC(=O)c1ccc([N+](=O)[O-])cc1)C2"
NITRO_7 = "CC1(C)[C@@H]2CC[C@@]1(C)[C@H](NC(=O)c1ccc([N+](=O)[O-])cc1)C2"


def test_describe_separates_epimers():
    a, b = describe(NITRO_8), describe(NITRO_7)
    assert a[1:] == b[1:] == ("ewg", "NO2")
    assert a[0] != b[0]


def test_non_family_is_none_and_predict_nan():
    assert describe("CCO") is None
    m = FamilyModel().fit([NITRO_8, NITRO_7], [1000.0, 10.0])
    assert np.isnan(m.predict(["CCO"])[0])
    p = m.predict([NITRO_8, NITRO_7])
    assert np.all(np.isfinite(p)) and np.all(p >= 0) and np.all(p <= m.cap)
