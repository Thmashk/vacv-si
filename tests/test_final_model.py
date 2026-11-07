import numpy as np
import pandas as pd

from src.final_model import FinalModel

SMILES = ["CCO", "CCN", "CCC", "c1ccccc1", "c1ccccc1O", "c1ccccc1N", "CC(=O)O", "CCOC", "CCCl", "CCBr", "CCCC", "CCCCC"]


def _df(smiles):
    return pd.DataFrame({"smiles_std": smiles, "Culture": ["Vero"] * len(smiles)})


def test_final_model_predicts_finite_nonnegative_and_caps_twins():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(len(SMILES), 6)).astype(np.float32)
    y = np.array([0, 1, 5, 20, 3, 100, 0, 8, 2, 40, 6, 7], float)
    y[0] = 36000.0  # гигант
    ext = pd.DataFrame({"smiles_std": ["CCCCCC"], "SI": [50.0]})
    m = FinalModel(ctx_idx=[0, 1, 2], twin_cap=18000.0).fit(_df(SMILES), X, y, ext)
    pred = m.predict(_df(["CCO", "CCCCCCC"]), X[:2])
    assert np.isfinite(pred).all() and (pred >= 0).all()
    assert pred[0] <= 18000.0  # копия гиганта ограничена twin_cap
