import numpy as np
import pandas as pd

from src.neighbors import NeighborBlend


def _df(smiles):
    return pd.DataFrame({"smiles_std": smiles, "Culture": ["Vero"] * len(smiles)})


def test_exact_twin_pulls_towards_known_value_and_far_molecule_keeps_base():
    train = _df(["CCO", "c1ccccc1O"])
    nb = NeighborBlend().fit(train, np.array([100.0, 1.0]))
    pred = nb.predict(_df(["CCO", "CCCCCCCCCCCCCCCCCC(=O)O"]), np.array([10.0, 10.0]))
    assert pred[0] > 70  # близко к известным 100
    assert abs(pred[1] - 10.0) < 1.0  # далёкая молекула: остаётся прогноз модели


def test_external_values_are_capped():
    train = _df(["CCO"])
    ext = pd.DataFrame({"smiles_std": ["CCN"], "SI": [222857.0]})
    nb = NeighborBlend(ext_cap=5000.0).fit(train, np.array([1.0]), ext)
    assert nb.val_.max() <= 5000.0
