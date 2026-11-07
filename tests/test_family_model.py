import numpy as np
import pytest
from rdkit import Chem

from src.config import TARGET
from src.data import load_train
from src.family_model import ACYL, FamilyModel, _canon, describe

NITRO_8 = "CC1(C)[C@@H]2CC[C@@]1(C)[C@@H](NC(=O)c1ccc([N+](=O)[O-])cc1)C2"
NITRO_7 = "CC1(C)[C@@H]2CC[C@@]1(C)[C@H](NC(=O)c1ccc([N+](=O)[O-])cc1)C2"


def test_describe_separates_epimers():
    a, b = describe(NITRO_8), describe(NITRO_7)
    assert a[1:] == b[1:] == ("ewg", "NO2")
    assert a[0] != b[0]


def test_non_family_is_none_and_predict_nan():
    assert describe("CCO") is None and describe(None) is None
    m = FamilyModel().fit([NITRO_8, NITRO_7], [1000.0, 10.0])
    assert np.isnan(m.predict(["CCO"])[0])
    p = m.predict([NITRO_8, NITRO_7])
    assert np.all(np.isfinite(p)) and np.all(p >= 0) and np.all(p <= m.cap)


def test_describe_does_not_depend_on_smiles_atom_order():
    """Регресс: коды CIP сортировались по номеру атома в молекуле, и ключ
    конфигурации менялся от порядка записи SMILES (аудит нашёл 30 из 44 строк)."""
    rng = np.random.default_rng(0)
    smiles = [s for s in load_train()["smiles_std"] if describe(s) is not None] + [NITRO_8, NITRO_7]
    assert len(smiles) >= 18
    for s in smiles:
        d0, m = describe(s), Chem.MolFromSmiles(s)
        for _ in range(8):
            perm = [int(v) for v in rng.permutation(m.GetNumAtoms())]
            shuffled = Chem.MolToSmiles(Chem.RenumberAtoms(m, perm), canonical=False)
            assert describe(shuffled) == d0, (s, shuffled)


def test_config_labels_follow_template_positions():
    """Фенхан-амиды трейна имеют только две конфигурации (SRR и SRS), а не четыре."""
    cfgs = {describe(s)[0] for s in load_train()["smiles_std"] if describe(s) is not None}
    assert {c for c in cfgs if c.startswith("fench")} <= {"fench_SRR", "fench_SRS"}
    assert {c for c in cfgs if c.startswith("born")} <= {"born_RRR", "born_RRS"}


def test_unseen_level_adds_prior_variance():
    m = FamilyModel().fit([NITRO_8, NITRO_7], [1000.0, 10.0])
    cfg, cls, fine = describe(NITRO_8)
    _, extra_seen = m._x((cfg, cls, fine))
    _, extra_new = m._x((cfg, cls, "NEW"))
    assert extra_seen == 0.0 and extra_new == pytest.approx(m.tau[2] ** 2)
    assert m._x(("fench_XXX", "NEWCLASS", "NEW"))[1] == pytest.approx(sum(t ** 2 for t in m.tau))


def test_fit_rejects_nonfinite_or_negative_si():
    for bad in (float("nan"), float("inf"), -1.0):
        with pytest.raises(ValueError):
            FamilyModel().fit([NITRO_8, NITRO_7], [bad, 10.0])


def test_acyl_keys_are_canonical_and_stereo_free():
    assert ACYL and all(_canon(k) == k and "@" not in k for k in ACYL)


def test_honest_leave_one_out_on_train_family():
    """Цифра из аудита: 0.77 в log10(1+SI); у константы 1.33, у общей модели 1.03."""
    tr = load_train()
    idx = [i for i, s in enumerate(tr["smiles_std"]) if describe(s) is not None]
    smi, y = tr["smiles_std"].to_numpy(), tr[TARGET].to_numpy(float)
    err = []
    for i in idx:
        rest = [j for j in idx if j != i]
        mu, _ = FamilyModel().fit(smi[rest], y[rest]).predict_log([smi[i]])
        err.append(mu[0] - np.log10(1.0 + y[i]))
    rmse = float(np.sqrt(np.mean(np.square(err))))
    assert len(idx) == 18 and rmse == pytest.approx(0.767, abs=0.01)
