import os

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

RAW = ["chembl_vacv_raw.csv", "pubchem_vacv_raw.csv"]
DOCS = ["chembl_docs.csv", "pubchem_docs.csv"]

POTENCY = {"IC50": 0, "EC50": 1, "ID50": 2, "ED50": 3}
CYTOTOX = {"CC50", "TC50", "TD50", "CD50", "Cytotoxicity"}
RATIO = {"Ratio CC50/IC50", "Ratio CC50/EC50", "Selectivity Index", "SI"}

TO_UM = {
    "M": 1e6,
    "mM": 1e3,
    "uM": 1.0,
    "um": 1.0,
    "nM": 1e-3,
    "pM": 1e-6,
}

GP = ["skeleton", "doc_key"]

chooser = rdMolStandardize.LargestFragmentChooser()
uncharger = rdMolStandardize.Uncharger()


def read(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, engine = "python", on_bad_lines = "skip")


def read_all(paths):
    parts = []
    for p in paths:
        if os.path.exists(p) and os.path.getsize(p) > 100:
            d = read(p)
            d["source"] = "PubChem" if "pubchem" in p else "ChEMBL"
            print(" ", p, len(d), "строк")
            parts.append(d)
        else:
            print(" ", p, "нет файла или пуст")
    if not parts:
        raise SystemExit("нет ни одного файла с данными")
    return pd.concat(parts, ignore_index = True, sort = False)


def load_docs():
    have = [p for p in DOCS if os.path.exists(p) and os.path.getsize(p) > 100]
    if not have:
        return pd.DataFrame(columns = ["document_chembl_id", "doi", "pubmed_id"])
    d = pd.concat([read(p) for p in have], ignore_index = True, sort = False)
    return d.drop_duplicates("document_chembl_id")


def as_pmid(x):
    if x is None or (isinstance(x, float) and np.isnan(x)) or str(x).strip() in ("", "nan"):
        return None
    try:
        return "PMID" + str(int(float(x)))
    except Exception:
        return None


def clean(smi):
    if not isinstance(smi, str) or not smi:
        return None, None, None
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None, None, None
    m = chooser.choose(m)
    m = uncharger.uncharge(m)
    try:
        key = Chem.MolToInchiKey(m)
    except Exception:
        key = None
    return Chem.MolToSmiles(m), key, Descriptors.MolWt(m)


def to_um(val, units, mw):
    if pd.isna(val):
        return np.nan
    if units in TO_UM:
        return float(val) * TO_UM[units]
    if units in ("ug.mL-1", "ug/ml") and mw and mw > 0:
        return float(val) * 1000.0 / mw
    return np.nan


def is_vacv(row):
    text = " ".join(str(row.get(c) or "") for c in ("target_organism", "assay_description", "target_pref_name"))
    text = text.lower()
    return "vaccinia" in text or "vacv" in text


def standardize(df):
    cache = {}
    smis, keys, mws = [], [], []
    for s in df["canonical_smiles"]:
        if s not in cache:
            cache[s] = clean(s)
        a, b, c = cache[s]
        smis.append(a)
        keys.append(b)
        mws.append(c)
    df["smiles"] = smis
    df["inchikey"] = keys
    df["mw"] = mws
    df["skeleton"] = [k[:14] if k else None for k in keys]
    return df[df["inchikey"].notna()]


def from_ratio(df):
    r = df[df["standard_type"].isin(RATIO) & df["vacv"]].copy()
    r["SI"] = pd.to_numeric(r["standard_value"], errors = "coerce")
    r = r[r["SI"].notna() & (r["SI"] > 0)]
    g = r.groupby(GP).agg(
        smiles = ("smiles", "first"),
        inchikey = ("inchikey", "first"),
        SI = ("SI", "median"),
        n_si = ("SI", "size"),
        source = ("source", "first"),
    ).reset_index()
    g["quality"] = "reported"
    g["si_source"] = "reported"
    return g


def drop_censored(d):
    if not len(d):
        return d
    d = d.copy()
    d["unc"] = ~d["censored"]
    has = d.groupby(GP)["unc"].max().rename("has_unc").reset_index()
    d = d.merge(has, on = GP)
    return d[(~d["has_unc"]) | d["unc"]]


def from_pairs(df):
    d = df[df["standard_type"].isin(set(POTENCY) | CYTOTOX)].copy()
    d["value_uM"] = [to_um(v, u, m) for v, u, m in zip(d["standard_value"], d["standard_units"], d["mw"])]
    d = d[d["value_uM"].notna() & (d["value_uM"] > 0)]

    pot = d[d["standard_type"].isin(POTENCY) & d["vacv"]].copy()
    tox = d[d["standard_type"].isin(CYTOTOX)].copy()
    print("потенция по вакцинии:", len(pot), "| цитотоксичность:", len(tox))

    pot = drop_censored(pot)
    tox = drop_censored(tox)

    pot["rank"] = pot["standard_type"].map(POTENCY)
    best = pot.groupby(GP)["rank"].min().rename("best").reset_index()
    pot = pot.merge(best, on = GP)
    pot = pot[pot["rank"] == pot["best"]]

    pot_g = pot.groupby(GP).agg(
        smiles = ("smiles", "first"),
        inchikey = ("inchikey", "first"),
        n_forms = ("inchikey", "nunique"),
        IC50 = ("value_uM", "median"),
        ic50_type = ("standard_type", "first"),
        n_ic50 = ("value_uM", "size"),
        ic50_censored = ("censored", "any"),
        source = ("source", "first"),
    ).reset_index()
    tox_g = tox.groupby(GP).agg(
        CC50 = ("value_uM", "median"),
        cc50_type = ("standard_type", "first"),
        n_cc50 = ("value_uM", "size"),
        cc50_censored = ("censored", "any"),
    ).reset_index()

    out = pot_g.merge(tox_g, on = GP, how = "inner")
    out["SI"] = out["CC50"] / out["IC50"]
    tie = (out["CC50"] - out["IC50"]).abs() / out["CC50"] < 1e-6
    out["quality"] = np.where(
        tie, "artifact",
        np.where(out["ic50_censored"] & out["cc50_censored"], "bad",
                 np.where(out["ic50_censored"], "upper_bound",
                          np.where(out["cc50_censored"], "lower_bound", "exact"))))
    print("IC50 равен CC50 (не измерение, а потолок концентрации):", int(tie.sum()))
    out["si_source"] = "paired"

    orphan = pot_g.merge(tox_g, on = GP, how = "left")
    print("потенция без пары CC50:", int(orphan["CC50"].isna().sum()))
    return out


def per_molecule(usable):
    rank = {"exact": 0, "reported": 1, "lower_bound": 2}
    u = usable.copy()
    u["q"] = u["quality"].map(rank)
    u = u.sort_values("q")
    g = u.groupby("skeleton").agg(
        smiles = ("smiles", "first"),
        inchikey = ("inchikey", "first"),
        n_forms = ("inchikey", "nunique"),
        SI = ("SI", "median"),
        si_min = ("SI", "min"),
        si_max = ("SI", "max"),
        n_studies = ("SI", "size"),
        quality = ("quality", "first"),
        IC50 = ("IC50", "median"),
        CC50 = ("CC50", "median"),
        doi = ("doi", "first"),
    ).reset_index()
    g["spread"] = g["si_max"] / g["si_min"].replace(0, np.nan)
    return g


def main():
    df = read_all(RAW)
    print("сырых строк:", len(df))

    docs = load_docs()
    pmid = {}
    doi_by_doc = {}
    for _, r in docs.iterrows():
        d = r["document_chembl_id"]
        pmid[d] = as_pmid(r.get("pubmed_id"))
        doi_by_doc[d] = r.get("doi")

    df = df[df["standard_type"].isin(set(POTENCY) | CYTOTOX | RATIO)]
    df = df[df["canonical_smiles"].notna() & df["standard_value"].notna()]
    df = standardize(df)
    df["censored"] = df["standard_relation"].fillna("=") != "="
    df["vacv"] = df.apply(is_vacv, axis = 1)
    df["doc_key"] = [pmid.get(d) or d for d in df["document_chembl_id"]]
    merged = int((df["doc_key"] != df["document_chembl_id"]).sum())
    print("после очистки:", len(df), "| строк сведено к общему PMID:", merged)

    doi_by_key = {}
    for d, v in doi_by_doc.items():
        k = pmid.get(d) or d
        if isinstance(v, str) and v.strip():
            doi_by_key[k] = v

    pairs = from_pairs(df)
    rat = from_ratio(df)
    print("пар собрано:", len(pairs), "| готовых SI из базы:", len(rat))

    have = set(zip(pairs["inchikey"], pairs["doc_key"]))
    add = rat[[(a, b) not in have for a, b in zip(rat["inchikey"], rat["doc_key"])]]
    print("из них новых:", len(add))

    full = pd.concat([pairs, add], ignore_index = True, sort = False)
    full["doi"] = full["doc_key"].map(doi_by_key)
    full["document"] = full["doc_key"]

    for c in ("IC50", "CC50"):
        if c not in full.columns:
            full[c] = np.nan

    cols = ["smiles", "IC50", "CC50", "SI", "doi"]
    rest = [c for c in full.columns if c not in cols]
    full = full[cols + rest].sort_values("SI", ascending = False)

    usable = full[full["quality"].isin(["exact", "lower_bound", "reported"])]
    mols = per_molecule(usable)

    full.to_csv("vacv_external_dataset.csv", index = False)
    usable.to_csv("vacv_external_usable.csv", index = False)
    mols.to_csv("vacv_external_molecules.csv", index = False)

    print("\nвсего строк:", len(full), "| молекул:", full["skeleton"].nunique(), "| с DOI:", int(full["doi"].notna().sum()))
    print("\nкачество:")
    print(full["quality"].value_counts())
    print("\nгодные строк:", len(usable), "| уникальных молекул:", len(mols))
    print(mols[["SI", "n_studies", "spread"]].describe())
    print("\nсхлопнуто стереоформ в одну молекулу:", int((mols["n_forms"] > 1).sum()))
    print("молекулы, измеренные больше чем в одной статье:", int((mols["n_studies"] > 1).sum()))
    bad = mols[(mols["n_studies"] > 1) & (mols["spread"] > 10)]
    print("из них расходятся больше чем в 10 раз:", len(bad))
    print("\nтоп-10 по SI:")
    for _, r in mols.sort_values("SI", ascending = False).head(10).iterrows():
        print("  SI =", round(r["SI"], 1), "| статей", int(r["n_studies"]), "|", r["quality"], "|", str(r["smiles"])[:55])


if __name__ == "__main__":
    main()
