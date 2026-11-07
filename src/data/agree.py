import os

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

chooser = rdMolStandardize.LargestFragmentChooser()
uncharger = rdMolStandardize.Uncharger()


def key_of(smi):
    if not isinstance(smi, str) or not smi:
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    m = uncharger.uncharge(chooser.choose(m))
    try:
        return Chem.MolToInchiKey(m)
    except Exception:
        return None


def path_of(name):
    for p in (name, "data/raw/" + name, "data/processed/" + name):
        if os.path.exists(p):
            return p
    return None


def main():
    train_path = path_of("train_v2.csv")
    if train_path is None:
        raise SystemExit(
            "не нашёл train_v2.csv ни рядом, ни в data/raw/.\n"
            "Это текущий файл организаторов; в старом train.csv заглушка\n"
            "равна 10000 вместо нуля, и сверка по нему недействительна.")
    tr = pd.read_csv(train_path)
    tr["inchikey"] = [key_of(s) for s in tr["SMILES"]]
    tr["skeleton"] = [k[:14] if k else None for k in tr["inchikey"]]

    print("=== сходится ли SI с делением CC50 / IC50 ===")
    calc = tr["CC50_uM"] / tr["IC50_uM_VACV"]
    d = (calc - tr["SI VACV"]).abs() / tr["SI VACV"].replace(0, np.nan)
    print("совпадает в пределах 1%:", int((d < 0.01).sum()), "из", int(d.notna().sum()))
    print("пустых CC50:", int(tr["CC50_uM"].isna().sum()), "| пустых IC50:", int(tr["IC50_uM_VACV"].isna().sum()))

    print("\n=== самые частые значения SI в трейне ===")
    print(tr["SI VACV"].value_counts().head(10))

    print("\n=== strain и Culture ===")
    print(tr["strain"].value_counts().head(5))
    print(tr["Culture"].value_counts().head(5))

    te_path = path_of("test.csv")
    if te_path:
        te = pd.read_csv(te_path)
        print("\n=== запрещённые колонки в test ===")
        print("непустых CC50_uM:", int(te["CC50_uM"].notna().sum()), "из", len(te))
        print("непустых IC50_uM_VACV:", int(te["IC50_uM_VACV"].notna().sum()), "из", len(te))

    ext_path = (path_of("vacv_external_merged.csv")
                or path_of("vacv_external_usable.csv"))
    if ext_path is None:
        raise SystemExit("не нашёл vacv_external_merged.csv, "
                         "сначала python src/data/merge_external.py")
    ext = pd.read_csv(ext_path)
    m = tr.merge(ext, on = "skeleton", how = "inner", suffixes = ("_k", "_e"))
    print("\n=== общие молекулы:", len(m), "===")
    if not len(m):
        return

    m["ratio"] = m["SI"] / m["SI VACV"]
    print("корреляция Спирмена:", round(m["SI VACV"].corr(m["SI"], method = "spearman"), 3))
    print("медиана отношения наш / их:", round(m["ratio"].median(), 3))
    print("в пределах 1.5 раза:", int(((m["ratio"] > 1 / 1.5) & (m["ratio"] < 1.5)).sum()), "из", len(m))

    # Самые сильные расхождения полезнее первых двадцати строк: именно по ним
    # видно, где наш сбор разошёлся с организаторами.
    print("\nсамые сильные расхождения:")
    m["_off"] = np.log10(m["ratio"].where(m["ratio"] > 0)).abs()
    cols = [c for c in ("SI VACV", "SI", "ratio", "quality", "origin", "doi")
            if c in m.columns]
    worst = m.sort_values("_off", ascending = False)[cols].head(10)
    for _, r in worst.iterrows():
        extra = []
        for c in ("quality", "origin"):
            if c in r.index:
                extra.append(str(r[c]))
        print("  их", round(r["SI VACV"], 2),
              "| наш", round(r["SI"], 2),
              "| x", round(r["ratio"], 2),
              ("| " + ", ".join(extra)) if extra else "")


if __name__ == "__main__":
    main()
