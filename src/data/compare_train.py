import os

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

# train_v2.csv, а не train.csv: в старом заглушка 10000 вместо нуля.
PATHS = ["train_v2.csv", "data/raw/train_v2.csv", "test.csv", "data/raw/test.csv"]
EXT = ["data/processed/vacv_external_merged.csv", "vacv_external_merged.csv",
       "data/processed/vacv_external_usable.csv", "vacv_external_usable.csv"]

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


def find(df, word):
    for c in df.columns:
        if word in c.lower():
            return c
    return None


def load(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    print("\n===", path, "| строк:", len(df))
    print("колонки:", list(df.columns))
    smi_col = find(df, "smiles")
    if smi_col is None:
        print("не нашёл колонку со SMILES")
        return None
    df["inchikey"] = [key_of(s) for s in df[smi_col]]
    df["skeleton"] = [k[:14] if k else None for k in df["inchikey"]]
    print("молекул:", df["inchikey"].nunique(), "| не распарсилось:", int(df["inchikey"].isna().sum()))
    si_col = find(df, "si")
    if si_col and pd.api.types.is_numeric_dtype(df[si_col]):
        print("распределение", si_col, ":")
        print(df[si_col].describe())
    return df


def main():
    ext_path = next((p for p in EXT if os.path.exists(p)), None)
    if ext_path is None:
        raise SystemExit("не нашёл vacv_external_merged.csv, "
                         "сначала python src/data/merge_external.py")
    print("внешний набор:", ext_path)
    ext = pd.read_csv(ext_path)
    print("внешних строк:", len(ext), "| молекул:", ext["skeleton"].nunique())

    for p in PATHS:
        df = load(p)
        if df is None:
            continue
        same = set(df["skeleton"].dropna()) & set(ext["skeleton"].dropna())
        print("пересечение с внешними данными:", len(same), "молекул")
        si_col = find(df, "si")
        if si_col and pd.api.types.is_numeric_dtype(df[si_col]):
            top = df[si_col].max()
            over = ext[ext["SI"] > top]
            print("внешних строк с SI выше максимума в этом файле:", len(over))


if __name__ == "__main__":
    main()
