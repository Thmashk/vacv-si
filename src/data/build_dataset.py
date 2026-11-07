"""Сборка итогового датасета из выгрузок коллекторов.

Вход:  data/external/*_long.csv — одна строка на одно измерение.
Выход: data/processed/vacv_external_dataset.csv — одна строка на пару
       «молекула и публикация», первыми идут обязательные по условиям
       колонки smiles, IC50, CC50, SI, doi.

Четыре шага:
  1. стандартизация структур — убрать соли, привести к каноническому виду,
     посчитать InChIKey для надёжного сравнения молекул;
  2. схлопывание повторов — одно значение на (молекула, публикация, тип);
  3. спаривание — к каждой противовирусной активности подобрать
     цитотоксичность из той же публикации;
  4. расчёт SI и пометка качества записи.

Запуск: python -m src.data.build_dataset
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import SaltRemover

from ..config import ROOT

RDLogger.DisableLog("rdApp.*")
_SALTS = SaltRemover.SaltRemover()

EXTERNAL = ROOT / "data" / "external"
OUT = ROOT / "data" / "processed" / "vacv_external_dataset.csv"

REQUIRED_FIRST = ["smiles", "IC50", "CC50", "SI", "doi"]


def standardize(smiles: str) -> tuple[str | None, str | None]:
    """Канонический SMILES и InChIKey крупнейшего фрагмента."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None, None
    mol = _SALTS.StripMol(mol, dontRemoveEverything=True)
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if not frags:
        return None, None
    mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol), Chem.MolToInchiKey(mol)
    except Exception:
        return None, None


def load_long() -> pd.DataFrame:
    files = sorted(EXTERNAL.glob("*_long.csv"))
    if not files:
        raise SystemExit("нет файлов data/external/*_long.csv — "
                         "сначала запустите коллекторы")
    parts = []
    for f in files:
        df = pd.read_csv(f)
        print(f"  {f.name}: {len(df)} строк")
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def add_structures(df: pd.DataFrame) -> pd.DataFrame:
    uniq = df["smiles"].dropna().unique()
    print(f"  уникальных SMILES: {len(uniq)}")
    table = {s: standardize(s) for s in uniq}
    df["smiles_std"] = df["smiles"].map(lambda s: table.get(s, (None, None))[0])
    df["inchikey"] = df["smiles"].map(lambda s: table.get(s, (None, None))[1])
    bad = df["smiles_std"].isna().sum()
    print(f"  не разобрано RDKit: {bad}")
    return df[df["smiles_std"].notna()].copy()


def collapse(df: pd.DataFrame) -> pd.DataFrame:
    """Одно значение на (молекула, публикация, тип измерения).

    Берём геометрическое среднее: концентрации распределены логнормально,
    поэтому усреднять нужно логарифмы, а не сами значения.
    """
    df = df[df["value_uM"].notna() & (df["value_uM"] > 0)].copy()
    df["log_uM"] = np.log10(df["value_uM"])
    df["censored"] = df["relation"].isin([">", "<", ">=", "<="])

    g = df.groupby(["inchikey", "document_id", "endpoint"], dropna=False)
    out = g.agg(
        smiles=("smiles_std", "first"),
        log_uM=("log_uM", "median"),
        n_measurements=("log_uM", "size"),
        log_spread=("log_uM", lambda s: float(s.max() - s.min())),
        censored=("censored", "all"),
        standard_type=("standard_type", "first"),
        organism=("organism", "first"),
        assay_id=("assay_id", "first"),
        doi=("doi", "first"),
        pubmed_id=("pubmed_id", "first"),
        source=("source", "first"),
    ).reset_index()
    out["value_uM"] = 10 ** out["log_uM"]
    print(f"  после схлопывания повторов: {len(out)}")
    return out


def pair(df: pd.DataFrame) -> pd.DataFrame:
    """К каждой противовирусной активности ищем цитотоксичность.

    Сначала в той же публикации — это самый надёжный вариант, потому что
    измерения делались одной группой в одних условиях. Если в публикации
    цитотоксичности нет, берём медиану по всем публикациям для этой
    молекулы и помечаем запись как менее надёжную.
    """
    pot = df[df.endpoint == "potency"].copy()
    tox = df[df.endpoint == "cytotoxicity"].copy()
    print(f"  активностей: {len(pot)}, цитотоксичностей: {len(tox)}")

    tox_doc = tox.set_index(["inchikey", "document_id"])
    tox_any = tox.groupby("inchikey").agg(
        value_uM=("value_uM", "median"),
        censored=("censored", "all"),
        n_measurements=("n_measurements", "sum"),
    )

    rows = []
    for r in pot.itertuples(index=False):
        key = (r.inchikey, r.document_id)
        if key in tox_doc.index:
            t = tox_doc.loc[key]
            if isinstance(t, pd.DataFrame):
                t = t.iloc[0]
            cc50, quality = float(t.value_uM), "same_document"
            cc_censored, cc_n = bool(t.censored), int(t.n_measurements)
        elif r.inchikey in tox_any.index:
            t = tox_any.loc[r.inchikey]
            cc50, quality = float(t.value_uM), "other_document"
            cc_censored, cc_n = bool(t.censored), int(t.n_measurements)
        else:
            continue

        ic50 = float(r.value_uM)
        rows.append({
            "smiles": r.smiles,
            "IC50": round(ic50, 6),
            "CC50": round(cc50, 6),
            "SI": round(cc50 / ic50, 6),
            "doi": r.doi,
            "inchikey": r.inchikey,
            "source": r.source,
            "ic50_type": r.standard_type,
            "organism": r.organism,
            "document_id": r.document_id,
            "pubmed_id": r.pubmed_id,
            "assay_id": r.assay_id,
            "pairing_quality": quality,
            "ic50_censored": bool(r.censored),
            "cc50_censored": cc_censored,
            "n_ic50": int(r.n_measurements),
            "n_cc50": cc_n,
            "ic50_log_spread": round(float(r.log_spread), 3),
            # Если обе величины цензурированы («>100» и «>100»), их отношение
            # вырождается в единицу и ничего не значит. Такие строки нельзя
            # учить как настоящие измерения.
            "si_reliable": not (bool(r.censored) and cc_censored),
        })

    out = pd.DataFrame(rows)
    print(f"  получилось пар: {len(out)}")
    return out


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Одна строка на молекулу: берём самую надёжную пару."""
    if df.empty:
        return df
    order = {"same_document": 0, "other_document": 1}
    df["_rank"] = df["pairing_quality"].map(order)
    df["_unreliable"] = ~df["si_reliable"]
    df["_censored"] = df["ic50_censored"] | df["cc50_censored"]
    df = df.sort_values(["inchikey", "_unreliable", "_rank", "_censored", "n_ic50"],
                        ascending=[True, True, True, True, False])
    df = df.drop_duplicates("inchikey", keep="first").drop(
        columns=["_rank", "_censored", "_unreliable"])
    cols = REQUIRED_FIRST + [c for c in df.columns if c not in REQUIRED_FIRST]
    return df[cols].sort_values("SI", ascending=False).reset_index(drop=True)


def main() -> None:
    print("1. читаю выгрузки коллекторов")
    long = load_long()
    print(f"  всего измерений: {len(long)}")

    print("2. стандартизую структуры")
    long = add_structures(long)

    print("3. схлопываю повторы")
    collapsed = collapse(long)

    print("4. спариваю активность и цитотоксичность")
    paired = pair(collapsed)

    print("5. оставляю по одной строке на молекулу")
    final = finalize(paired)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(OUT, index=False)

    print(f"\nитог: {len(final)} молекул, записано в {OUT}")
    if len(final):
        rel = final[final.si_reliable]
        print(f"  пара из одной публикации: "
              f"{(final.pairing_quality == 'same_document').sum()}")
        print(f"  с DOI: {final.doi.notna().sum()}")
        print(f"  надёжный SI (не обе величины цензурированы): {len(rel)}")
        print(f"  медиана SI по надёжным: {rel.SI.median():.2f}, "
              f"максимум: {final.SI.max():.1f}")
        print(f"  источники: {dict(final.source.value_counts())}")


if __name__ == "__main__":
    main()
