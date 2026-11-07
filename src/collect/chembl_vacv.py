"""Сбор данных по вирусу осповакцины из ChEMBL.

Источник: https://www.ebi.ac.uk/chembl/api/data (ChEMBL REST API).
Лицензия данных: CC BY-SA 3.0.

Логика сбора. Противовирусную активность (IC50, EC50) измеряют в ассеях,
у которых организм — вирус осповакцины. Цитотоксичность (CC50) измеряют на
клетках хозяина, поэтому её ассеи помечены другим организмом и по вирусу
не находятся. Зато обе величины почти всегда опубликованы в одной статье.

Поэтому идём в три шага:
  1. находим ассеи по вирусу (assay_organism содержит vaccinia);
  2. берём все публикации, в которых эти ассеи встречаются;
  3. выгружаем из этих публикаций все измерения нужных типов, включая
     цитотоксичность на клетках.

Запуск:  python -m src.collect.chembl_vacv
Результат: data/external/chembl_long.csv — одна строка на одно измерение.
"""
from __future__ import annotations

import argparse

import pandas as pd

from ..config import ROOT
from .common import (
    LONG_COLUMNS,
    build_url,
    classify_endpoint,
    fetch_json,
    mol_weight,
    to_micromolar,
)

BASE = "https://www.ebi.ac.uk/chembl/api/data"
OUT = ROOT / "data" / "external" / "chembl_long.csv"

ORGANISM_QUERY = "vaccinia"
WANTED_TYPES = "IC50,EC50,CC50,TC50,CD50,GI50,ID50"
PAGE = 200


def _paginate(path: str, key: str, params: dict, use_cache: bool = True) -> list[dict]:
    """Проходит все страницы ответа ChEMBL."""
    rows, offset = [], 0
    while True:
        url = build_url(BASE, path, {**params, "format": "json",
                                     "limit": PAGE, "offset": offset})
        data = fetch_json(url, use_cache=use_cache)
        if not data:
            break
        rows += data.get(key, [])
        total = data.get("page_meta", {}).get("total_count", 0)
        offset += PAGE
        if offset >= total:
            break
    return rows


def fetch_virus_assays(use_cache: bool = True) -> list[dict]:
    return _paginate("assay", "assays",
                     {"assay_organism__icontains": ORGANISM_QUERY}, use_cache)


def fetch_activities(document_ids: list[str], use_cache: bool = True) -> list[dict]:
    acts = []
    for i in range(0, len(document_ids), 25):
        chunk = ",".join(document_ids[i:i + 25])
        acts += _paginate("activity", "activities",
                          {"document_chembl_id__in": chunk,
                           "standard_type__in": WANTED_TYPES}, use_cache)
        print(f"    публикации {i + 1}-{min(i + 25, len(document_ids))}: "
              f"всего измерений {len(acts)}", end="\r")
    print()
    return acts


def fetch_dois(document_ids: list[str], use_cache: bool = True) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for i in range(0, len(document_ids), 25):
        chunk = ",".join(document_ids[i:i + 25])
        for d in _paginate("document", "documents",
                           {"document_chembl_id__in": chunk}, use_cache):
            out[d["document_chembl_id"]] = {
                "doi": d.get("doi"),
                "pubmed_id": d.get("pubmed_id"),
            }
    return out


def collect(use_cache: bool = True) -> pd.DataFrame:
    print("ChEMBL: ищу ассеи по вирусу осповакцины")
    assays = fetch_virus_assays(use_cache)
    virus_assay_ids = {a["assay_chembl_id"] for a in assays}
    documents = sorted({a["document_chembl_id"] for a in assays
                        if a.get("document_chembl_id")})
    print(f"  ассеев по вирусу: {len(virus_assay_ids)}, "
          f"публикаций: {len(documents)}")

    print("ChEMBL: выгружаю измерения из этих публикаций")
    activities = fetch_activities(documents, use_cache)
    print(f"  всего измерений нужных типов: {len(activities)}")

    print("ChEMBL: подтягиваю DOI публикаций")
    docs = fetch_dois(documents, use_cache)
    print(f"  публикаций с DOI: {sum(1 for v in docs.values() if v['doi'])}")

    mw_cache: dict[str, float | None] = {}
    rows = []
    for a in activities:
        smiles = a.get("canonical_smiles")
        if not smiles:
            continue
        endpoint = classify_endpoint(
            a.get("standard_type"), a.get("assay_description"),
            a.get("assay_chembl_id") in virus_assay_ids,
        )
        if endpoint is None:
            continue

        units = a.get("standard_units")
        mw = None
        if units and "g." in str(units).lower():
            if smiles not in mw_cache:
                mw_cache[smiles] = mol_weight(smiles)
            mw = mw_cache[smiles]

        doc_id = a.get("document_chembl_id")
        meta = docs.get(doc_id, {})
        rows.append({
            "source": "ChEMBL",
            "record_id": a.get("activity_id"),
            "molecule_id": a.get("molecule_chembl_id"),
            "smiles": smiles,
            "standard_type": a.get("standard_type"),
            "relation": a.get("standard_relation"),
            "value": a.get("standard_value"),
            "units": units,
            "value_uM": to_micromolar(a.get("standard_value"), units, mw),
            "endpoint": endpoint,
            "organism": a.get("assay_organism") or a.get("target_organism"),
            "assay_id": a.get("assay_chembl_id"),
            "assay_description": (a.get("assay_description") or "")[:300],
            "document_id": doc_id,
            "doi": meta.get("doi"),
            "pubmed_id": meta.get("pubmed_id"),
        })

    df = pd.DataFrame(rows, columns=LONG_COLUMNS)
    print(f"  строк в длинном формате: {len(df)}")
    print(f"    противовирусная активность: {(df.endpoint == 'potency').sum()}")
    print(f"    цитотоксичность: {(df.endpoint == 'cytotoxicity').sum()}")
    print(f"    с пересчётом в мкМ: {df.value_uM.notna().sum()}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cache", action="store_true",
                    help="игнорировать кэш и перекачать всё заново")
    args = ap.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = collect(use_cache=not args.no_cache)
    df.to_csv(OUT, index=False)
    print(f"записано: {OUT}")


if __name__ == "__main__":
    main()
