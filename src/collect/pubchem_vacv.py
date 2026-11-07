"""Сбор данных по вирусу осповакцины из PubChem BioAssay.

Источники:
  https://pubchem.ncbi.nlm.nih.gov/rest/pug      (PUG-REST)
  https://eutils.ncbi.nlm.nih.gov/entrez/eutils  (E-utilities, поиск ассеев)

Условия использования: данные PubChem открыты, но права на каждый набор
задаёт депозитор. Ограничение скорости — не более 5 запросов в секунду.

Логика та же, что в коллекторе ChEMBL: ищем ассеи по вирусу, выгружаем
измерения, отделяем противовирусную активность от цитотоксичности по
названию измерения и описанию ассея.

Запуск:  python -m src.collect.pubchem_vacv
Результат: data/external/pubchem_long.csv
"""
from __future__ import annotations

import argparse

import pandas as pd

from ..config import ROOT
from .common import (
    LONG_COLUMNS,
    classify_endpoint,
    fetch_json,
    mol_weight,
    to_micromolar,
)

PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
OUT = ROOT / "data" / "external" / "pubchem_long.csv"

SEARCH_TERMS = ["vaccinia virus", "orthopoxvirus"]
WANTED_NAMES = ("ic50", "ec50", "cc50", "tc50", "cd50", "gi50", "id50")


def find_assay_ids(use_cache: bool = True, retmax: int = 400) -> list[int]:
    aids: list[int] = []
    for term in SEARCH_TERMS:
        url = (f"{EUTILS}/esearch.fcgi?db=pcassay&retmode=json"
               f"&retmax={retmax}&term={term.replace(' ', '+')}")
        data = fetch_json(url, use_cache=use_cache)
        if not data:
            continue
        ids = data.get("esearchresult", {}).get("idlist", [])
        aids += [int(x) for x in ids]
        print(f"  запрос {term!r}: найдено {len(ids)} ассеев")
    return sorted(set(aids))


def assay_summary(aid: int, use_cache: bool = True) -> dict:
    data = fetch_json(f"{PUG}/assay/aid/{aid}/summary/JSON", use_cache=use_cache)
    if not data:
        return {}
    try:
        s = data["AssaySummaries"]["AssaySummary"][0]
    except (KeyError, IndexError):
        return {}
    refs = s.get("PubMedID") or []
    return {
        "name": s.get("Name", ""),
        "description": " ".join(s.get("Description", []))[:300],
        "pubmed_id": refs[0] if refs else None,
    }


def assay_rows(aid: int, use_cache: bool = True) -> list[dict]:
    """Концентрационные данные ассея в виде списка словарей."""
    data = fetch_json(f"{PUG}/assay/aid/{aid}/concise/JSON", use_cache=use_cache)
    if not data:
        return []
    try:
        table = data["Table"]
        cols = table["Columns"]["Column"]
        return [dict(zip(cols, r["Cell"])) for r in table["Row"]]
    except (KeyError, TypeError):
        return []


def smiles_for_cids(cids: list[int], use_cache: bool = True) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for i in range(0, len(cids), 100):
        chunk = ",".join(str(c) for c in cids[i:i + 100])
        url = (f"{PUG}/compound/cid/{chunk}/property/"
               f"SMILES,MolecularWeight/JSON")
        data = fetch_json(url, use_cache=use_cache)
        if not data:
            continue
        for p in data.get("PropertyTable", {}).get("Properties", []):
            smi = p.get("SMILES") or p.get("CanonicalSMILES")
            if not smi:
                continue
            try:
                mw = float(p.get("MolecularWeight"))
            except (TypeError, ValueError):
                mw = None
            out[int(p["CID"])] = {"smiles": smi, "mw": mw}
        print(f"    структуры: {len(out)}/{len(cids)}", end="\r")
    print()
    return out


def collect(use_cache: bool = True, max_assays: int = 250) -> pd.DataFrame:
    print("PubChem: ищу ассеи по вирусу")
    aids = find_assay_ids(use_cache)
    print(f"  всего уникальных ассеев: {len(aids)}")
    if len(aids) > max_assays:
        aids = aids[:max_assays]
        print(f"  беру первые {max_assays}")

    raw, cids = [], set()
    for n, aid in enumerate(aids, 1):
        meta = assay_summary(aid, use_cache)
        text = f"{meta.get('name', '')} {meta.get('description', '')}".lower()
        if "vaccinia" not in text and "orthopox" not in text and "poxvirus" not in text:
            continue
        for row in assay_rows(aid, use_cache):
            name = str(row.get("Activity Name", "")).lower()
            if not any(w in name for w in WANTED_NAMES):
                continue
            cid = row.get("CID")
            if not cid:
                continue
            cids.add(int(cid))
            raw.append({"aid": aid, "meta": meta, "row": row})
        print(f"  ассеи: {n}/{len(aids)}, подходящих измерений {len(raw)}", end="\r")
    print()

    if not raw:
        print("  подходящих измерений не найдено")
        return pd.DataFrame(columns=LONG_COLUMNS)

    print("PubChem: подтягиваю структуры")
    structures = smiles_for_cids(sorted(cids), use_cache)

    rows = []
    for item in raw:
        row, meta, aid = item["row"], item["meta"], item["aid"]
        cid = int(row["CID"])
        s = structures.get(cid)
        if not s:
            continue
        name = str(row.get("Activity Name", ""))
        # PubChem отдаёт концентрации в микромолях, если не сказано иное.
        units = row.get("Activity Unit") or "uM"
        mw = s["mw"] or mol_weight(s["smiles"])
        is_virus = True  # ассей уже отфильтрован по упоминанию вируса
        endpoint = classify_endpoint(
            name.upper().replace("ACTIVITY", "").strip(),
            f"{meta.get('name','')} {meta.get('description','')} {name}",
            is_virus,
        )
        if endpoint is None:
            continue
        rows.append({
            "source": "PubChem",
            "record_id": f"AID{aid}-SID{row.get('SID')}",
            "molecule_id": f"CID{cid}",
            "smiles": s["smiles"],
            "standard_type": name.upper(),
            "relation": row.get("Activity Qualifier"),
            "value": row.get("Activity Value [uM]") or row.get("Activity Value"),
            "units": units,
            "value_uM": to_micromolar(
                row.get("Activity Value [uM]") or row.get("Activity Value"),
                units, mw),
            "endpoint": endpoint,
            "organism": "Vaccinia virus",
            "assay_id": f"AID{aid}",
            "assay_description": meta.get("name", "")[:300],
            "document_id": f"AID{aid}",
            "doi": None,
            "pubmed_id": meta.get("pubmed_id"),
        })

    df = pd.DataFrame(rows, columns=LONG_COLUMNS)
    print(f"  строк в длинном формате: {len(df)}")
    if len(df):
        print(f"    противовирусная активность: {(df.endpoint == 'potency').sum()}")
        print(f"    цитотоксичность: {(df.endpoint == 'cytotoxicity').sum()}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--max-assays", type=int, default=250)
    args = ap.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = collect(use_cache=not args.no_cache, max_assays=args.max_assays)
    df.to_csv(OUT, index=False)
    print(f"записано: {OUT}")


if __name__ == "__main__":
    main()
