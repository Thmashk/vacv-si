import csv
import os
import random
import re
import sys

import pandas as pd
import requests
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

DATASET = "vacv_external_dataset.csv"
MOLECULES = "vacv_external_molecules.csv"
REQUIRED = ["smiles", "IC50", "CC50", "SI", "doi"]
DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")

SOURCES = {
    "ChEMBL REST API": "https://www.ebi.ac.uk/chembl/api/data/status.json",
    "PubChem PUG-REST": "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2244/property/MolecularFormula/JSON",
    "NCBI E-utilities": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/einfo.fcgi?db=pcassay&retmode=json",
}

errors = []
warnings = []
lines = []


def say(s = ""):
    print(s)
    lines.append(s)


def hard(s):
    errors.append(s)
    say("  ОШИБКА: " + s)


def soft(s):
    warnings.append(s)
    say("  предупреждение: " + s)


def check_schema(df):
    say("## Схема")
    if list(df.columns[:5]) != REQUIRED:
        hard("первые пять колонок должны быть " + ", ".join(REQUIRED) + ", а сейчас " + ", ".join(df.columns[:5]))
    else:
        say("  пять обязательных колонок идут первыми: ок")
    for c in ("smiles", "SI"):
        n = int(df[c].isna().sum())
        if n:
            hard("пустых значений в " + c + ": " + str(n))
    say("  строк: " + str(len(df)))


def check_values(df):
    say("\n## Значения")
    for c in ("IC50", "CC50", "SI"):
        v = pd.to_numeric(df[c], errors = "coerce")
        bad = int((v <= 0).sum())
        if bad:
            hard("неположительных значений в " + c + ": " + str(bad))
    both = df[df["IC50"].notna() & df["CC50"].notna()].copy()
    calc = both["CC50"] / both["IC50"]
    off = (calc - both["SI"]).abs() / both["SI"]
    n = int((off > 0.01).sum())
    if n:
        hard("SI не сходится с CC50 / IC50 более чем на 1% в " + str(n) + " строках")
    else:
        say("  SI сходится с CC50 / IC50 во всех " + str(len(both)) + " строках с обеими величинами")


def check_doi(df):
    say("\n## DOI")
    has = df["doi"].notna()
    say("  строк с DOI: " + str(int(has.sum())) + " из " + str(len(df)))
    bad = [d for d in df.loc[has, "doi"] if not DOI_RE.match(str(d))]
    if bad:
        soft("не прошли формат DOI: " + str(len(bad)) + ", например " + str(bad[0]))
    else:
        say("  формат DOI: все проходят")
    if int(has.sum()) < len(df) * 0.5:
        soft("меньше половины строк имеют DOI")


def check_smiles(df):
    say("\n## SMILES")
    bad = 0
    for s in df["smiles"]:
        if Chem.MolFromSmiles(str(s)) is None:
            bad += 1
    if bad:
        hard("не разбирается RDKit: " + str(bad) + " SMILES")
    else:
        say("  все " + str(len(df)) + " SMILES разбираются RDKit")


def check_duplicates(df):
    say("\n## Дубликаты")
    if "skeleton" not in df.columns:
        soft("нет колонки skeleton, проверка дубликатов пропущена")
        return
    g = df.groupby("skeleton")["SI"].agg(["size", "min", "max"])
    multi = g[g["size"] > 1]
    say("  молекул более чем с одной записью: " + str(len(multi)))
    if len(multi):
        spread = multi["max"] / multi["min"].replace(0, pd.NA)
        wide = int((spread > 10).sum())
        say("  из них расходятся больше чем в 10 раз: " + str(wide))
        if wide > len(multi) * 0.5:
            soft("больше половины повторных измерений расходятся более чем в 10 раз")


def check_sources():
    say("\n## Доступность источников")
    for name, url in SOURCES.items():
        try:
            r = requests.get(url, timeout = 30)
            ok = r.status_code == 200
        except Exception as e:
            ok = False
            r = None
        say("  " + name + " -> " + url)
        if ok:
            say("     доступен, код 200")
        else:
            soft(name + " недоступен" + ("" if r is None else " (код " + str(r.status_code) + ")"))


def spot_check(n = 20):
    say("\n## Сверка с живым API")
    if not os.path.exists("chembl_vacv_raw.csv"):
        soft("нет chembl_vacv_raw.csv, сверка пропущена")
        return
    rows = []
    with open("chembl_vacv_raw.csv", encoding = "utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("activity_id") and not r["activity_id"].startswith("PC-") and (r.get("standard_value") or "").strip():
                rows.append(r)
    if not rows:
        soft("нет строк ChEMBL для сверки")
        return
    random.seed(0)
    sample = random.sample(rows, min(n, len(rows)))
    same = 0
    checked = 0
    for r in sample:
        url = "https://www.ebi.ac.uk/chembl/api/data/activity/" + str(r["activity_id"]) + ".json"
        try:
            resp = requests.get(url, timeout = 30)
        except Exception:
            continue
        if resp.status_code != 200:
            continue
        checked += 1
        live = resp.json().get("standard_value")
        mine = r.get("standard_value")
        empty = not str(live or "").strip() and not str(mine or "").strip()
        if empty:
            same += 1
            continue
        try:
            if abs(float(live) - float(mine)) < 1e-6:
                same += 1
            else:
                say("  расхождение: activity_id " + str(r["activity_id"]) + ", у нас " + str(mine) + ", в API " + str(live))
        except Exception:
            say("  не сравнить: activity_id " + str(r["activity_id"]) + ", у нас " + str(mine) + ", в API " + str(live))
    say("  перезапрошено строк: " + str(checked) + ", значение совпало: " + str(same))
    if checked and same < checked:
        soft("часть значений разошлась с API, возможно база обновилась")


def main():
    if not os.path.exists(DATASET):
        print("нет файла", DATASET)
        sys.exit(1)
    df = pd.read_csv(DATASET)

    say("# Проверка датасета")
    say()
    check_schema(df)
    check_values(df)
    check_doi(df)
    check_smiles(df)
    check_duplicates(df)
    check_sources()
    spot_check()

    say("\n## Итог")
    say("  ошибок: " + str(len(errors)) + ", предупреждений: " + str(len(warnings)))

    with open("reports/dataset_validation_r2.md", "w", encoding = "utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\nотчёт записан в reports/dataset_validation_r2.md")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
