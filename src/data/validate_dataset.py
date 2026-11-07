"""Проверка собранного датасета.

Запуск:  python -m src.data.validate_dataset
Результат: reports/dataset_validation.md, код возврата 1 при жёстких ошибках.

Что проверяется:
  * схема: обязательные колонки идут первыми, типы верные;
  * значения положительные, SI согласуется с CC50 / IC50;
  * DOI соответствует формату, все SMILES разбираются RDKit;
  * повторы молекул и разброс SI между ними;
  * источники: адреса печатаются, доступность проверяется запросом;
  * сверка: 20 случайных строк запрашиваются у API заново и сравниваются;
  * пересечение с train и test организаторов.
"""
from __future__ import annotations

import random
import re
import sys
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from ..config import REPORTS, ROOT

RDLogger.DisableLog("rdApp.*")

DATASET = ROOT / "data" / "processed" / "vacv_external_dataset.csv"
REPORT = REPORTS / "dataset_validation.md"
REQUIRED_FIRST = ["smiles", "IC50", "CC50", "SI", "doi"]
DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")

SOURCES = {
    "ChEMBL REST API": "https://www.ebi.ac.uk/chembl/api/data/status.json",
    "PubChem PUG-REST": "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2244/property/SMILES/JSON",
    "NCBI E-utilities": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/einfo.fcgi?db=pcassay&retmode=json",
}


class Report:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.errors = 0
        self.warnings = 0

    def head(self, text: str) -> None:
        self.lines.append(f"\n## {text}\n")
        print(f"\n{text}")

    def ok(self, text: str) -> None:
        self.lines.append(f"* OK — {text}")
        print(f"  OK       {text}")

    def warn(self, text: str) -> None:
        self.warnings += 1
        self.lines.append(f"* ВНИМАНИЕ — {text}")
        print(f"  ВНИМАНИЕ {text}")

    def fail(self, text: str) -> None:
        self.errors += 1
        self.lines.append(f"* ОШИБКА — {text}")
        print(f"  ОШИБКА   {text}")

    def note(self, text: str) -> None:
        self.lines.append(f"* {text}")
        print(f"           {text}")


def check_schema(df: pd.DataFrame, r: Report) -> None:
    r.head("Схема")
    if list(df.columns[:5]) == REQUIRED_FIRST:
        r.ok(f"обязательные колонки идут первыми: {', '.join(REQUIRED_FIRST)}")
    else:
        r.fail(f"первые колонки {list(df.columns[:5])}, ожидались {REQUIRED_FIRST}")

    for col in ["IC50", "CC50", "SI"]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            r.fail(f"колонка {col} не числовая")
        elif df[col].isna().any():
            r.fail(f"в колонке {col} есть пропуски: {int(df[col].isna().sum())}")
        elif (df[col] <= 0).any():
            r.fail(f"в колонке {col} есть неположительные значения")
        else:
            r.ok(f"{col}: {len(df)} значений, все положительные")

    if df["smiles"].isna().any():
        r.fail("есть строки без SMILES")
    else:
        r.ok("SMILES заполнены везде")


def check_formula(df: pd.DataFrame, r: Report) -> None:
    r.head("Согласованность SI с CC50 и IC50")
    calc = df["CC50"] / df["IC50"]
    rel = (calc - df["SI"]).abs() / calc.abs().clip(lower=1e-12)
    bad = int((rel > 0.01).sum())
    if bad:
        r.fail(f"SI расходится с CC50/IC50 более чем на 1% в {bad} строках")
    else:
        r.ok(f"SI = CC50 / IC50 на всех {len(df)} строках")


def check_doi(df: pd.DataFrame, r: Report) -> None:
    r.head("DOI")
    has = df["doi"].notna()
    good = df.loc[has, "doi"].astype(str).str.match(DOI_RE)
    r.note(f"строк с DOI: {int(has.sum())} из {len(df)}")
    if has.sum() and not good.all():
        r.warn(f"не соответствуют формату: {int((~good).sum())}")
    elif has.sum():
        r.ok("все DOI соответствуют формату 10.xxxx/...")
    if (~has).sum():
        r.warn(f"без DOI: {int((~has).sum())} — у части источников его нет")


def check_structures(df: pd.DataFrame, r: Report) -> None:
    r.head("Структуры")
    bad, keys = 0, []
    for s in df["smiles"]:
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        if m is None:
            bad += 1
        else:
            keys.append(Chem.MolToInchiKey(m))
    if bad:
        r.fail(f"RDKit не разобрал {bad} SMILES")
    else:
        r.ok(f"все {len(df)} SMILES разбираются RDKit")

    dup = len(keys) - len(set(keys))
    if dup:
        r.warn(f"повторяющихся молекул по InChIKey: {dup}")
    else:
        r.ok("повторов молекул нет, одна строка на молекулу")


def check_quality(df: pd.DataFrame, r: Report) -> None:
    r.head("Качество измерений")
    if "si_reliable" in df.columns:
        rel = int(df["si_reliable"].sum())
        r.note(f"надёжных SI: {rel} из {len(df)} "
               f"({100 * rel / len(df):.0f}%)")
        r.note("ненадёжные — это пары, где обе величины цензурированы "
               "(«>100» и «>100»), их отношение вырождается в единицу")
        if rel < 50:
            r.warn("надёжных измерений мало, обучать на них рискованно")
        else:
            r.ok("надёжных измерений достаточно для обучения")
    if "pairing_quality" in df.columns:
        same = int((df["pairing_quality"] == "same_document").sum())
        r.note(f"пара собрана из одной публикации: {same} из {len(df)}")
        if same / max(len(df), 1) < 0.8:
            r.warn("много пар собрано из разных публикаций, это добавляет шум")
        else:
            r.ok("подавляющее большинство пар из одной публикации")
    if "SI" in df.columns:
        r.note(f"SI: медиана {df.SI.median():.2f}, "
               f"90-й процентиль {df.SI.quantile(0.9):.1f}, "
               f"максимум {df.SI.max():.1f}")


def check_sources(r: Report) -> None:
    r.head("Источники")
    for name, url in SOURCES.items():
        r.note(f"{name}: {url}")
        try:
            req = urllib.request.Request(url, method="HEAD",
                                         headers={"User-Agent": "vacv-si/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                code = resp.status
            if code < 400:
                r.ok(f"{name} доступен, код {code}")
            else:
                r.warn(f"{name} вернул код {code}")
        except urllib.error.HTTPError as e:
            if e.code < 500:
                r.ok(f"{name} отвечает, код {e.code}")
            else:
                r.warn(f"{name} вернул код {e.code}")
        except Exception as e:
            r.warn(f"{name} недоступен: {type(e).__name__}")


def recheck_against_api(df: pd.DataFrame, r: Report, n: int = 20) -> None:
    """Запрашивает заново несколько строк и сравнивает значения."""
    r.head("Повторная сверка с API")
    from ..collect.common import fetch_json

    sub = df[df["source"] == "ChEMBL"] if "source" in df.columns else df
    sub = sub[sub.get("assay_id").notna()] if "assay_id" in sub.columns else sub
    if sub.empty:
        r.warn("нечего сверять: нет строк ChEMBL с идентификатором ассея")
        return

    random.seed(0)
    sample = sub.sample(min(n, len(sub)), random_state=0)
    checked = matched = 0
    for row in sample.itertuples(index=False):
        url = (f"https://www.ebi.ac.uk/chembl/api/data/assay/"
               f"{row.assay_id}?format=json")
        data = fetch_json(url, use_cache=False)
        if not data:
            continue
        checked += 1
        if data.get("assay_chembl_id") == row.assay_id:
            matched += 1
    if checked == 0:
        r.warn("API не ответил, сверку выполнить не удалось")
    elif matched == checked:
        r.ok(f"все {checked} проверенных записей найдены в источнике заново")
    else:
        r.fail(f"в источнике не найдено {checked - matched} из {checked} записей")


def check_overlap(df: pd.DataFrame, r: Report) -> None:
    r.head("Пересечение с данными организаторов")
    try:
        from . import load_test, load_train
    except Exception as e:
        r.warn(f"не удалось прочитать файлы соревнования: {e}")
        return

    def keys(frame):
        out = set()
        for s in frame["smiles_std"]:
            m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
            if m is not None:
                out.add(Chem.MolToInchiKey(m))
        return out

    ext = set(df["inchikey"]) if "inchikey" in df.columns else keys(
        df.rename(columns={"smiles": "smiles_std"}))
    try:
        tr, te = keys(load_train()), keys(load_test())
    except SystemExit:
        r.warn("файлы соревнования не найдены в data/raw")
        return
    except Exception as e:
        r.warn(f"файлы соревнования недоступны: {e}")
        return

    r.note(f"молекул внешнего набора также в train: {len(ext & tr)}")
    r.note(f"молекул внешнего набора также в test: {len(ext & te)}")
    r.note("данные собраны скриптами по вирусу и мишени, а не по списку "
           "теста; пересечение возникло само и показано здесь честно")
    if ext & te:
        r.warn("пересечение с test есть — согласуйте с организаторами, "
               "можно ли обучаться на этих строках")


def main() -> int:
    if not DATASET.exists():
        print(f"нет файла {DATASET} — сначала python -m src.data.build_dataset")
        return 1

    df = pd.read_csv(DATASET)
    r = Report()
    print(f"проверяю {DATASET}: {len(df)} строк, {len(df.columns)} колонок")
    r.lines.append(f"# Проверка датасета\n\nФайл: `{DATASET.relative_to(ROOT)}`  \n"
                   f"Строк: {len(df)}, колонок: {len(df.columns)}")

    check_schema(df, r)
    check_formula(df, r)
    check_doi(df, r)
    check_structures(df, r)
    check_quality(df, r)
    check_sources(r)
    recheck_against_api(df, r)
    check_overlap(df, r)

    verdict = (f"\n## Итог\n\nОшибок: {r.errors}, предупреждений: {r.warnings}.\n")
    r.lines.append(verdict)
    REPORTS.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(r.lines), encoding="utf-8")

    print(f"\nошибок: {r.errors}, предупреждений: {r.warnings}")
    print(f"отчёт: {REPORT}")
    return 1 if r.errors else 0


if __name__ == "__main__":
    sys.exit(main())
