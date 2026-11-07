"""Объединение внешних данных R2 с набором из PR #1.

Зачем. Оба набора собраны из ChEMBL и PubChem, но по-разному. Набор R2
покрывает PubChem (618 строк против 16), набор PR #1 дедуплицирован по
полному InChIKey и сохраняет стереохимию. Пересечение по скелету InChIKey
составляет 478 молекул, то есть у каждой стороны есть своё.

Что делает скрипт.

1. Базой берёт набор R2: у него шире охват PubChem и есть класс `artifact`,
   который ловит записи с IC50 в точности равным CC50.
2. Из набора PR #1 добирает только те молекулы, которых в базе нет и у
   которых измерение действительно чистое: `si_reliable`, IC50 не равен
   CC50, ни одна из величин не усечена.
3. Приводит каждую строку к внутренней согласованности: SI считается как
   CC50 / IC50 из одной конкретной публикации, а не как медиана по
   нескольким. Разброс между публикациями сохраняется отдельными колонками.

Почему пункт 3 важен. В `vacv_external_molecules.csv` у молекул с
несколькими публикациями SI был медианой по публикациям, а IC50 и CC50
брались из одной. На 13 молекулах это расходилось больше чем на процент,
и `tests/test_dataset.py::test_si_equals_ratio` на таком файле падает.

Запуск из корня репозитория:

    python3 src/data/merge_external.py

Зависимостей нет, только стандартная библиотека: скрипт должен работать
и без собранного .venv.
"""
from __future__ import annotations

import csv
import os
import statistics
from collections import Counter

# Входы
PAIRS_R2 = "data/processed/vacv_external_pairs_r2.csv"
MOLS_R2 = "data/processed/vacv_external_molecules.csv"
DATASET_PR1 = "data/processed/vacv_external_dataset.csv"

# Выходы
OUT_MERGED = "data/processed/vacv_external_merged.csv"
OUT_REPORTED = "data/processed/vacv_external_reported_only.csv"
OUT_REPORT = "reports/r2_merge_report.md"

# Обязательные по условиям задачи колонки идут первыми.
REQUIRED_FIRST = ["smiles", "IC50", "CC50", "SI", "doi"]
COLUMNS = REQUIRED_FIRST + [
    "inchikey",
    "skeleton",
    "source",
    "origin",
    "quality",
    "si_reliable",
    "pairing_quality",
    "ic50_censored",
    "cc50_censored",
    "n_studies",
    "si_min",
    "si_max",
    "n_forms",
    "document",
]

# Для обучения годятся только эти классы: у них есть и IC50, и CC50.
USABLE_QUALITY = ("exact", "lower_bound")

notes: list[str] = []


def log(line: str = "") -> None:
    print(line)
    notes.append(line)


def read(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(row: dict, key: str) -> float | None:
    try:
        v = float(row.get(key, ""))
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def skel(inchikey: str | None) -> str:
    return (inchikey or "")[:14]


def is_true(value: str | None) -> bool:
    return str(value).strip().lower() == "true"


def pick_representative(cands: list[dict]) -> dict:
    """Одна публикация на молекулу, чтобы SI = CC50 / IC50 выполнялось точно.

    Сначала предпочитаем точные измерения усечённым, затем берём публикацию
    с SI ближе всего к медиане по молекуле — так представитель не оказывается
    случайным выбросом, — и при равенстве ту, где больше повторов.
    """
    exact = [c for c in cands if c["quality"] == "exact"]
    pool = exact or cands
    sis = [num(c, "SI") for c in pool]
    sis = [s for s in sis if s is not None]
    median = statistics.median(sis) if sis else 0.0

    def rank(c: dict) -> tuple:
        si = num(c, "SI") or 0.0
        try:
            reps = int(c.get("n_ic50") or 0) + int(c.get("n_cc50") or 0)
        except ValueError:
            reps = 0
        return (abs(si - median), -reps)

    return min(pool, key=rank)


def build_from_r2() -> tuple[list[dict], list[dict], Counter]:
    """База: молекулы R2, по одной согласованной строке на молекулу."""
    pairs = read(PAIRS_R2)
    mols = read(MOLS_R2)

    by_skel: dict[str, list[dict]] = {}
    for p in pairs:
        by_skel.setdefault(p["skeleton"] or skel(p.get("inchikey")), []).append(p)

    rows: list[dict] = []
    reported: list[dict] = []
    stats: Counter = Counter()

    for m in mols:
        sk = m["skeleton"] or skel(m.get("inchikey"))
        cands = [
            p for p in by_skel.get(sk, [])
            if p["quality"] in USABLE_QUALITY
            and num(p, "IC50") is not None
            and num(p, "CC50") is not None
        ]
        if not cands:
            # Остались только опубликованные готовые SI без IC50 и CC50.
            reported.append({
                "smiles": m["smiles"],
                "SI": m["SI"],
                "doi": m["doi"],
                "inchikey": m["inchikey"],
                "skeleton": sk,
                "quality": m["quality"],
                "n_studies": m["n_studies"],
                "why_excluded": "в источнике опубликован готовый SI, IC50 и CC50 отсутствуют",
            })
            stats["r2_reported_only"] += 1
            continue

        rep = pick_representative(cands)
        ic50, cc50 = num(rep, "IC50"), num(rep, "CC50")
        sources = {p["source"] for p in by_skel.get(sk, []) if p.get("source")}

        rows.append({
            "smiles": rep["smiles"],
            "IC50": f"{ic50:.6g}",
            "CC50": f"{cc50:.6g}",
            # Считаем заново из выбранной пары: согласованность по построению.
            "SI": f"{cc50 / ic50:.6g}",
            "doi": rep.get("doi") or "",
            "inchikey": rep.get("inchikey") or m["inchikey"],
            "skeleton": sk,
            "source": "+".join(sorted(sources)) or rep.get("source", ""),
            "origin": "r2",
            "quality": rep["quality"],
            # Усечённый CC50 означает, что настоящий SI выше указанного.
            "si_reliable": str(rep["quality"] == "exact"),
            "pairing_quality": "same_document",
            "ic50_censored": str(is_true(rep.get("ic50_censored"))),
            "cc50_censored": str(is_true(rep.get("cc50_censored"))),
            "n_studies": m.get("n_studies", ""),
            "si_min": m.get("si_min", ""),
            "si_max": m.get("si_max", ""),
            "n_forms": m.get("n_forms", ""),
            "document": rep.get("doc_key", ""),
        })
        stats["r2_kept"] += 1
        stats[f"r2_quality_{rep['quality']}"] += 1

    return rows, reported, stats


def harvest_from_pr1(have: set[str]) -> tuple[list[dict], Counter]:
    """Из набора PR #1 берём только чистые молекулы, которых у нас нет."""
    data = read(DATASET_PR1)
    stats: Counter = Counter({"pr1_total": len(data)})

    # Отдельно, по всему набору целиком, а не только по новым молекулам:
    # сколько там записей с IC50 равным CC50 и сколько из них считаются
    # надёжными. Это замечание для R3, см. отчёт.
    for r in data:
        ic50, cc50 = num(r, "IC50"), num(r, "CC50")
        if ic50 is None or cc50 is None or abs(ic50 - cc50) >= 1e-12:
            continue
        stats["pr1_artifact_total"] += 1
        if is_true(r.get("si_reliable")):
            stats["pr1_artifact_but_reliable"] += 1

    best: dict[str, dict] = {}
    for r in data:
        sk = skel(r.get("inchikey"))
        if not sk:
            continue
        if sk in have:
            stats["pr1_already_have"] += 1
            continue

        ic50, cc50 = num(r, "IC50"), num(r, "CC50")
        if ic50 is None or cc50 is None:
            stats["pr1_drop_no_values"] += 1
            continue
        # Потолок концентрации, записанный дважды: SI вырождается в единицу.
        if abs(ic50 - cc50) < 1e-12:
            stats["pr1_drop_artifact"] += 1
            continue
        if not is_true(r.get("si_reliable")):
            stats["pr1_drop_unreliable"] += 1
            continue
        if is_true(r.get("ic50_censored")) or is_true(r.get("cc50_censored")):
            stats["pr1_drop_censored"] += 1
            continue
        if r.get("pairing_quality") != "same_document":
            stats["pr1_drop_other_document"] += 1
            continue

        # У PR #1 дедупликация по полному ключу, поэтому один скелет может
        # прийти дважды в виде стереоизомеров. Оставляем один.
        prev = best.get(sk)
        if prev is None or (num(r, "n_ic50") or 0) > (num(prev, "n_ic50") or 0):
            best[sk] = r

    rows = []
    for sk, r in sorted(best.items()):
        ic50, cc50 = num(r, "IC50"), num(r, "CC50")
        rows.append({
            "smiles": r["smiles"],
            "IC50": f"{ic50:.6g}",
            "CC50": f"{cc50:.6g}",
            "SI": f"{cc50 / ic50:.6g}",
            "doi": r.get("doi") or "",
            "inchikey": r.get("inchikey", ""),
            "skeleton": sk,
            "source": r.get("source", ""),
            "origin": "pr1",
            "quality": "exact",
            "si_reliable": "True",
            "pairing_quality": r.get("pairing_quality", ""),
            "ic50_censored": "False",
            "cc50_censored": "False",
            "n_studies": "1",
            "si_min": f"{cc50 / ic50:.6g}",
            "si_max": f"{cc50 / ic50:.6g}",
            "n_forms": "1",
            "document": r.get("document_id", ""),
        })
    stats["pr1_kept"] = len(rows)
    return rows, stats


def validate(rows: list[dict]) -> list[str]:
    """Повторяет проверки tests/test_dataset.py без pandas и RDKit.

    Разбор SMILES здесь не проверяется: он требует RDKit, а обе исходные
    выгрузки уже прошли его в своих валидаторах, новых структур мы не
    придумываем.
    """
    problems = []
    if list(rows[0])[:5] != REQUIRED_FIRST:
        problems.append("обязательные пять колонок не идут первыми")

    for col in ("IC50", "CC50", "SI"):
        missing = [r for r in rows if num(r, col) is None]
        if missing:
            problems.append(f"{col}: {len(missing)} пропусков или неположительных")

    off = []
    for r in rows:
        ic50, cc50, si = num(r, "IC50"), num(r, "CC50"), num(r, "SI")
        if None in (ic50, cc50, si):
            continue
        calc = cc50 / ic50
        if abs(calc - si) / max(abs(calc), 1e-12) >= 0.01:
            off.append(r)
    if off:
        problems.append(f"SI не равен CC50 / IC50 на {len(off)} строках")

    import re
    doi_re = re.compile(r"^10\.\d{4,9}/\S+$")
    bad_doi = [r for r in rows if r["doi"] and not doi_re.match(r["doi"])]
    if bad_doi:
        problems.append(f"{len(bad_doi)} DOI не соответствуют формату")

    keys = [r["inchikey"] for r in rows if r["inchikey"]]
    if len(keys) != len(set(keys)):
        problems.append("есть повторяющиеся InChIKey")
    sks = [r["skeleton"] for r in rows]
    if len(sks) != len(set(sks)):
        problems.append("есть повторяющиеся скелеты")

    for col in ("si_reliable", "pairing_quality", "source"):
        if col not in rows[0]:
            problems.append(f"нет служебной колонки {col}")
    if not any(is_true(r["si_reliable"]) for r in rows):
        problems.append("нет ни одной надёжной записи")
    return problems


def main() -> None:
    for path in (PAIRS_R2, MOLS_R2, DATASET_PR1):
        if not os.path.exists(path):
            raise SystemExit(f"нет входного файла {path}, запускайте из корня репозитория")

    base, reported, s1 = build_from_r2()
    have = {r["skeleton"] for r in base}
    extra, s2 = harvest_from_pr1(have)

    merged = base + extra
    merged.sort(key=lambda r: -(num(r, "SI") or 0))

    log("# Объединение внешних данных")
    log()
    log("## Что откуда взято")
    log()
    log(f"* база R2: {s1['r2_kept']} молекул")
    log(f"  * точные измерения: {s1.get('r2_quality_exact', 0)}")
    log(f"  * CC50 усечён, настоящий SI выше указанного: {s1.get('r2_quality_lower_bound', 0)}")
    log(f"* добрано из PR #1: {s2['pr1_kept']} молекул")
    log(f"* итого в объединённом наборе: {len(merged)}")
    log()
    log("## Что из PR #1 отброшено и почему")
    log()
    log(f"* всего строк в наборе PR #1: {s2['pr1_total']}")
    log(f"* уже есть в базе R2 по скелету InChIKey: {s2.get('pr1_already_have', 0)}")
    log(f"* IC50 в точности равен CC50: {s2.get('pr1_drop_artifact', 0)} — "
        "это потолок концентрации, записанный дважды, а не измерение")
    log(f"* помечены si_reliable = False: {s2.get('pr1_drop_unreliable', 0)}")
    log(f"* усечена хотя бы одна величина: {s2.get('pr1_drop_censored', 0)} — "
        "SI там граница, а не значение")
    log(f"* цитотоксичность из другой публикации: {s2.get('pr1_drop_other_document', 0)}")
    log(f"* нет IC50 или CC50: {s2.get('pr1_drop_no_values', 0)}")
    log()
    log("## Вынесено отдельно")
    log()
    log(f"* {s1.get('r2_reported_only', 0)} молекул, у которых в источнике опубликован "
        "готовый SI, а IC50 и CC50 отсутствуют. Лежат в "
        f"`{OUT_REPORTED}`. В основной файл не вошли: без IC50 и CC50 "
        "формула SI не проверяется, а происхождение такого SI "
        "подтвердить нечем.")
    if reported:
        by_doi = Counter(r["doi"] for r in reported)
        by_si = Counter(r["SI"] for r in reported)
        top_doi, n_doi = by_doi.most_common(1)[0]
        top_si, n_si = by_si.most_common(1)[0]
        log(f"* проверка этой группы: {n_doi} из {len(reported)} записей пришли из одной "
            f"публикации `{top_doi}`, и у {n_si} из них SI в точности равен {top_si}. "
            "Это не независимые измерения, а одно общее значение, проставленное "
            "в статье целой серии соединений, скорее всего в виде «SI больше «"
            f"{top_si}». Обучаться на них как на разных измерениях нельзя.")
    log()

    sis = [num(r, "SI") for r in merged]
    sis = sorted(s for s in sis if s is not None)
    log("## Распределение SI")
    log()
    log(f"* медиана {statistics.median(sis):.3g}, "
        f"90-й процентиль {sis[int(0.9 * len(sis))]:.3g}, максимум {max(sis):.3g}")
    log(f"* надёжных записей: {sum(1 for r in merged if is_true(r['si_reliable']))} из {len(merged)}")
    log(f"* по источникам: {dict(Counter(r['source'] for r in merged))}")
    log()

    problems = validate(merged)
    log("## Проверка")
    log()
    if problems:
        for p in problems:
            log(f"* ОШИБКА — {p}")
    else:
        log("* OK — обязательные пять колонок идут первыми")
        log("* OK — IC50, CC50 и SI заполнены и положительны везде")
        log("* OK — SI равен CC50 / IC50 на всех строках")
        log("* OK — все DOI соответствуют формату")
        log("* OK — одна строка на молекулу, повторов нет")
        log("* OK — служебные колонки si_reliable, pairing_quality, source на месте")
        log()
        log("Разбор SMILES требует RDKit и здесь не проверяется: обе исходные "
            "выгрузки прошли его в своих валидаторах, новых структур скрипт "
            "не создаёт.")
    log()
    log("## Для R3: замечание по набору из PR #1")
    log()
    log(f"В `{DATASET_PR1}` есть записи, где IC50 в точности равен CC50. "
        "Это не измерение: при максимальной проверенной концентрации эффекта "
        "не увидели ни на вирусе, ни на клетках, и обе величины записаны "
        "одним и тем же потолком. Их отношение механически даёт SI = 1.")
    log()
    log(f"* таких записей всего: {s2.get('pr1_artifact_total', 0)} из {s2['pr1_total']}")
    log(f"* из них проходят фильтр `si_reliable = True`: {s2.get('pr1_artifact_but_reliable', 0)}")
    log()
    log("Причина в том, что `si_reliable` ставится в `False` только когда "
        "усечены обе величины. Если усечён лишь IC50, а CC50 записан тем же "
        "числом, флаг остаётся `True`. Отсюда медиана SI ровно 1.00 в "
        "`reports/dataset_validation.md`, хотя в трейне организаторов она 1.84.")
    log()
    log("Предлагаемая правка в `src/data/build_dataset.py`, одна строка:")
    log()
    log("```python")
    log('"si_reliable": not (bool(r.censored) and cc_censored)')
    log("                 and abs(float(r.value_uM) - cc50) > 1e-12,")
    log("```")
    log()
    log("Файл принадлежит R2 по таблице в `CONTRIBUTING.md`, но набор "
        "`vacv_external_dataset.csv` уже слит и прибит тестом, поэтому "
        "правку оставляю на согласование, а не вношу молча.")

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    with open(OUT_MERGED, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(merged)

    with open(OUT_REPORTED, "w", newline="", encoding="utf-8") as f:
        cols = ["smiles", "SI", "doi", "inchikey", "skeleton", "quality",
                "n_studies", "why_excluded"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(reported)

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(notes) + "\n")

    print()
    print(f"записано: {OUT_MERGED} ({len(merged)} строк)")
    print(f"записано: {OUT_REPORTED} ({len(reported)} строк)")
    print(f"записано: {OUT_REPORT}")
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
