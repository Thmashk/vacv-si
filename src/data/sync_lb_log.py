"""Переносит счёт с лидерборда из submissions/lb_scores.json в журнал.

Зачем. Журнал `submissions/log.csv` заполняется при сборке сабмита, а счёт
приходит позже, после отправки, и колонка LB остаётся пустой. Из-за этого в
журнале не оказалось счёта лучшего сабмита команды. Результат без записи в
журнал невоспроизводим, поэтому перенос сделан скриптом, а не руками.

Правила:

* заполняются только пустые ячейки LB; уже заполненные не трогаются;
* если одно и то же имя файла встречается в журнале несколько раз, счёт
  ставится только последней записи. Ранние записи фиксируют сборку, а не
  отправку, и дублировать в них счёт значило бы показать один результат
  дважды;
* если значения в журнале и в JSON расходятся, скрипт об этом сообщает и
  ничего не меняет: расхождение надо разбирать, а не затирать.

Запуск из корня репозитория:

    python3 src/data/sync_lb_log.py          # показать, что изменится
    python3 src/data/sync_lb_log.py --apply  # записать

Зависимостей нет, только стандартная библиотека.
"""
from __future__ import annotations

import csv
import io
import json
import os
import sys

LOG = "submissions/log.csv"
SCORES = "submissions/lb_scores.json"


def main() -> None:
    apply = "--apply" in sys.argv[1:]
    for path in (LOG, SCORES):
        if not os.path.exists(path):
            raise SystemExit(f"нет файла {path}, запускайте из корня репозитория")

    with open(LOG, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    with open(SCORES, encoding="utf-8") as f:
        scores = json.load(f)

    for col in ("файл", "LB"):
        if col not in fields:
            raise SystemExit(f"в журнале нет колонки {col!r}")

    # Последняя запись для каждого имени файла — та, что соответствует отправке.
    last: dict[str, int] = {}
    for i, r in enumerate(rows):
        last[(r.get("файл") or "").strip()] = i

    filled, skipped_dup, missing, conflicts = [], [], [], []
    by_line: dict[int, str] = {}        # номер строки данных -> значение LB
    for i, r in enumerate(rows):
        name = (r.get("файл") or "").strip()
        stem = name[:-4] if name.endswith(".csv") else name
        current = (r.get("LB") or "").strip()
        score = scores.get(stem)

        if score is None:
            if not current:
                missing.append(name)
            continue
        if current:
            if abs(float(current) - float(score)) > 1e-6:
                conflicts.append((name, current, score))
            continue
        if last.get(name) != i:
            skipped_dup.append(name)
            continue
        # Формат как у уже заполненных ячеек: до пяти знаков, без хвостовых нулей.
        by_line[i] = f"{score:.5f}".rstrip("0").rstrip(".")
        filled.append((name, score))

    print(f"журнал: {len(rows)} записей, счётов в JSON: {len(scores)}\n")
    if conflicts:
        print("РАСХОЖДЕНИЯ, ничего не меняю:")
        for n, a, b in conflicts:
            print(f"  {n}: в журнале {a}, в JSON {b}")
        print()
    print(f"заполнить LB ({len(filled)}):")
    for n, s in filled:
        print(f"  {n:20} {s}")
    if skipped_dup:
        print(f"\nпропущены как ранние записи того же файла ({len(skipped_dup)}):")
        for n in sorted(set(skipped_dup)):
            print(f"  {n}")
    if missing:
        print(f"\nсчёта нет ни в журнале, ни в JSON — не отправлялись ({len(missing)}):")
        for n in sorted(set(missing)):
            print(f"  {n}")

    if conflicts:
        raise SystemExit("\nсначала разберите расхождения")
    if not filled:
        print("\nнечего заполнять")
        return
    if not apply:
        print("\nэто предпросмотр, для записи добавьте --apply")
        return

    # Журнал дописывают параллельно, поэтому меняем только нужные ячейки и
    # оставляем остальные байты как есть. Полный перезапись через csv.writer
    # нормализует переводы строк и кавычки по всему файлу и даёт конфликт на
    # каждой строке; в этом журнале, например, одна строка на \r\n, а
    # остальные на \n.
    with open(LOG, newline="", encoding="utf-8") as f:
        raw = f.readlines()

    lb_at = fields.index("LB")
    changed = 0
    for i, name_score in by_line.items():
        line = raw[i + 1]                      # +1: строка заголовка
        body = line.rstrip("\r\n")
        term = line[len(body):]
        cells = next(csv.reader([body]))
        if len(cells) != len(fields):
            raise SystemExit(f"строка {i + 2}: {len(cells)} полей вместо {len(fields)}")
        cells[lb_at] = name_score
        out = io.StringIO()
        csv.writer(out, lineterminator="").writerow(cells)
        raw[i + 1] = out.getvalue() + term
        changed += 1

    with open(LOG, "w", newline="", encoding="utf-8") as f:
        f.writelines(raw)
    print(f"\nзаписано в {LOG}: изменено ячеек — {changed}")


if __name__ == "__main__":
    main()
