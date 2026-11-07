import datetime

import pandas as pd
import requests

DATASET = "vacv_external_dataset.csv"
MOLECULES = "vacv_external_molecules.csv"


def chembl_version():
    try:
        r = requests.get("https://www.ebi.ac.uk/chembl/api/data/status.json", timeout = 30)
        if r.status_code == 200:
            return r.json().get("chembl_db_version", "неизвестно")
    except Exception:
        pass
    return "неизвестно"


def main():
    today = datetime.date.today().isoformat()
    ver = chembl_version()
    df = pd.read_csv(DATASET)
    mols = pd.read_csv(MOLECULES)
    q = df["quality"].value_counts()
    by_src = df["source"].value_counts()

    sources = """# Источники данных

Выгрузка от {today}.

## ChEMBL

- Адрес: https://www.ebi.ac.uk/chembl/api/data/
- Версия базы на момент выгрузки: {ver}
- Скрипт: `collect_chembl.py`
- Как искали: поиск мишеней и ассеев по словам vaccinia, VACV, orthopoxvirus через `target/search` и `assay/search`, затем активности по их идентификаторам. Цитотоксичность добиралась из тех же документов запросом по `standard_type`.
- Лицензия: CC BY-SA 3.0 (формулировку сверять на сайте ChEMBL)

## PubChem BioAssay

- Адрес: https://pubchem.ncbi.nlm.nih.gov/rest/pug/
- Поиск ассеев: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pcassay
- Скрипт: `collect_pubchem.py`
- Как искали: поиск AID через E-utilities по словам vaccinia и orthopoxvirus, значения через эндпоинт `concise` (уже в микромолях), SMILES по CID, DOI через PubMed ID.
- Лицензия: условия задаёт каждый депозитор

## Проверка

`validate_dataset_r2.py` обращается к каждому адресу выше, перезапрашивает 20 случайных строк у ChEMBL и сверяет значения. Отчёт: `reports/dataset_validation_r2.md`.

Данные для молекул из `test.csv` вручную не искались. Поиск шёл только по вирусу и мишени.
""".format(today = today, ver = ver)

    card = """# Карточка датасета

`vacv_external_dataset.csv` и производные от него. Собран {today}.

## Что это

Внешние данные по активности малых молекул против вируса осповакцины, собранные скриптами из ChEMBL и PubChem для хакатона VACV. Используются как добавка к обучающей выборке Kaggle.

## Файлы

| Файл | Что внутри |
| --- | --- |
| `vacv_external_dataset.csv` | все пары «молекула и статья», {n_all} строк, включая отбракованные |
| `vacv_external_usable.csv` | только пригодные категории качества |
| `vacv_external_molecules.csv` | одна строка на молекулу, {n_mol} молекул, для обучения берут этот |

Первые пять колонок: `smiles, IC50, CC50, SI, doi`. Концентрации в микромолях, SI безразмерный.

## Как считался SI

Два пути. Основной: IC50 или EC50 против вакцинии делится на CC50 из цитотоксичного ассея **той же статьи**. Запасной: если в источнике уже указан готовый индекс селективности, берётся он, с пометкой `reported`.

Документы ChEMBL и PubChem сводятся к общему PubMed ID, иначе одна статья считалась бы дважды. Молекулы дедуплицируются по первым 14 символам InChIKey, то есть по скелету без стереохимии.

## Колонка quality

| Значение | Сколько | Что означает |
| --- | --- | --- |
{qrows}

Для обучения берут `exact`, `reported` и, с меньшим весом, `lower_bound`.

- `artifact` — IC50 в точности равен CC50. Это не измерение, а потолок концентрации, записанный дважды.
- `lower_bound` — CC50 усечён («больше такого-то»), значит настоящий SI выше указанного.
- `upper_bound` — усечён IC50, настоящий SI ниже.
- `bad` — усечены оба.

## Источники строк

{srows}

## Ограничения

- Заявленный авторами SI мог считаться по EC50 или на другой клеточной линии, проверить это по паре невозможно. Такие строки помечены `si_source = reported`.
- Энантиомеры склеиваются в одну молекулу. Колонка `n_forms` показывает, где это произошло.
- Измерения из разных лабораторий расходятся: из {n_multi} молекул, измеренных более чем в одной статье, {n_wide} расходятся больше чем в десять раз.
- Максимальный SI в датасете {si_max:.0f}, что выше максимума обучающей выборки Kaggle. Перед обучением стоит обрезать или клипнуть хвост.

## Лицензии

ChEMBL — CC BY-SA 3.0. PubChem — по условиям депозиторов. Адреса и даты выгрузки в `SOURCES_R2.md`.
"""

    qrows = "\n".join("| `" + k + "` | " + str(v) + " | |" for k, v in q.items())
    srows = "\n".join("- " + str(k) + ": " + str(v) + " строк" for k, v in by_src.items())
    multi = mols[mols["n_studies"] > 1]
    wide = multi[multi["spread"] > 10]

    card = card.format(
        today = today,
        n_all = len(df),
        n_mol = len(mols),
        qrows = qrows,
        srows = srows,
        n_multi = len(multi),
        n_wide = len(wide),
        si_max = mols["SI"].max(),
    )

    open("SOURCES_R2.md", "w", encoding = "utf-8").write(sources)
    open("DATASET_CARD_R2.md", "w", encoding = "utf-8").write(card)
    print("записаны SOURCES_R2.md и DATASET_CARD_R2.md")
    print("версия ChEMBL:", ver)


if __name__ == "__main__":
    main()
