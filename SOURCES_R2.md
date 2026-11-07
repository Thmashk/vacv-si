# Источники данных

Выгрузка от 2026-11-07.

## ChEMBL

- Адрес: https://www.ebi.ac.uk/chembl/api/data/
- Версия базы на момент выгрузки: ChEMBL_37
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
