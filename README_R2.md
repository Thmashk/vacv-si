# Внешние данные (роль R2)

Два коллектора, сборщик, валидатор и генератор карточек. Все скрипты читают и пишут файлы относительно той папки, откуда их запускают, так что гонять их надо из корня проекта, где лежат файлы Kaggle.

Порядок:

```
python src/collect/collect_chembl.py
python src/collect/collect_pubchem.py
python src/data/build_dataset_r2.py
python src/data/validate_dataset_r2.py
python src/data/make_cards.py
```

Коллекторы умеют докачку: упали или прервал — запускай снова, продолжит с места обрыва. Служебные файлы `_done_*.txt` и `_ids_*.txt` в гит не кладём.

Для обучения брать `data/processed/vacv_external_molecules.csv` — там одна строка на молекулу. Файл `vacv_external_usable.csv` содержит отдельные пары «молекула и статья», одна молекула может встретиться несколько раз, при разбиении на свёртки это даст утечку.

Проверки: `compare_train.py` считает пересечения с train и test, `agree.py` сверяет наш SI с SI организаторов на общих молекулах. Обоим нужны `train.csv` и `test.csv` рядом.

Контрольные суммы обработанных файлов в `CHECKSUMS.txt`. Что внутри данных и откуда они — в `DATASET_CARD_R2.md` и `SOURCES_R2.md`.
