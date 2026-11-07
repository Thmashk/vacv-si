PY := ./.venv/bin/python
PIP := ./.venv/bin/pip

.PHONY: env hooks test collect data validate-data cv submit submit-safe submit-aggressive all clean

env:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(MAKE) hooks

hooks:
	chmod +x .githooks/pre-push
	git config core.hooksPath .githooks
	@echo "Хуки включены: прямой пуш в main заблокирован, тесты идут перед пушем."

test:
	$(PY) -m pytest tests -q

# Выгрузка из открытых баз. Ответы кэшируются, повторный запуск быстрый.
collect:
	$(PY) -m src.collect.chembl_vacv
	$(PY) -m src.collect.pubchem_vacv

# Сборка итогового датасета из выгрузок.
data: collect
	$(PY) -m src.data.build_dataset

validate-data:
	$(PY) -m src.data.validate_dataset

cv:
	$(PY) -m src.run_cv

submit:
	$(PY) -m src.make_submission --model blend --name blend

submit-safe:
	$(PY) -m src.make_submission --model condmean --tail-scale 1.0 --name safe \
		--comment "условное среднее, честный хвост"

submit-aggressive:
	$(PY) -m src.make_submission --model condmean --tail-scale 1.8 --name aggressive \
		--comment "усиленный хвост: занижение гиганта дороже завышения обычной молекулы"

all: test cv submit

clean:
	rm -rf reports/*.npz reports/*.md submissions/*.csv
