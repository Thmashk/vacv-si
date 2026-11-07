PY := ./.venv/bin/python
PIP := ./.venv/bin/pip

.PHONY: env hooks test collect data validate-data cv submit final submit-safe submit-aggressive all clean

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

# Воспроизведение финального сабмита целиком, от обучения до CSV.
# opt_mix3 — взвешенная комбинация трёх прогнозов; веса считаются по
# оценкам лидерборда из submissions/lb_scores.json, он лежит в репозитории.
final:
	$(PY) -m src.make_submission --model blend --name blend
	$(PY) -m src.make_submission --model condmean --tail-scale 1.8 --name aggressive
	$(PY) -m src.submit_external --source r2 --model log --max-cap 5000 --name r2log_cap5k
	$(PY) -m src.opt_blend --parts blend aggressive r2log_cap5k --ridge 0.1 --name opt_mix3
	@echo "готово: submissions/opt_mix3.csv — это файл, отправленный на Kaggle"

submit-safe:
	$(PY) -m src.make_submission --model condmean --tail-scale 1.0 --name safe \
		--comment "условное среднее, честный хвост"

submit-aggressive:
	$(PY) -m src.make_submission --model condmean --tail-scale 1.8 --name aggressive \
		--comment "усиленный хвост: занижение гиганта дороже завышения обычной молекулы"

all: test cv final

clean:
	rm -rf reports/*.npz reports/*.md submissions/*.csv
