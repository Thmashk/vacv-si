PY := ./.venv/bin/python
PIP := ./.venv/bin/pip

.PHONY: env test cv submit submit-safe submit-aggressive all clean

env:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

test:
	$(PY) -m pytest tests -q

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
