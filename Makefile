.PHONY: install fetch fetch-long fetch-brent fetch-all run run-short run-long test clean-figures

install:
	python -m venv .venv
	.venv/bin/pip install -r requirements.txt

fetch:
	.venv/bin/python -m src.fetch_nbk

fetch-long:
	.venv/bin/python -m src.fetch_nbk_archive

fetch-brent:
	.venv/bin/python -m src.fetch_brent --start 2004-06-01

fetch-all: fetch fetch-long fetch-brent

run:
	.venv/bin/python -m src.pipeline

run-short:
	.venv/bin/python -m src.pipeline --only short

run-long:
	.venv/bin/python -m src.pipeline --only long

test:
	.venv/bin/python -m pytest -q

clean-figures:
	rm -f reports/figures/*.png
