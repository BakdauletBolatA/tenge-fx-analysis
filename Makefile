.PHONY: install fetch run test clean-figures

install:
	python -m venv .venv
	.venv/bin/pip install -r requirements.txt

fetch:
	.venv/bin/python -m src.fetch_nbk

run:
	.venv/bin/python -m src.pipeline

test:
	.venv/bin/python -m pytest -q

clean-figures:
	rm -f reports/figures/*.png
