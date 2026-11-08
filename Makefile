.PHONY: download stats detect evaluate figures baseline test lint

download:
	python -m src.download

stats:
	python -m src.stats

detect:
	python -m src.detect

evaluate:
	python -m src.evaluate_det

figures:
	python -m src.visualize

baseline: download stats detect evaluate figures

test:
	pytest -q

lint:
	ruff check .
