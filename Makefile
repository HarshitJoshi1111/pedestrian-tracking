.PHONY: download stats detect evaluate figures baseline detect-train tune track reference evaluate-track tracking test lint

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

detect-train:
	python -m src.detect --split train --imgsz 1280

tune:
	python -m src.track --tune

track:
	python -m src.track

reference:
	python -m src.reference_track

evaluate-track:
	python -m src.evaluate_track

tracking: detect-train tune track reference evaluate-track

test:
	pytest -q

lint:
	ruff check .
