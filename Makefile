.PHONY: download stats detect evaluate figures baseline detect-train tune track reference evaluate-track tracking prepare-yolo train detect-finetuned finetune analytics render test lint

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
	python -m src.track --tune --apply

track:
	python -m src.track

reference:
	python -m src.reference_track

evaluate-track:
	python -m src.evaluate_track

tracking: detect-train tune track reference evaluate-track

prepare-yolo:
	python -m src.prepare_yolo

train:
	python -m src.train

detect-finetuned:
	python -m src.detect --models yolo11n-mot17.pt --imgsz 1280

finetune: prepare-yolo train detect-finetuned evaluate track evaluate-track

analytics:
	python -m src.analytics

render:
	python -m src.render --gif

test:
	pytest -q

lint:
	ruff check .
