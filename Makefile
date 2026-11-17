.PHONY: download stats detect evaluate figures baseline detect-train tune track reference evaluate-track tracking prepare-yolo train detect-finetuned finetune analytics render export detect-onnx benchmark deploy live test lint

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

export:
	python -m src.deploy export

detect-onnx:
	python -m src.deploy detect

benchmark:
	python -m src.deploy benchmark

deploy: export detect-onnx evaluate benchmark

live:
	python -m src.app data/raw/MOT17-09/img1/%06d.jpg --line 1100 500 1100 1080 --out data/demo/live.mp4

test:
	pytest -q

lint:
	ruff check .
