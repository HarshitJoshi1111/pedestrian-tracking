# Speed on Apple M1

100 frames of MOT17-09 (1920x1080), model input 736x1280, after 10 warm-up frames. Time per frame including pre-processing and non-maximum suppression, without reading the file. The tracker adds about 0.5 ms per frame.

| Model | Runtime | ms/frame (mean) | median | 95th percentile | FPS |
|---|---|---|---|---|---|
| yolo11n.pt | PyTorch cpu | 96.2 | 95.7 | 99.1 | 10.4 |
| yolo11n.pt | PyTorch mps | 44.3 | 44.1 | 47.1 | 22.6 |
| yolo11n.pt | ONNX Runtime CPU | 103.8 | 103.7 | 106.7 | 9.6 |
| yolo11n.pt | ONNX Runtime CoreML | 30.7 | 30.7 | 31.2 | 32.6 |
| yolo11s.pt | PyTorch cpu | 213.0 | 212.9 | 219.1 | 4.7 |
| yolo11s.pt | PyTorch mps | 88.9 | 88.9 | 92.4 | 11.2 |
| yolo11s.pt | ONNX Runtime CPU | 399.6 | 400.3 | 408.4 | 2.5 |
| yolo11s.pt | ONNX Runtime CoreML | 55.6 | 55.3 | 60.4 | 18.0 |

| Model | PyTorch file | ONNX file |
|---|---|---|
| yolo11n.pt | 5.6 MB | 11.0 MB |
| yolo11s.pt | 19.3 MB | 38.3 MB |
