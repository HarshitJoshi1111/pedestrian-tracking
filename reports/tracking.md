# Tracking on the validation sequences

MOT17-02, -09, -10: 145 people. MT/ML: people tracked for at least 80% / less than 20% of the frames they appear in. Tracker time excludes the detector. Settings tuned on the training sequences. ultralytics-bytetrack: the library's own tracker with its default settings, run inside the detector.

| Run | Detector | MOTA | IDF1 | IDP | IDR | ID switches | False positives | Misses | MT | ML | Track IDs | Tracker ms/frame |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sort-style | yolo11s-1280 | 0.415 | 0.406 | 0.612 | 0.303 | 321 | 1,317 | 19,864 | 31 | 50 | 456 | 0.45 |
| no-low-boxes | yolo11s-1280 | 0.431 | 0.462 | 0.680 | 0.350 | 205 | 1,422 | 19,274 | 32 | 49 | 240 | 0.5 |
| bytetrack | yolo11s-1280 | 0.445 | 0.446 | 0.624 | 0.347 | 228 | 1,930 | 18,228 | 35 | 43 | 250 | 0.54 |
| bytetrack+interp | yolo11s-1280 | 0.450 | 0.450 | 0.604 | 0.358 | 230 | 2,509 | 17,481 | 39 | 43 | 250 | 0.54 |
| bytetrack+interp-yolo11n | yolo11n-1280 | 0.436 | 0.457 | 0.645 | 0.354 | 188 | 1,984 | 18,541 | 38 | 47 | 201 | 0.5 |
| bytetrack+interp-finetuned | yolo11n-mot17-1280 | 0.401 | 0.423 | 0.587 | 0.331 | 227 | 2,879 | 18,911 | 35 | 48 | 225 | 0.5 |
| bytetrack+interp-nms07 | yolo11s-onnx-iou07-1280 | 0.443 | 0.458 | 0.612 | 0.367 | 245 | 2,744 | 17,463 | 40 | 44 | 258 | 0.54 |
| ultralytics-bytetrack | yolo11s-1280 | 0.419 | 0.448 | 0.589 | 0.362 | 378 | 3,416 | 17,563 | 35 | 40 | 544 | - |

## bytetrack+interp per sequence

| Sequence | MOTA | IDF1 | ID switches | People | Track IDs |
|---|---|---|---|---|---|
| MOT17-02 | 0.353 | 0.422 | 64 | 62 | 83 |
| MOT17-09 | 0.634 | 0.517 | 48 | 26 | 44 |
| MOT17-10 | 0.513 | 0.452 | 118 | 57 | 123 |
