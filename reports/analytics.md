# Analytics on the static-camera sequences

Tracks: yolo11s-1280 + tracker settings from params.yaml (interpolation of gaps up to 20 frames). A counted crossing is correct if a real person crossed in the same direction within 15 frames. Crowd MAE: mean absolute difference between tracked and true people per frame (the truth includes fully hidden people).

| Sequence | Split | True in / out | Counted in / out | Count error | Precision | Recall | People/frame | Crowd MAE |
|---|---|---|---|---|---|---|---|---|
| MOT17-02 | val | 5 / 16 | 2 / 10 | 43% | 0.83 | 0.48 | 31.0 | 16.49 |
| MOT17-09 | val | 2 / 10 | 3 / 9 | 0% | 0.92 | 0.92 | 10.1 | 1.36 |
| MOT17-04 | train | 8 / 5 | 9 / 6 | 15% | 0.87 | 1.00 | 45.3 | 14.47 |

The tracker settings were tuned on the training sequences, so MOT17-04 is not an independent test.
