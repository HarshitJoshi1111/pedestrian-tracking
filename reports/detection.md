# Detection baseline (COCO-pretrained YOLO, class person, no training)

Validation sequences, boxes with IoU >= 0.5 count as found. F1, precision and recall at the confidence cut-off with the best F1. FPS = pre-processing + inference + NMS per frame.

| Run | AP50 | AP75 | AP50:95 | Max recall | Conf | Precision | Recall | F1 | ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|---|
| yolo11n-640 | 0.487 | 0.219 | 0.245 | 0.587 | 0.11 | 0.695 | 0.448 | 0.545 | 23.42 | 42.7 |
| yolo11n-1280 | 0.593 | 0.282 | 0.305 | 0.690 | 0.21 | 0.793 | 0.535 | 0.639 | 42.1 | 23.8 |
| yolo11s-640 | 0.533 | 0.255 | 0.274 | 0.616 | 0.10 | 0.729 | 0.501 | 0.594 | 31.09 | 32.2 |
| yolo11s-1280 | 0.596 | 0.293 | 0.312 | 0.680 | 0.23 | 0.819 | 0.546 | 0.655 | 84.85 | 11.8 |

## Recall by visibility (share of the person not occluded)

| Run | 0-0.3 | 0.3-0.6 | 0.6-1 |
|---|---|---|---|
| GT boxes | 13,925 | 5,109 | 17,711 |
| yolo11n-640 | 0.117 | 0.386 | 0.725 |
| yolo11n-1280 | 0.163 | 0.484 | 0.842 |
| yolo11s-640 | 0.147 | 0.450 | 0.794 |
| yolo11s-1280 | 0.171 | 0.511 | 0.851 |

## Recall by box height in pixels

| Run | 0-50 px | 50-100 px | 100-200 px | 200+ px |
|---|---|---|---|---|
| GT boxes | 2,001 | 11,634 | 12,968 | 10,142 |
| yolo11n-640 | 0.019 | 0.174 | 0.510 | 0.767 |
| yolo11n-1280 | 0.113 | 0.360 | 0.567 | 0.777 |
| yolo11s-640 | 0.054 | 0.281 | 0.546 | 0.783 |
| yolo11s-1280 | 0.130 | 0.368 | 0.581 | 0.787 |

## AP50 per sequence

| Run | MOT17-02 | MOT17-09 | MOT17-10 |
|---|---|---|---|
| yolo11n-640 | 0.387 | 0.691 | 0.544 |
| yolo11n-1280 | 0.496 | 0.721 | 0.674 |
| yolo11s-640 | 0.424 | 0.712 | 0.614 |
| yolo11s-1280 | 0.503 | 0.730 | 0.685 |
