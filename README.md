# Satellite Object Detection

This repository contains the dataset preprocessing, training, and evaluation scripts for a Master's thesis comparing object detection architectures on satellite imagery. The project evaluates three modern model families: YOLOv12, RF-DETR, and D-FINE across multiple size variants.

## Dataset

A custom **[dataset](https://mega.nz/file/TvxlVKZY#EYTy0WMJ7E_iaAh_DIGqXh2VgQLWutUm2iUjO5wdiaI)** was created by combining images from two public benchmarks: DIOR and DOTA-v2.0.

- **Class Mapping:** Bounding boxes from the source datasets were standardized to Horizontal Bounding Boxes (HBB) across 6 target classes: Plane, Bridge, Airport, Harbor, Vehicle, and Ship. Small and large vehicle categories from the source datasets were merged into a single Vehicle class.
- **Tiling:** Large satellite images were sliced into 800x800 pixel tiles with a 20% overlap.
- **Filtering:** Tiles containing no annotations were pruned, retaining a fixed ratio of 5% background tiles to limit negative samples.
- **Splitting:** Stratified multi-label split was applied to balance class distributions across training, validation, and test sets. Annotations were formatted in both COCO JSON and YOLO TXT.

## Hardware Environments

Models were trained and evaluated on two workstation configurations differing in GPU hardware:

- **Workstation 1:** NVIDIA RTX 6000 Ada Generation (48 GB VRAM)
- **Workstation 2:** NVIDIA RTX 6000 Blackwell (96 GB VRAM)

## Results

Performance was evaluated on the test split using standard COCO mAP metrics.

### Workstation 1

| Architecture | Variant | mAP50 | mAP50-95 |
|:---|:---:|:---:|:---:|
| **YOLOv12** | M | 0.886 | 0.650 |
| **YOLOv12** | L | 0.889 | 0.655 |
| **YOLOv12** | XL | 0.892 | 0.664 |
| **RF-DETR** | L | 0.820 | 0.596 |
| **RF-DETR** | XL | 0.806 | 0.577 |
| **RF-DETR** | 2XL | 0.815 | 0.598 |
| **D-FINE** | M | 0.795 | 0.583 |
| **D-FINE** | L | 0.783 | 0.574 |
| **D-FINE** | XL | 0.783 | 0.575 |

### Workstation 2

| Architecture | Variant | mAP50 | mAP50-95 |
|:---|:---:|:---:|:---:|
| **YOLOv12** | M | 0.896 | 0.626 |
| **YOLOv12** | L | 0.897 | 0.639 |
| **YOLOv12** | XL | **0.899** | **0.635** |
| **RF-DETR** | L | 0.806 | 0.576 |
| **RF-DETR** | XL | 0.816 | 0.591 |
| **RF-DETR** | 2XL | 0.826 | 0.607 |
| **D-FINE** | M | 0.792 | 0.587 |
| **D-FINE** | L | 0.788 | 0.581 |
| **D-FINE** | XL | 0.793 | 0.590 |
