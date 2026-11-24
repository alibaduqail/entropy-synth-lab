# thermal-yolo: stock YOLOv8n vs a thermally trained YOLOv8n

Question: does a YOLOv8n trained on thermal images behave differently from the stock (COCO,
visible-light) YOLOv8n the project uses, in what it detects and in what it costs on the Jetson?

## What was built

- **Stock**: `~/yolov8n.pt`, 80 COCO classes, the model used in every earlier run.
- **Thermal**: the same weights fine-tuned on the Roboflow-100 "flir-camera-objects" set
  (a CC-BY-4.0 copy of FLIR ADAS: 9,306 train / 2,854 val / 1,452 test images, 640x640,
  classes person, bicycle, car, dog). Trained on the Jetson, imgsz 640.
- Training was planned for 20 epochs. It hit a CUDA allocator failure in epoch 8 (batch 16),
  was resumed at batch 8, and the process died during validation in epoch 17 (most likely
  memory). The best checkpoint up to epoch 16 is used (val mAP50 0.68 over the four classes).
- Code: `synthlab/thermal_yolo/`. Weights: Jetson `~/synthlab/thermal-yolo/train/flir_v8n/weights/best.pt`
  (not in git).

## 1. On FLIR test images (the thermal model's own domain)

`results/thermal-yolo/flir_test_scores.json`, 1,452 held-out images, both models at imgsz 640.

| class | stock mAP50 | thermal mAP50 | stock recall | thermal recall |
|---|---|---|---|---|
| person | 0.52 | 0.84 | 0.50 | 0.78 |
| car | 0.48 | 0.87 | 0.38 | 0.77 |
| bicycle | 0.20 | 0.65 | 0.15 | 0.62 |

## 2. On the real Lepton recordings (harness, conf 0.25)

`results/thermal-yolo/by_bank.csv` (paced rows) and `class_hist.json`. No ground-truth boxes
exist; "one person" is the scene label.

| real bank | scene | stock: frames with a person | thermal: frames with a person | stock persons/frame | thermal persons/frame |
|---|---|---|---|---|---|
| real_A0_0/1/2, radiometric_A0, heatlamp_off | empty | 0 | 0 | 0 | 0 |
| real_A1_0 | one person + lamp | 1.00 | 0.93 | 1.00 | 1.58 |
| real_A1_1 | one person + lamp | 0.72 | 0.98 | 0.72 | 1.35 |
| real_A1_2 | one person + lamp | 1.00 | 0.81 | 1.00 | 0.95 |
| real_radiometric_A1 | one person | 0.99 | 0.06 | 0.99 | 0.06 |
| real_heatlamp_on | one person + lamp | 0.76 | 0.78 | 0.79 | 1.21 |
| mean of the five | | 0.90 | 0.71 | 0.90 | 1.03 |

- Neither model raises a false alarm in any empty-room bank, even at conf 0.10.
- The thermal model is less sure: mean top confidence 0.46 vs 0.70 (paced, real banks).
- It splits one person into several boxes (1.2-1.6 persons per frame in four of five banks;
  about 3 per frame at conf 0.10).
- In `real_radiometric_A1` it calls the person a **car** in 70% of frames and a person in 5%.
  At conf 0.10 it still finds a person in only 28% of frames.
- The stock model's mistakes are different: toothbrush, sports ball, vase (heat-lamp banks).
- On synthetic banks the thermal model is worse on recombined real people (people_1: 35% vs
  100% of frames) and no better on physics-rendered people (0% vs 9%).

Reading: FLIR ADAS is outdoor driving footage from a 640x512 camera; people are small, upright
and far away. A seated person filling a 160x120 Lepton frame is out of that distribution. A
"thermally trained" model is not automatically better on this room.

## 3. Cost on the Jetson (PyTorch FP32, same harness, alternating rounds)

`results/thermal-yolo/paired_diff.csv`. Unpaced: 21 banks x 2 rounds per model. Paced (8.7 fps):
10 real banks x 1 round per model. All 104 trials passed the harness telemetry checks.

| paced, 10 real banks | stock | thermal | thermal - stock (95% CI) |
|---|---|---|---|
| YOLO inference (ms) | 30.85 | 29.49 | -1.36 (-1.37 to -1.34) |
| post-processing (ms) | 3.76 | 3.77 | +0.01 (-0.32 to +0.35) |
| whole pipeline (ms/frame) | 74.68 | 73.40 | -1.28 (-1.61 to -0.96) |
| power (W) | 5.816 | 5.744 | -0.072 (-0.087 to -0.058) |
| energy (J/frame) | 0.6685 | 0.6602 | -0.0083 (-0.0099 to -0.0067) |

- Inference is 1.36 ms (4.4%) faster and is flat across scenes for both models. The backbone is
  identical; the detection head scores 4 classes, not 80.
- Post-processing is the same on average. It follows what each model detects: lower where the
  thermal model misses the person (radiometric_A1: 4.5 vs 5.6 ms), higher where it reports
  extra boxes (real_A1_1: 6.0 vs 4.9 ms).
- Unpaced, 21 banks: pipeline -1.31 ms (CI -1.97 to -0.63), energy -13 mJ/frame (-2.9%).
- Pre-NMS candidates are fewer with the thermal model (paced 2.8 vs 4.5 per frame).
- Empty vs occupied gap (paced, real): stock 73.3-73.8 -> 75.5-76.0 ms; thermal 71.9-72.5 ->
  73.9-75.2 ms. The input-dependent swing is about 2 ms for both.

## What it means for the study

- Swapping in a thermally trained YOLO does not change the overnight conclusion: inference is
  input-independent, and the only input-dependent thermal cost is post-processing, which
  follows detections.
- The cost differences between the two models (about 1.3 ms, 0.07 W) come from the class count
  of the head, not from thermal training as such.
- Detection behaviour does change, and on this room it changed for the worse with FLIR training.

## Limits

- One fine-tune, one seed, stopped at epoch 16 of 20.
- One paced round per model; power differences of 0.07 W are near the across-bank noise seen
  overnight (0.045 W residual SD), though consistent in sign across all 10 banks.
- No ground-truth boxes on Lepton frames: detection rates are against scene labels only.
- A model fine-tuned on labelled Lepton frames from this room was not tried; that is the fair
  test of "thermal training helps here".
- TensorRT was not rebuilt for the thermal model.
