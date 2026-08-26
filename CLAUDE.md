# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

P2PNet is a PyTorch implementation of a crowd counting and localization model (ICCV 2021 Oral). It uses a purely point-based framework built on VGG16 with an FPN decoder, predicting point proposals and confidence scores simultaneously without density maps.

## Commands

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Train on SHTechPartA:**
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --data_root $DATA_ROOT \
    --dataset_file SHHA \
    --epochs 3500 \
    --lr_drop 3500 \
    --output_dir ./logs \
    --checkpoints_dir ./weights \
    --tensorboard_dir ./logs \
    --lr 0.0001 \
    --lr_backbone 0.00001 \
    --batch_size 8 \
    --eval_freq 1 \
    --gpu_id 0
```

**Resume training from checkpoint:**
```bash
python train.py ... --resume ./weights/latest.pth
```

**Run inference/visualization demo:**
```bash
CUDA_VISIBLE_DEVICES=0 python run_test.py --weight_path ./weights/SHTechA.pth --output_dir ./logs/
```
The demo image path is hardcoded to `./vis/demo1.jpg` in `run_test.py:64`. The pretrained weight (MAE=51.96 on SHTechPartA) is at `./weights/SHTechA.pth`.

## Architecture

**Data flow:**
1. Input images → VGG16-BN backbone (`models/backbone.py`) extracts 4 intermediate feature maps (C1–C4)
2. FPN-style `Decoder` (`models/p2pnet.py:155`) fuses C3, C4, C5 via lateral connections → [P3, P4, P5]
3. P4 (8× downsampled) feeds two parallel heads:
   - `RegressionModel`: predicts offsets from anchor points → absolute point coords
   - `ClassificationModel`: predicts foreground/background scores
4. `AnchorPoints` generates a grid of reference points at pyramid level 3 only (stride 8), with `row×line` anchors per cell (default 2×2=4)
5. Output: `{'pred_logits': [B, N, 2], 'pred_points': [B, N, 2]}`

**Training:**
- `SetCriterion_Crowd` computes Hungarian matching via `HungarianMatcher_Crowd` (scipy `linear_sum_assignment`) between predictions and GT points
- Two losses: cross-entropy classification loss + MSE point regression loss (weighted by `point_loss_coef=0.0002`)
- Adam optimizer; backbone uses 10× lower LR than the rest; StepLR scheduler drops at `lr_drop` epoch
- Checkpoints: `latest.pth` saved every epoch, `best_mae.pth` saved when best MAE is matched

**Dataset (SHHA):**
- `crowd_datasets/SHHA/SHHA.py`: reads `.list` files mapping `img_path → gt_path` (one `x y` coord per line)
- Training augmentations: random scale [0.7, 1.3], random 128×128 crops (4 patches per image), random horizontal flip
- Inference: image dimensions must be multiples of 128 (see `run_test.py:69-71`)
- Default list files: `shanghai_tech_part_a_train.list` / `shanghai_tech_part_a_test.list` under `data_root`

**Inference threshold:** Confidence threshold is 0.5 (hardcoded in `engine.py:143` and `run_test.py:83`).

## Key Implementation Notes

- The model only uses `features[1:4]` (C2–C4) from the backbone — `features[0]` (C1) is computed but unused in the FPN forward pass (`p2pnet.py:215`)
- `RegressionModel` output is multiplied by 100 before adding anchor points (`p2pnet.py:219`), acting as a scale factor for the regression offsets
- `vgg_.py` is a local copy of VGG with pretrained weights support; do not replace with `torchvision.models.vgg` directly
- The codebase uses `from engine import *` — all training/eval functions and `utils` are imported this way into `train.py`
- Metrics reported: MAE (mean absolute error) and MSE (root mean squared error) on head count per image
