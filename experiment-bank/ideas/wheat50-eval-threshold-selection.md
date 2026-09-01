# Wheat50 Evaluation Threshold Selection

## Goal

Re-measure every finished run at a working point chosen for it, rather than at
the 0.5 that `engine.py` had hard-coded, and determine which of the project's
conclusions survive.

## Why the fixed threshold is a problem

Counting turns 65536 anchor points on a 1024-pixel validation image into a
count by keeping those whose class-1 probability exceeds a threshold. With a
mean of 3.3 ground-truth points per image, 65533 of them have to fall below it.
Nothing ties the calibration of that decision to 0.5, and `--eos_coef 0.5`
reweights the background class during training without the threshold ever
following.

## Method

Sweep 0.05 to 0.95 in steps of 0.05 on the `latest.pth` checkpoint, which is
the end-of-training model the plateau metric describes.

The threshold is chosen on the training images, evaluated in validation mode
with no augmentation, and the metrics are reported on the validation images.
Choosing it on the split it is reported on would leak, in the same way that
reporting the best MAE over all epochs does.

## Evaluation

The 1.5 MAE noise floor stays as fixed before the first DINOv3 run. Every
conclusion drawn so far is re-checked against it at the new working points.
