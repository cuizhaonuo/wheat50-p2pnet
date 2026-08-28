# Wheat50 Foreground Non-overlapping Copy-Paste

## Goal

Test whether adding complete `wheat_2` instances without any mask overlap
improves counting and point-loss convergence over the foreground-only
baseline.

## Hypothesis

Copy-Paste increases foreground diversity and positive-query density while
the zero-overlap constraint preserves unambiguous instance pixels and points.

## Configuration

- Dataset variant: foreground-only `wheat_2` images
- Training crop: 512 x 512, one crop per image
- Random scale: 0.8 to 1.2
- Horizontal flip probability: 0.5
- Copy-Paste probability: 1.0
- Maximum pasted instances per crop: 3
- Placement attempts per instance: 20
- Required overlap with original/pasted masks: zero pixels
- Batch size: 16
- Backbone/head learning rates: 1e-5 / 1e-4
- Weight decay: 1e-4
- Epochs: 250
- Learning-rate drops: epochs 250 and 350
- Matcher class/point costs: 1 / 0.05
- Point-loss coefficient: 0.0002
- Background coefficient: 0.5
- Anchor layout: 2 x 2
- Gradient clipping: 0.1
- Evaluation frequency: every 5 epochs

## Evaluation

Compare best MAE/MSE and the full point-loss trajectory with the common
foreground-only baseline (MAE 5.8025, MSE 8.0323). Verify that every pasted
instance remains complete and non-overlapping during data loading.
