# Wheat50 Foreground Complete Positive Crop 768

## Goal

Test whether larger crops that always contain at least one complete
`wheat_2` instance improve counting and point-loss convergence over the
foreground-only baseline.

## Hypothesis

A 768 x 768 crop provides more context and more positive points per training
sample, while requiring a complete instance prevents empty or truncated-only
training targets.

## Configuration

- Dataset variant: foreground-only `wheat_2` images
- Training crop: 768 x 768, one crop per image
- Every crop must completely contain at least one `wheat_2` instance
- Empty source records are resampled from positive records
- Random scale: 0.8 to 1.2
- Horizontal flip probability: 0.5
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
foreground-only baseline (MAE 5.8025, MSE 8.0323). Confirm that all training
crops are non-empty and contain a complete target instance.
