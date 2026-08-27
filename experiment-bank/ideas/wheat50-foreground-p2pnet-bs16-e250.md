# Wheat50 Foreground P2PNet, 250 Epochs

## Goal

Measure whether removing all pixels outside `wheat_2` instance masks improves
P2PNet counting and point-loss convergence without changing the baseline
optimization settings.

## Hypothesis

Blackening non-target image content should reduce visual interference and may
allow classification and coordinate regression to converge more consistently.

## Configuration

- Dataset variant: foreground-only `wheat_2` images
- Dataset adapter: `COCO`
- Target category: `wheat_2`
- Training crop: 512 x 512, one crop per image
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

Only the dataset root and total epoch count differ from the 400-epoch
baseline. Because the first learning-rate milestone is epoch 250, this run
uses the initial learning rate throughout its optimization steps.

## Evaluation

Compare validation MAE, MSE, total loss, classification loss, and unscaled
point loss with the original-image baseline. In particular, check whether the
early all-background collapse and subsequent point-loss increase are reduced.
