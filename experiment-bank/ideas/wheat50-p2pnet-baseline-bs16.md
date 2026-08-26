# Wheat50 P2PNet Baseline, Batch Size 16

## Goal

Train the single-class P2PNet point detector on the flattened `wheat_2.points`
targets and establish the first wheat50 counting/localization baseline.

## Hypothesis

The original P2PNet optimization settings should provide a stable starting
point after connecting `loss_point` to the weighted training loss. A batch size
of 16 should improve throughput without exhausting the shared A100 GPU.

## Configuration

- Dataset adapter: `COCO`
- Target category: `wheat_2`
- Training crop: 512 x 512, one crop per image
- Random scale: 0.8 to 1.2
- Horizontal flip probability: 0.5
- Batch size: 16
- Backbone/head learning rates: 1e-5 / 1e-4
- Weight decay: 1e-4
- Epochs: 400
- Learning-rate drops: epochs 250 and 350
- Matcher class/point costs: 1 / 0.05
- Point-loss coefficient: 0.0002
- Background coefficient: 0.5
- Anchor layout: 2 x 2
- Gradient clipping: 0.1
- Evaluation frequency: every 5 epochs

## Known Data Limitations

- The current train/validation split has source-image leakage, so validation
  metrics will measure this split rather than unseen-source generalization.
- Eight empty `wheat_2` annotations are ignored by the dataset adapter.

## Evaluation

Track total loss, classification loss, unscaled point loss, validation MAE,
validation MSE, and qualitative point localization. The first run establishes
the baseline rather than testing a claimed metric improvement.
