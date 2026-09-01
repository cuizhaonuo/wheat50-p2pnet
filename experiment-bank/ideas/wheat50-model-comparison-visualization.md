# Wheat50 Model Comparison Visualization

## Goal

Compare the original-image baseline, foreground baseline, non-overlapping
Copy-Paste, and positive 768 crop experiments in a common set of plots and
inspect their behavior on individual wheat spikes.

## Evaluation

- Put all five TensorBoard scalar tags into one comparison figure.
- Run every `best_mae.pth` checkpoint on the foreground validation split with
  confidence threshold 0.5.
- For `count_MAE_spike`, assign a predicted point to a spike only when its
  center is inside that ground-truth instance mask. This is evaluation-time
  association and is not an output of P2PNet.
- Plot ground-truth spikelet count against assigned predicted count for every
  non-empty annotated spike.
- Select the three validation images with the most countable `wheat_2`
  instances and compare all four prediction overlays.

## Expected Outputs

Store the figures, per-spike CSV, and machine-readable summary below
`figures/wheat50-model-comparison/`.
