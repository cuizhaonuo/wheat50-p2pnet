# Wheat50 Model Comparison

Generated from the four available `best_mae.pth` checkpoints on the
foreground validation split with confidence threshold 0.5.

## Main Figures

- [TensorBoard comparison](tensorboard-comparison.png): all recorded loss,
  MAE, and MSE tags in one figure.
- [Per-spike scatter](count-mae-spike-scatter.png): ground-truth spikelet
  count versus predicted points inside each ground-truth spike mask.
- [Machine-readable summary](summary.json)
- [Per-spike counts](per-spike-counts.csv)

## Inference Overlays

The `top-spike-images/` directory contains individual model outputs and a
four-panel comparison for each of the three validation images containing the
most countable spikes. Cyan lines are ground-truth instance boundaries, green
dots are predicted points, and the slightly larger yellow rings are
ground-truth spikelet points.

The ground-truth masks are used only to calculate per-spike evaluation metrics;
P2PNet itself still predicts an ungrouped point set and confidence scores.

Validation annotation `1923` has no points and is excluded from per-spike
metrics. This exclusion is recorded in `summary.json`.
