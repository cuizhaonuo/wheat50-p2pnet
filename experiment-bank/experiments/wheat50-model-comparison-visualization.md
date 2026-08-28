# Wheat50 Model Comparison Visualization

## Status

Completed on 2026-08-28 using each experiment's `best_mae.pth` checkpoint.

## Configuration

See [the evaluation idea](../ideas/wheat50-model-comparison-visualization.md).

- Dataset: foreground validation split, 81 images
- Confidence threshold: 0.5
- Countable spike instances: 256
- Assignment: predicted point center inside the ground-truth instance mask
- Outputs: `vis/wheat50-model-comparison/`

## Metrics

| Model | Image MAE | Image RMSE | count_MAE_spike | count_RMSE_spike | Pearson r |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original-image baseline | 19.1481 | 25.1261 | 7.2031 | 8.6639 | 0.4122 |
| Foreground baseline | 5.8025 | 8.0323 | 2.7695 | 3.6970 | 0.3414 |
| Non-overlap Copy-Paste | **3.8889** | **5.1771** | **2.1719** | **2.7825** | **0.5018** |
| Positive crop 768 | 6.0741 | 9.1625 | 3.0195 | 3.9256 | 0.4332 |

## Dense-image Visualizations

- Rank 1: image 34, 10 spikes and 132 ground-truth spikelet points
- Rank 2: image 52, 9 spikes and 134 ground-truth spikelet points
- Rank 3: image 33, 9 spikes and 130 ground-truth spikelet points

Each visualization contains cyan instance outlines, green solid prediction
points, and slightly larger yellow ground-truth rings for all four checkpoints.

## Observations

- The foreground baseline, Copy-Paste, and positive-crop image metrics exactly
  reproduce their recorded best-checkpoint validation metrics.
- Copy-Paste gives the best image-level and per-spike counting accuracy and the
  strongest per-spike correlation among the four models.
- The original-image baseline performs poorly on foreground-only inputs, so
  its image MAE here is not the same domain as its original TensorBoard MAE.
- Validation annotation 1923 has an empty `points` array and was excluded from
  per-spike metrics. The remaining model predictions inside that omitted mask
  are reported as unassigned predictions.

## Outcome

The visualization supports non-overlapping Copy-Paste as the strongest current
augmentation. It lowers `count_MAE_spike` by 0.5977 relative to the foreground
baseline, a 21.6% reduction, while also improving whole-image MAE from 5.8025
to 3.8889.
