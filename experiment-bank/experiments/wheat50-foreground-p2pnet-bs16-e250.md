# Wheat50 Foreground P2PNet, 250 Epochs

## Status

Completed on 2026-08-27 from code commit `2bdf5f7`.

## Configuration

See [the experiment idea](../ideas/wheat50-foreground-p2pnet-bs16-e250.md).

## Metrics

- Best validation MAE: 5.8025 at epoch 240
- Best validation MSE: 8.0323 at epoch 240
- Epoch 0 training loss: 0.3584
- Epoch 0 unscaled point loss: 6.2284
- Epoch 249 training loss: 0.0110
- Epoch 249 unscaled point loss: 43.8204
- Minimum unscaled point loss: 4.9273 at epoch 4
- Maximum unscaled point loss: 49.4345 at epoch 61
- Training time: 0:54:48

## Observations

- Foreground-only images improved the best MAE from 16.8025 to 5.8025 and
  MSE from 22.4458 to 8.0323 relative to the original-image baseline.
- Classification loss decreased from 0.3572 to 0.0022.
- The point-loss instability remained: unscaled point loss rose from its
  epoch-4 minimum of 4.9273 to 43.8204 at the end.
- Shortening the run to 250 epochs retained the best region, which occurred
  at epoch 240.

## Outcome

Removing non-target pixels substantially improved counting, but did not make
coordinate regression converge smoothly. This foreground-only run is the
common baseline for the Copy-Paste and positive-crop experiments.

## Superseded metrics

The MAE and MSE above were measured at the hard-coded 0.5 threshold. See
[the threshold selection record](wheat50-eval-threshold-selection.md) for the
numbers at a threshold chosen per run, and for which conclusions survive.
