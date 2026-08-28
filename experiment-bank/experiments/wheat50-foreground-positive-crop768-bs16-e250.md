# Wheat50 Foreground Complete Positive Crop 768

## Status

Completed automatically on 2026-08-27 19:59 CST from code commit `596bf67`.

## Configuration

See [the experiment idea](../ideas/wheat50-foreground-positive-crop768-bs16-e250.md).

Artifacts are stored in `logs/wheat50-foreground-positive-crop768-bs16-e250`,
`runs/wheat50-foreground-positive-crop768-bs16-e250`, and
`checkpoints/wheat50-foreground-positive-crop768-bs16-e250` in the main
worktree.

## Metrics

- Best validation MAE: 6.0741 at epoch 205
- MSE at best-MAE epoch: 9.1625
- Best validation MSE: 9.0342 at epoch 100
- MAE at best-MSE epoch: 6.4074
- Final epoch: 249
- Final training loss: 0.0104
- Final classification loss: 0.0019
- Final weighted point loss: 0.0086
- Final unscaled point loss: 42.8766
- Minimum unscaled point loss: 4.2413 at epoch 0
- Maximum unscaled point loss: 47.3787 at epoch 45
- Training time: 2:44:35

## Observations

- Automated finalization confirmed that all configured epochs ended and validation ran successfully.
- Best MAE occurred at epoch 205; best MSE occurred at epoch 100.
- Unscaled point loss ranged from 4.2413 to 47.3787.

## Outcome

The experiment degraded best MAE by 0.2716 and changed best MSE by +1.0019 relative to the foreground-only baseline (5.8025/8.0323). The unscaled point loss finished at 42.8766 after a minimum of 4.2413.
