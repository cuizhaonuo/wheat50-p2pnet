# Wheat50 Foreground Non-overlapping Copy-Paste

## Status

Completed automatically on 2026-08-27 19:06 CST from code commit `de3c438`.

## Configuration

See [the experiment idea](../ideas/wheat50-foreground-copypaste-bs16-e250.md).

Artifacts are stored in `logs/wheat50-foreground-copypaste-bs16-e250`,
`runs/wheat50-foreground-copypaste-bs16-e250`, and
`checkpoints/wheat50-foreground-copypaste-bs16-e250` in the main worktree.

## Metrics

- Best validation MAE: 3.8889 at epoch 240
- MSE at best-MAE epoch: 5.1771
- Best validation MSE: 5.1771 at epoch 240
- MAE at best-MSE epoch: 3.8889
- Final epoch: 249
- Final training loss: 0.0133
- Final classification loss: 0.0049
- Final weighted point loss: 0.0084
- Final unscaled point loss: 41.9789
- Minimum unscaled point loss: 3.7611 at epoch 1
- Maximum unscaled point loss: 50.7497 at epoch 24
- Training time: 1:52:04

## Observations

- Automated finalization confirmed that all configured epochs ended and validation ran successfully.
- Best MAE occurred at epoch 240; best MSE occurred at epoch 240.
- Unscaled point loss ranged from 3.7611 to 50.7497.

## Outcome

The experiment improved best MAE by 1.9136 and changed best MSE by -2.8552 relative to the foreground-only baseline (5.8025/8.0323). The unscaled point loss finished at 41.9789 after a minimum of 3.7611.

## Superseded metrics

The MAE and MSE above were measured at the hard-coded 0.5 threshold. See
[the threshold selection record](wheat50-eval-threshold-selection.md) for the
numbers at a threshold chosen per run, and for which conclusions survive.
