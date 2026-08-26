# Wheat50 P2PNet Baseline, Batch Size 16

## Status

Running since 2026-08-26 14:42 CST in tmux session
`wheat50_p2pnet_bs16` (PID 2487919), from code commit `796c879`.

## Configuration

See [the experiment idea](../ideas/wheat50-p2pnet-baseline-bs16.md).

## Preflight

- Batch size 8: 5.638 GiB peak allocated, 9.635 GiB peak reserved.
- Batch size 16: 11.953 GiB peak allocated, 20.029 GiB peak reserved.
- The real-data forward/backward test produced a nonzero regression-head
  gradient with finite classification and point losses.

## Metrics

- Epoch 0 training loss: 0.2830
- Epoch 0 unscaled point loss: 3.7298
- Epoch 5 validation MAE: 44.3457
- Epoch 5 validation MSE: 54.2623
- Epoch 8 training loss: 0.0099
- Epoch 8 unscaled point loss: 2.8284
- Epoch 15 validation MAE: 25.8395
- Epoch 15 validation MSE: 32.8312

## Observations

- The first nine epochs completed without out-of-memory or non-finite loss.
- `loss_point` is present in the averaged training statistics and decreases
  during the initial epochs, but increased again by epoch 15; localization
  quality needs continued monitoring.
- The latest checkpoint contains the model, optimizer, learning-rate
  scheduler, and epoch state for exact training continuation.
- There is no comparable pre-fix validation baseline because the regression
  loss was previously disconnected from the optimized loss.

## Outcome

Pending.
