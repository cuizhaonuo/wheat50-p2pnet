# Wheat50 DINOv3 ConvNeXt-Base Frozen Backbone

## Status

Completed on 2026-08-31 from code commit `c365e5b`.

## Configuration

See [the experiment idea](../ideas/wheat50-dinov3-convnext-base-frozen-bs16-e250.md).

Artifacts are stored in `logs/wheat50-dinov3-convnext-base-frozen-bs16-e250`,
`runs/wheat50-dinov3-convnext-base-frozen-bs16-e250`, and
`checkpoints/wheat50-dinov3-convnext-base-frozen-bs16-e250` in the main worktree.

## Metrics

- Mean validation MAE over the last ten evaluations: 9.5086, standard deviation 1.3126
- Mean validation MSE over the last ten evaluations: 12.6600
- Best validation MAE: 4.6049 at epoch 20
- MSE at best-MAE epoch: 6.4108
- Final epoch: 249
- Final training loss: 0.0130
- Final classification loss: 0.0045
- Final weighted point loss: 0.0085
- Final unscaled point loss: 42.4625
- Minimum unscaled point loss: 5.7211 at epoch 1
- Maximum unscaled point loss: 55.3198 at epoch 10
- Trainable parameters: 6.76M of 66.88M
- Training time: 1:33:00, against 1:52:04 for the VGG Copy-Paste run

## Observations

- Validation MAE reached its best value at epoch 20 and degraded monotonically
  afterwards, ending 4.9 above that point. The head overfits the frozen
  features within the first tenth of the schedule and never recovers.
- The plateau standard deviation, 1.3126, is more than twice the 0.5800 of the
  VGG Copy-Paste run, so this configuration also converges less stably.
- The gap to the VGG Copy-Paste plateau, 9.5086 against 4.7719, is 4.74 MAE,
  far beyond the 1.5 noise floor, so the difference is a result rather than
  run-to-run variation.
- The early peak followed by steady degradation is the behaviour the absent
  weight decay predicts: `args.weight_decay` never reaches the optimizer, so
  6.76M randomly initialised parameters train unregularised on 320 images.

## Outcome

A frozen general-domain DINOv3 ConvNeXt-Base backbone is clearly worse than a
fine-tuned VGG16-BN on this task. Read together with the fine-tuned run, which
recovers 3.95 MAE on the same features, the frozen protocol rather than the
features is the limiting factor, so the result does not by itself argue that
DINOv3 features are unsuitable for point localisation.

## Superseded metrics

The MAE and MSE above were measured at the hard-coded 0.5 threshold. See
[the threshold selection record](wheat50-eval-threshold-selection.md) for the
numbers at a threshold chosen per run, and for which conclusions survive.
