# Wheat50 DINOv3 ConvNeXt-Base Fine-tuned Backbone

## Status

Completed on 2026-08-31 from code commit `c365e5b`.

## Configuration

See [the experiment idea](../ideas/wheat50-dinov3-convnext-base-finetune-bs16-e250.md).

Artifacts are stored in `logs/wheat50-dinov3-convnext-base-finetune-bs16-e250`,
`runs/wheat50-dinov3-convnext-base-finetune-bs16-e250`, and
`checkpoints/wheat50-dinov3-convnext-base-finetune-bs16-e250` in the main worktree.

## Metrics

- Mean validation MAE over the last ten evaluations: 5.5593, standard deviation 1.1892
- Mean validation MSE over the last ten evaluations: 7.6223
- Best validation MAE: 3.4938 at epoch 240
- MSE at best-MAE epoch: 4.5718
- Final epoch: 249
- Final training loss: 0.0135
- Final classification loss: 0.0046
- Final weighted point loss: 0.0089
- Final unscaled point loss: 44.6107
- Minimum unscaled point loss: 5.4651 at epoch 3
- Maximum unscaled point loss: 53.6149 at epoch 10
- Trainable parameters: 66.88M
- Training time: 2:10:42, against 1:52:04 for the VGG Copy-Paste run

## Observations

- Best MAE arrives at epoch 240, the same late position as the VGG Copy-Paste
  run, so the schedule is healthy here in a way it is not for the frozen run.
- Fine-tuning recovers 3.95 MAE over the frozen run on identical features and
  an identical head, which places the frozen result on the protocol rather
  than on the pretrained representation.
- The plateau mean, 5.5593, sits 0.79 above the VGG Copy-Paste plateau of
  4.7719. That gap is below the 1.5 noise floor, so on a single seed the two
  are indistinguishable.
- The best MAE, 3.4938, is lower than the VGG Copy-Paste best of 3.8889, but
  best-of-N is the metric this project demoted: it is optimistic by 0.9 to 8.3
  MAE across the finished runs, and 2.07 here.
- The plateau standard deviation, 1.1892, is twice the 0.5800 of the VGG
  Copy-Paste run.

## Outcome

Replacing VGG16-BN with a fine-tuned DINOv3 ConvNeXt-Base does not measurably
change counting accuracy on this dataset. This is the general-domain control
the Copy-Paste result needed: its 3.19 MAE gain over the foreground baseline
is not explained by "any modern backbone beats VGG16", because a foundation
backbone with 3.1 times the parameters lands inside the noise floor of the
VGG run.

The higher plateau variance of both DINOv3 runs points at the two inherited
training-configuration defects, the missing weight decay and the constant
unwarmed learning rate, as the next thing to fix before scaling to
ConvNeXt-Large or moving on to the wheat-domain FoMo4Wheat backbone.

## Superseded metrics

The MAE and MSE above were measured at the hard-coded 0.5 threshold. See
[the threshold selection record](wheat50-eval-threshold-selection.md) for the
numbers at a threshold chosen per run, and for which conclusions survive.
