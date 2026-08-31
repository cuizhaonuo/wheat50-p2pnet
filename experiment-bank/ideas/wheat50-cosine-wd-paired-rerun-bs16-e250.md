# Wheat50 Cosine Schedule and Weight Decay, Paired Rerun

## Goal

Re-run the two best configurations, VGG16-BN with Copy-Paste and a fine-tuned
DINOv3 ConvNeXt-Base, under a corrected optimizer and schedule, so that the
backbone comparison is no longer confounded by two defects inherited from the
original P2PNet training script.

## The defects being corrected

The optimizer was constructed as `torch.optim.Adam(param_dicts, lr=args.lr)`,
so `args.weight_decay` never reached it and every experiment recorded so far
trained with no regularization at all. Separately, the learning-rate
milestones default past the final epoch, so the rate stayed at its initial
value for the whole run with no warmup.

Two observations point at these. The frozen DINOv3 run reached its best MAE at
epoch 20 and then degraded monotonically to 9.51, which is what an
unregularized 6.76M-parameter head does on 320 images. And both DINOv3 runs
finished with a plateau standard deviation near 1.2 against 0.58 for the VGG
Copy-Paste run, which is what a constant learning rate does at the end of
training.

## Hypothesis

The cosine decay is expected to carry most of the effect. It removes the late
drift that inflates plateau variance, and it should make the plateau mean both
lower and more stable for all configurations.

Weight decay is expected to contribute little at this scale, and this is worth
stating in advance rather than discovering afterwards. AdamW shrinks weights by
a factor of `1 - lr * weight_decay` per step. At 20 steps per epoch over 250
epochs, the repository's declared 1e-4 amounts to a total shrink of 0.005 per
cent, and even 0.05 amounts to 2.5 per cent. The correction is worth making
because the configuration should mean what it says, not because 5000 steps
leave room for decay to dominate.

## Configuration

Identical to the two experiments being repeated, except:

- Optimizer: AdamW, with weight decay reaching it for the first time
- Weight decay: 0.05, applied to weights of rank two or higher only, so biases
  and normalization scales are exempt
- Learning-rate schedule: cosine to zero over the epochs after the warmup
- Warmup: 5 epochs of linear ramp from one fifth of the target rate
- Everything else, including seed 42, unchanged

## Evaluation

Four runs form a two-by-two comparison, and the two older halves already
exist:

| Backbone | Old configuration | New configuration |
| --- | ---: | --- |
| VGG16-BN Copy-Paste | 4.772 +/- 0.580 | this rerun |
| DINOv3 ConvNeXt-Base fine-tuned | 5.559 +/- 1.189 | this rerun |

The primary metric stays the mean validation MAE over the last ten
evaluations, with 1.5 MAE as the noise floor. Two questions are being asked:
whether the corrected schedule lowers the plateau mean and its variance for
both backbones, and whether the two backbones remain indistinguishable once
the confound is gone.
