# Wheat50 DINOv3 ConvNeXt-Base Frozen Backbone

## Goal

Establish whether a general-domain vision foundation model beats the 2014
ImageNet-supervised VGG16-BN currently used as the P2PNet backbone, while
keeping every other component of the Copy-Paste experiment fixed.

## Hypothesis

DINOv3 ConvNeXt-Base was distilled from ViT-7B on LVD-1689M, so its frozen
stage features should already separate wheat spikes from background better
than VGG16-BN does after fine-tuning on 320 images. With only 988 annotated
instances, freezing also removes the main overfitting risk of a 88M-parameter
backbone.

## Why this backbone rather than a ViT

The ConvNeXt variants are hierarchical, so stages 1-3 map onto the decoder
inputs at strides 4, 8 and 16 with no token reshaping, no patch-size padding
and no positional-encoding interpolation. The ViT variants would additionally
change the train/eval token grid from 37x37 to 74x74, which is a distribution
shift the convolutional baseline does not have. This experiment isolates the
backbone, not the adaptation machinery.

## Configuration

Identical to the non-overlapping Copy-Paste experiment except for the
backbone:

- Backbone: `dinov3_convnext_base`, LVD-1689M weights, frozen
- Decoder input channels: 128 / 256 / 512, from the ConvNeXt stage dimensions
- Trainable parameters: 6.8M (head, decoder), backbone excluded from the optimizer
- Dataset variant: foreground-only `wheat_2` images
- Training crop: 512 x 512, one crop per image
- Random scale: 0.8 to 1.2
- Horizontal flip probability: 0.5
- Copy-Paste probability: 1.0
- Maximum pasted instances per crop: 3
- Placement attempts per instance: 20
- Batch size: 16
- Head learning rate: 1e-4
- Weight decay: 1e-4
- Epochs: 250
- Learning-rate drops: epochs 250 and 350
- Matcher class/point costs: 1 / 0.05
- Point-loss coefficient: 0.0002
- Background coefficient: 0.5
- Anchor layout: 2 x 2
- Gradient clipping: 0.1
- Evaluation frequency: every 5 epochs
- Seed: 42

## Evaluation

The primary metric is the mean validation MAE over the last ten evaluations,
not the best MAE. Across the four finished experiments the best-of-N metric is
optimistic by 0.9 to 8.3 MAE, because it selects the luckiest of 49 noisy
evaluations on only 81 validation images whose mean count is 3.3.

Reference values on the same data, as plateau mean and standard deviation:

| Experiment | Best MAE | Last-10 mean | Std |
| --- | ---: | ---: | ---: |
| Foreground baseline | 5.802 | 7.957 | 1.471 |
| Positive 768 crops | 6.074 | 7.110 | 0.843 |
| Non-overlapping Copy-Paste | 3.889 | 4.772 | 0.580 |

The plateau standard deviations put the noise floor near 1.5 MAE. Differences
smaller than that are recorded as indistinguishable rather than as a result,
because this run uses a single seed and cannot separate training variance.
