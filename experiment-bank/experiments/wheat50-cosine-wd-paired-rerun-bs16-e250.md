# Wheat50 Cosine Schedule and Weight Decay, Paired Rerun

## Status

Completed on 2026-09-01 from code commit `1b918b2`.

## Configuration

See [the experiment idea](../ideas/wheat50-cosine-wd-paired-rerun-bs16-e250.md).

Artifacts are stored under `wheat50-foreground-copypaste-cosine-wd-bs16-e250`
and `wheat50-dinov3-convnext-base-finetune-cosine-wd-bs16-e250` in the main
worktree's `logs`, `runs` and `checkpoints` directories.

## Metrics

The two-by-two, as mean validation MAE over the last ten evaluations:

| Backbone | Old configuration | Corrected configuration | Change |
| --- | ---: | ---: | ---: |
| VGG16-BN Copy-Paste | 4.7719 +/- 0.5800 | 6.5634 +/- 0.2276 | +1.7915 |
| DINOv3 ConvNeXt-Base fine-tuned | 5.5593 +/- 1.1892 | 4.7796 +/- 0.3730 | -0.7797 |

VGG16-BN under the corrected configuration:

- Mean validation MAE over the last ten evaluations: 6.5634, standard deviation 0.2276
- Mean validation MSE over the last ten evaluations: 9.2814
- Best validation MAE: 5.5309 at epoch 190, MSE 8.0890 at that epoch
- Final training loss: 0.01362, classification 0.00532, weighted point 0.00831
- Final unscaled point loss: 41.5339, ranging from 2.8705 to 48.6798
- Training time: 1:39:43

DINOv3 ConvNeXt-Base under the corrected configuration:

- Mean validation MAE over the last ten evaluations: 4.7796, standard deviation 0.3730
- Mean validation MSE over the last ten evaluations: 6.4484
- Best validation MAE: 3.6296 at epoch 110, MSE 5.1973 at that epoch
- Best validation MSE: 4.8419 at epoch 130
- Final training loss: 0.01255, classification 0.00372, weighted point 0.00883
- Final unscaled point loss: 44.1292, ranging from 5.6493 to 53.9599
- Training time: 2:10:27

## Observations

- The prediction that the schedule would stabilize the plateau held for both
  backbones. The standard deviation fell from 0.5800 to 0.2276 for VGG16-BN
  and from 1.1892 to 0.3730 for DINOv3, a reduction of 61 and 69 per cent.
- The prediction about the plateau mean held for DINOv3 and failed for
  VGG16-BN, which got worse by 1.7915, beyond the 1.5 noise floor.
- VGG16-BN did not undertrain. Its final training loss, 0.01362, and its mean
  over the last twenty epochs, 0.01364, match the old run's 0.01329 and
  0.01364 exactly. Training loss held while validation MAE rose by 1.79, so
  the corrected configuration moved it to a different and worse optimum rather
  than to an unfinished one.
- DINOv3 improved on both sides, reaching a lower final training loss, 0.01255
  against 0.01352, and a lower plateau.
- Under the corrected configuration the two backbones differ by 1.7834 MAE in
  DINOv3's favour, which passes the 1.5 floor fixed before the runs. Under the
  defective configuration they differed by 0.7874 and were indistinguishable.
- No absolute improvement was produced. The best plateau in the project is
  still 4.7719 for the old VGG Copy-Paste run, which the corrected DINOv3 run
  ties at 4.7796.

## Limitation

Three changes were bundled: weight decay reaching the optimizer, five epochs
of warmup, and the cosine decay. The VGG16-BN regression cannot be attributed
to any one of them from these runs. The decay term is the least likely
candidate on arithmetic alone, since at 5000 steps its total effect is a 2.5
per cent shrink on the head group and 0.25 per cent on the backbone group.

## Outcome

The comparison reverses once the training-configuration confound is removed. A
fine-tuned DINOv3 ConvNeXt-Base is better than VGG16-BN by 1.78 MAE under a
schedule both share, where under the old configuration the two were
indistinguishable. The earlier conclusion, that a general-domain foundation
backbone does not measurably beat VGG16-BN, was an artefact of a schedule that
happened to suit VGG16-BN.

The next step is an ablation on VGG16-BN, separating the cosine decay from the
weight decay, since the corrected schedule costs that backbone 1.79 MAE and
the reason is unknown.

## Superseded metrics

The MAE and MSE above were measured at the hard-coded 0.5 threshold. See
[the threshold selection record](wheat50-eval-threshold-selection.md) for the
numbers at a threshold chosen per run, and for which conclusions survive.
