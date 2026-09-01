# Wheat50 Evaluation Threshold Selection

## Status

Completed on 2026-09-01 from code commit `e91fe0f`. This record supersedes the
metrics in every earlier experiment record, all of which were measured at a
threshold of 0.5.

## Configuration

See [the experiment idea](../ideas/wheat50-eval-threshold-selection.md).
Reproduce with `python tools/sweep_eval_threshold.py <experiment_name> ...`.

## Metrics

Threshold chosen on the 320 training images, metrics reported on the 81
validation images, all from `latest.pth`:

| Run | Threshold | Val MAE | Val MSE | MAE at 0.5 | Leaked optimum |
| --- | ---: | ---: | ---: | ---: | ---: |
| Foreground baseline | 0.50 | 6.235 | 10.083 | 6.235 | 6.222 |
| Positive 768 crops | 0.40 | 6.679 | 9.695 | 7.494 | 6.679 |
| Copy-Paste, original schedule | 0.60 | 4.160 | 5.649 | 4.062 | 4.012 |
| Copy-Paste, weight decay only | 0.70 | 4.605 | 6.651 | 6.025 | 4.407 |
| Copy-Paste, cosine only | 0.70 | 5.321 | 7.487 | 7.284 | 5.000 |
| Copy-Paste, corrected schedule | 0.65 | 5.198 | 7.096 | 6.753 | 5.198 |
| DINOv3 ConvNeXt-Base, frozen | 0.65 | 6.173 | 8.689 | 8.556 | 6.173 |
| DINOv3 ConvNeXt-Base, fine-tuned | 0.70 | 4.012 | 5.490 | 4.679 | 4.012 |
| DINOv3 ConvNeXt-Base, fine-tuned, corrected | 0.75 | 3.531 | 4.635 | 4.667 | 3.346 |

The last column is what the threshold would give if it were chosen on the
validation split, and is listed only to bound the leak. The training split
transfers to validation within 0.32 MAE on every run, so the selection is
sound and the leak it avoids was worth up to 0.32.

## Observations

- The best threshold rises with the training configuration, from 0.50 for the
  oldest baseline through 0.60 and 0.65 for the VGG16-BN Copy-Paste runs to
  0.75 for the corrected DINOv3 run. The fixed 0.5 sat at the optimum of the
  oldest baseline and drifted further from every later run.
- The correction is largest exactly where the earlier records reported
  regressions: 2.35 MAE for the frozen DINOv3 run, 1.96 for the cosine-only
  arm, 1.56 for the corrected Copy-Paste run.
- DINOv3 separates background far better than VGG16-BN at every threshold. At
  0.05 its fine-tuned runs overcount by 10 to 11 per image while the VGG16-BN
  arms overcount by 24 to 29, so its error curve is both lower and flatter.

## What survives the 1.5 noise floor

Two findings, both re-confirmed at the corrected working points:

- Non-overlapping Copy-Paste beats the foreground-only baseline, 4.160 against
  6.235, a gain of 2.08.
- Fine-tuning the DINOv3 backbone beats freezing it, 3.531 against 6.173, a
  gain of 2.64.

Nothing else does. In particular these differences are all below the floor and
are recorded as indistinguishable:

| Comparison | Difference |
| --- | ---: |
| Best DINOv3, 3.531, against best VGG16-BN, 4.160 | 0.63 |
| Copy-Paste original schedule against corrected | 1.04 |
| Copy-Paste original schedule against weight decay only | 0.45 |
| Copy-Paste original schedule against cosine only | 1.16 |
| DINOv3 fine-tuned, original schedule against corrected | 0.48 |

## Corrections to earlier records

Three conclusions in earlier records were threshold artefacts and do not hold:

- That a fine-tuned DINOv3 ConvNeXt-Base beats VGG16-BN by 1.78 MAE. At
  per-run thresholds the gap is 0.63 and the two are indistinguishable.
- That the corrected schedule cost VGG16-BN 1.79 MAE. At per-run thresholds it
  costs 1.04, below the floor.
- That both weight decay and the cosine decay individually hurt VGG16-BN by
  1.62 and 2.38. At per-run thresholds they cost 0.45 and 1.16, and the
  ablation attributes nothing.

## Outcome

The project has two established findings, Copy-Paste and fine-tuning over
freezing. Every backbone and schedule comparison made so far sits inside the
noise floor once the working point is chosen per run.

Two things follow. Future runs should report at a threshold selected on the
training split, and any comparison of two configurations at a shared threshold
should be treated as uninterpretable. And the floor itself deserves revisiting:
it was derived from plateau standard deviations of 0.37 to 1.47 under the old
schedule, while the corrected runs plateau at 0.23 to 0.37. A floor fixed after
these runs would be lower, but lowering it now would be choosing a threshold
for significance after seeing the results.
