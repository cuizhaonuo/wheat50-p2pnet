# Wheat50 Copy-Paste Schedule Ablation

## Status

Completed on 2026-09-01 from code commit `1b918b2`. Read together with
[the threshold selection record](wheat50-eval-threshold-selection.md), which
supersedes the conclusion below.

## Configuration

See [the experiment idea](../ideas/wheat50-copypaste-schedule-ablation-bs16-e250.md).

## Metrics

At the shared 0.5 threshold the ablation was run under, as mean validation MAE
over the last ten evaluations:

| Arm | Weight decay | Warmup | Schedule | MAE | Std |
| --- | ---: | ---: | --- | ---: | ---: |
| Original Copy-Paste | 0 | 0 | constant | 4.7719 | 0.5800 |
| Weight decay only | 0.05 | 0 | constant | 6.3961 | 0.8586 |
| Cosine only | 0 | 5 | cosine | 7.1471 | 0.4049 |
| Corrected, both | 0.05 | 5 | cosine | 6.5634 | 0.2276 |

At thresholds chosen on the training split, from `latest.pth`:

| Arm | Threshold | Val MAE | Val MSE |
| --- | ---: | ---: | ---: |
| Original Copy-Paste | 0.60 | 4.160 | 5.649 |
| Weight decay only | 0.70 | 4.605 | 6.651 |
| Cosine only | 0.70 | 5.321 | 7.487 |
| Corrected, both | 0.65 | 5.198 | 7.096 |

Training times were 1:33:02 and 1:44:59 for the two new arms.

## Observations

- Read at the shared threshold, both factors appeared to hurt: 1.62 for weight
  decay and 2.38 for the cosine decay, both past the noise floor. That reading
  was wrong. Each arm's own best threshold is higher than the original run's,
  0.70 against 0.60, so a shared 0.5 penalised them and not the original.
- At per-arm thresholds the costs fall to 0.45 and 1.16, both below the floor.
  The ablation therefore attributes nothing: no factor is shown to hurt.
- The prediction that weight decay would be near-neutral was right in the end,
  but for the wrong reason. The arithmetic offered in the idea, a 2.5 per cent
  total shrink over 5000 steps, treats decay as one-way rather than as an
  equilibrium against gradient noise, and it ignores that decaying the
  pre-BatchNorm convolutions of VGG16-BN leaves the function unchanged while
  raising the effective learning rate. Neither correction was tested here.
- The one effect that is real and unambiguous is on stability. Plateau standard
  deviation fell from 0.5800 to 0.2276 with both changes, and to 0.4049 with
  the cosine decay alone. Weight decay alone raised it, to 0.8586.

## Outcome

The regression the ablation set out to explain was an artefact of comparing
runs at a threshold that suited one of them. No attribution is possible and
none is needed. What remains usable is that the cosine decay tightens the
plateau, which makes the plateau metric itself more reliable, at no measurable
cost in accuracy.
