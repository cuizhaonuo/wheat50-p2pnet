# Wheat50 Copy-Paste Schedule Ablation

## Goal

Attribute the 1.79 MAE that the corrected training configuration cost
VGG16-BN. The corrected run changed three things at once, so the paired rerun
established that the regression exists without saying which change caused it.

## Design

Each arm changes exactly one factor relative to the original Copy-Paste run,
rather than removing one factor from the corrected run, so every arm has the
same reference point:

| Arm | Weight decay | Warmup | Schedule | MAE |
| --- | ---: | ---: | --- | ---: |
| Original Copy-Paste | 0 | 0 | constant | 4.7719 +/- 0.5800 |
| Weight decay only | 0.05 | 0 | constant | this run |
| Cosine only | 0 | 5 | cosine | this run |
| Corrected, both | 0.05 | 5 | cosine | 6.5634 +/- 0.2276 |

The constant-rate arms keep the original milestones at epochs 250 and 350,
which sit past the last epoch and therefore never fire, reproducing the
original schedule exactly.

Warmup stays bundled with the cosine arm. Five epochs out of 250 is a small
residual confound, and warmup is part of what "fixing the schedule" means, so
splitting it into a fifth arm would cost 1.6 hours for little information.

## Hypothesis

Weight decay is the unlikely candidate. AdamW shrinks weights by
`1 - lr * weight_decay` per step, so across 5000 steps 0.05 amounts to a 2.5
per cent shrink on the head group and 0.25 per cent on the backbone group at
its 1e-5 rate. If the weight-decay arm alone reproduces most of the 1.79, that
arithmetic is wrong somewhere and the finding matters more than the ablation.

The cosine arm is the expected cause. VGG16-BN kept its training loss when the
corrected schedule was applied, 0.01362 against 0.01329, while validation MAE
rose, so the schedule moved it to a different optimum of equal training fit
and worse generalization.

## Why this matters beyond VGG16-BN

If annealing to zero is what hurts VGG16-BN, the DINOv3 result under the same
schedule, 4.7796, is measured under a schedule that is not neutral, and its
own headroom is unmeasured. The backbone comparison stands either way, since
both were run under the same configuration, but the absolute numbers would
still be leaving something on the table.

## Evaluation

Mean validation MAE over the last ten evaluations, with the 1.5 noise floor.
Note that the corrected runs plateau far more tightly, at 0.23 to 0.37 rather
than 0.58 to 1.19, so a floor derived from the newer runs would be lower. The
floor stays at 1.5 because it was fixed before any of these runs; a tighter
one can be adopted for future experiments, not applied retroactively to make a
smaller difference count.
