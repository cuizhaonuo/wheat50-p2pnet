# Wheat50 Score-Threshold Calibration

## Goal

Remove the evaluation bias caused by P2PNet's hard-coded score threshold of
0.5. Earlier sweeps showed that the best threshold changes systematically with
the backbone and optimization configuration, so reporting every run at 0.5
does not compare equally calibrated classifiers.

## Design

For each completed Wheat50 experiment with a saved `latest.pth` checkpoint:

1. Run deterministic, full-image inference on all 320 training images and all
   81 validation images. Random crops, scaling, flipping, and Copy-Paste are
   disabled on both splits.
2. Evaluate the predeclared threshold grid from 0.05 through 0.95 in steps of
   0.05 from a single forward pass per image.
3. Select the threshold with the lowest training-set MAE. Ties resolve to the
   lower threshold by grid order.
4. Report MAE and MSE at that fixed threshold on the validation set. Validation
   results must not influence threshold selection.

The grid is unchanged from the preceding diagnostic sweep. It will not be
refined after inspecting validation performance.

## Scope

The calibration covers the foreground baseline, positive-crop, original
Copy-Paste, the two Copy-Paste optimizer/schedule ablations, the corrected
Copy-Paste run, and the frozen, old-configuration fine-tuned, and corrected
fine-tuned DINOv3 ConvNeXt-Base runs.

## Interpretation

These are end-of-training checkpoint metrics, not means over the last ten
evaluation epochs. They replace the earlier validation-selected threshold
diagnostic as the leakage-free threshold comparison. Historical MAE at the
fixed 0.5 threshold and plateau summaries remain useful diagnostics, but must
not be presented as calibrated model comparisons.
