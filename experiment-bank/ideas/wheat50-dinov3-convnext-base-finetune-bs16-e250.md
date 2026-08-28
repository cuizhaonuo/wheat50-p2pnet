# Wheat50 DINOv3 ConvNeXt-Base Fine-tuned Backbone

## Goal

Separate two explanations for whatever the frozen DINOv3 ConvNeXt-Base run
produces: the quality of the pretrained features, and the capacity of the
6.8M-parameter detection head sitting on top of them.

## Hypothesis

If the frozen run underperforms VGG16-BN, fine-tuning tells us whether the
DINOv3 features are unsuitable for point localisation or whether the frozen
protocol simply starves the head. If the frozen run already wins, fine-tuning
shows how much of the remaining headroom 320 training images can actually
buy before overfitting.

## Configuration

Identical to the frozen run except:

- Backbone: `dinov3_convnext_base`, LVD-1689M weights, trained
- Backbone learning rate: 1e-5, the value the VGG experiments use
- Trainable parameters: 66.9M

## Known limitations

The training script applies no weight decay, because `args.weight_decay` is
never passed to the optimizer, and its learning-rate milestones sit past the
last epoch, so the rate is constant and unwarmed. Both are inherited from the
VGG experiments and are kept here so the comparison stays paired. They are
the main suspects if this run overfits, and fixing them belongs in a separate
change that re-runs the VGG baseline as well.

## Evaluation

Same protocol as the frozen run: mean validation MAE over the last ten
evaluations, with 1.5 MAE as the noise floor below which differences are
recorded as indistinguishable.
