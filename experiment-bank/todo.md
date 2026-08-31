# Tasks

Use only these status values:

- `pending`: the task or experiment has not finished.
- `compeleted`: the experiment has ended and its idea and result records are updated.

| Task | Status | Idea | Result |
| --- | --- | --- | --- |
| Train the wheat50 P2PNet baseline with batch size 16 | compeleted | [Idea](ideas/wheat50-p2pnet-baseline-bs16.md) | [Result](experiments/wheat50-p2pnet-baseline-bs16.md) |
| Train P2PNet on wheat50 foreground images for 250 epochs | compeleted | [Idea](ideas/wheat50-foreground-p2pnet-bs16-e250.md) | [Result](experiments/wheat50-foreground-p2pnet-bs16-e250.md) |
| Evaluate non-overlapping Copy-Paste on wheat50 foreground images | compeleted | [Idea](ideas/wheat50-foreground-copypaste-bs16-e250.md) | [Result](experiments/wheat50-foreground-copypaste-bs16-e250.md) |
| Evaluate complete positive 768 crops on wheat50 foreground images | compeleted | [Idea](ideas/wheat50-foreground-positive-crop768-bs16-e250.md) | [Result](experiments/wheat50-foreground-positive-crop768-bs16-e250.md) |
| Visualize and compare Wheat50 P2PNet experiments | compeleted | [Idea](ideas/wheat50-model-comparison-visualization.md) | [Result](experiments/wheat50-model-comparison-visualization.md) |
| Train P2PNet on wheat50 with a frozen DINOv3 ConvNeXt-Base backbone | compeleted | [Idea](ideas/wheat50-dinov3-convnext-base-frozen-bs16-e250.md) | [Result](experiments/wheat50-dinov3-convnext-base-frozen-bs16-e250.md) |
| Train P2PNet on wheat50 with a fine-tuned DINOv3 ConvNeXt-Base backbone | compeleted | [Idea](ideas/wheat50-dinov3-convnext-base-finetune-bs16-e250.md) | [Result](experiments/wheat50-dinov3-convnext-base-finetune-bs16-e250.md) |
| Rerun VGG Copy-Paste and DINOv3 ConvNeXt-Base under the corrected optimizer and schedule | pending | [Idea](ideas/wheat50-cosine-wd-paired-rerun-bs16-e250.md) | [Result](experiments/wheat50-cosine-wd-paired-rerun-bs16-e250.md) |
