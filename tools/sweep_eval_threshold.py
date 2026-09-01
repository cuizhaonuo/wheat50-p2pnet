"""Choose the evaluation score threshold on the training split.

The counting metric turns tens of thousands of anchor points into a count by
keeping those whose class-1 probability exceeds a threshold, and that threshold
was long fixed at 0.5. Its best value moves with the training configuration, so
comparing runs at a shared 0.5 compares them at working points that suit some
of them better than others.

This picks the threshold on the training images, evaluated the way validation
images are, and reports the validation metrics there. Selecting it on the split
it is reported on would leak, in the same way that reporting the best MAE over
all epochs does.

Usage:
    python tools/sweep_eval_threshold.py <experiment_name> [<experiment_name> ...]
"""
import argparse, ast, os, re, sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crowd_datasets.COCO.coco_wheat_points import CocoWheatPoints
from models import build_model
from util.repository import get_shared_repository_root
from train import get_args_parser

ROOT = get_shared_repository_root(__file__)
GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)


def parse_run_args(name):
    """Recover the training arguments from the Namespace echoed into run_log.txt."""
    # older runs predate some arguments, so start from the current defaults
    fields = vars(get_args_parser().parse_args([]))
    text = open(f'{ROOT}/logs/{name}/run_log.txt', errors='ignore').read()
    body = re.search(r'Namespace\((.*?)\)(?:loss/|Start)', text, re.S).group(1)
    for key, value in re.findall(r"(\w+)=('[^']*'|\[[^\]]*\]|[^,]+?)(?:, (?=\w+=)|$)", body):
        try:
            fields[key] = ast.literal_eval(value.strip())
        except (ValueError, SyntaxError):
            fields[key] = value.strip()
    return fields


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiments', nargs='+')
    parser.add_argument('--checkpoint', default='latest.pth')
    args = parser.parse_args()

    rows = {}
    for name in args.experiments:
        run = parse_run_args(name)
        model_args = SimpleNamespace(
            backbone=run['backbone'], row=run['row'], line=run['line'],
            backbone_weights='', freeze_backbone=False,
            set_cost_class=run['set_cost_class'], set_cost_point=run['set_cost_point'],
            point_loss_coef=run['point_loss_coef'], eos_coef=run['eos_coef'])
        model = build_model(model_args).cuda().eval()
        state = torch.load(f'{ROOT}/checkpoints/{name}/{args.checkpoint}',
                           map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'])

        # the threshold is chosen on the training images and reported on the
        # validation ones; selecting it on the same split it is reported on
        # leaks, the way best-of-N over epochs does
        def split(annotations, images):
            return CocoWheatPoints(
                data_root=run['data_root'], annotation_file=annotations,
                image_dir=images, category_name=run['coco_category_name'],
                train=False)

        train_set = split(run['coco_train_annotations'], run['coco_train_images'])
        val_set = split(run['coco_val_annotations'], run['coco_val_images'])

        grid = torch.tensor(GRID, device='cuda').view(-1, 1)

        def sweep(dataset):
            counts, truths = [], []
            with torch.no_grad():
                for image, target in dataset:
                    out = model(image.unsqueeze(0).cuda())
                    scores = torch.softmax(out['pred_logits'], -1)[0, :, 1]
                    counts.append((scores.unsqueeze(0) > grid).sum(1).cpu().numpy())
                    truths.append(target[0]['point'].shape[0])
            error = np.stack(counts) - np.array(truths)[:, None]
            return np.abs(error).mean(0), np.sqrt((error ** 2).mean(0))

        train_mae, _ = sweep(train_set)
        val_mae, val_mse = sweep(val_set)
        rows[name] = (train_mae, val_mae, val_mse)
        del model
        torch.cuda.empty_cache()

    half = list(GRID).index(0.5)
    print(f"\n{'实验':50s} {'阈值':>5s} {'验证 MAE':>9s} {'验证 MSE':>9s} "
          f"{'@0.5 MAE':>9s} {'泄漏最优':>9s}")
    for name, (train_mae, val_mae, val_mse) in rows.items():
        picked = int(train_mae.argmin())
        print(f"{name:50s} {GRID[picked]:5.2f} {val_mae[picked]:9.3f} {val_mse[picked]:9.3f} "
              f"{val_mae[half]:9.3f} {val_mae.min():9.3f}")


if __name__ == '__main__':
    main()
