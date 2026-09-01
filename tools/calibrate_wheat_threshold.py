"""Calibrate P2PNet count thresholds without using validation labels."""

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from crowd_datasets.COCO.coco_wheat_points import CocoWheatPoints
from models import build_model
from train import get_args_parser
from util.repository import get_shared_repository_root


def get_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Choose a score threshold on full training images and report "
            "metrics on full validation images."
        )
    )
    parser.add_argument("experiments", nargs="+")
    parser.add_argument("--checkpoint", default="latest.pth")
    parser.add_argument("--threshold-start", default=0.05, type=float)
    parser.add_argument("--threshold-stop", default=0.95, type=float)
    parser.add_argument("--threshold-step", default=0.05, type=float)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        default="logs/wheat50-threshold-calibration/results.json",
        help="JSON path, relative to the main worktree when not absolute",
    )
    parser.add_argument("--expected-train-images", default=320, type=int)
    parser.add_argument("--expected-val-images", default=81, type=int)
    return parser


def parse_run_args(shared_root, experiment):
    """Recover training arguments from the Namespace in a run log."""
    fields = vars(get_args_parser().parse_args([]))
    log_path = shared_root / "logs" / experiment / "run_log.txt"
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"Namespace\((.*?)\)(?:loss/|Start)", text, re.S)
    if match is None:
        raise ValueError(f"Could not parse training arguments from {log_path}")
    pattern = r"(\w+)=('[^']*'|\[[^\]]*\]|[^,]+?)(?:, (?=\w+=)|$)"
    for key, value in re.findall(pattern, match.group(1)):
        try:
            fields[key] = ast.literal_eval(value.strip())
        except (ValueError, SyntaxError):
            fields[key] = value.strip()
    return fields


def build_split(run, annotations, images):
    """Build a deterministic full-image split without training augmentation."""
    return CocoWheatPoints(
        data_root=run["data_root"],
        annotation_file=annotations,
        image_dir=images,
        category_name=run["coco_category_name"],
        train=False,
    )


def evaluate_grid(model, dataset, thresholds, device):
    counts = []
    truths = []
    with torch.inference_mode():
        for image, targets in dataset:
            outputs = model(image.unsqueeze(0).to(device))
            scores = torch.softmax(outputs["pred_logits"], -1)[0, :, 1]
            predicted = (scores.unsqueeze(0) > thresholds[:, None]).sum(1)
            counts.append(predicted.cpu().numpy())
            truths.append(targets[0]["point"].shape[0])

    errors = np.stack(counts) - np.asarray(truths)[:, None]
    return {
        "mae": np.abs(errors).mean(axis=0),
        "mse": np.sqrt(np.square(errors).mean(axis=0)),
    }


def metric_at(metrics, index):
    return {
        "mae": float(metrics["mae"][index]),
        "mse": float(metrics["mse"][index]),
    }


def main():
    args = get_parser().parse_args()
    shared_root = get_shared_repository_root(CODE_ROOT)
    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = shared_root / output_path

    threshold_values = np.round(
        np.arange(
            args.threshold_start,
            args.threshold_stop + args.threshold_step / 2,
            args.threshold_step,
        ),
        decimals=8,
    )
    if not len(threshold_values):
        raise ValueError("Threshold grid must not be empty")

    device = torch.device(args.device)
    thresholds = torch.tensor(
        threshold_values, device=device, dtype=torch.float32
    )
    results = {
        "selection_split": "train",
        "report_split": "val",
        "checkpoint": args.checkpoint,
        "thresholds": threshold_values.tolist(),
        "experiments": [],
    }

    for experiment in args.experiments:
        print(f"Evaluating {experiment}", flush=True)
        run = parse_run_args(shared_root, experiment)
        model_args = SimpleNamespace(
            backbone=run["backbone"],
            backbone_weights="",
            freeze_backbone=False,
            row=run["row"],
            line=run["line"],
            set_cost_class=run["set_cost_class"],
            set_cost_point=run["set_cost_point"],
            point_loss_coef=run["point_loss_coef"],
            eos_coef=run["eos_coef"],
        )
        model = build_model(model_args).to(device).eval()
        checkpoint_path = (
            shared_root / "checkpoints" / experiment / args.checkpoint
        )
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        model.load_state_dict(checkpoint["model"], strict=True)

        train_set = build_split(
            run, run["coco_train_annotations"], run["coco_train_images"]
        )
        val_set = build_split(
            run, run["coco_val_annotations"], run["coco_val_images"]
        )
        if len(train_set) != args.expected_train_images:
            raise ValueError(
                f"{experiment} has {len(train_set)} training images; "
                f"expected {args.expected_train_images}"
            )
        if len(val_set) != args.expected_val_images:
            raise ValueError(
                f"{experiment} has {len(val_set)} validation images; "
                f"expected {args.expected_val_images}"
            )

        train_metrics = evaluate_grid(model, train_set, thresholds, device)
        selected_index = int(train_metrics["mae"].argmin())
        val_metrics = evaluate_grid(model, val_set, thresholds, device)
        half_indices = np.flatnonzero(np.isclose(threshold_values, 0.5))
        half_index = int(half_indices[0]) if len(half_indices) else None
        validation_best_index = int(val_metrics["mae"].argmin())

        result = {
            "experiment": experiment,
            "backbone": run["backbone"],
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "train_images": len(train_set),
            "val_images": len(val_set),
            "selected_threshold": float(threshold_values[selected_index]),
            "train_selected": metric_at(train_metrics, selected_index),
            "val_selected": metric_at(val_metrics, selected_index),
            "val_at_0_5": (
                metric_at(val_metrics, half_index)
                if half_index is not None
                else None
            ),
            "validation_selected_diagnostic": {
                "threshold": float(threshold_values[validation_best_index]),
                **metric_at(val_metrics, validation_best_index),
            },
            "train_mae_by_threshold": train_metrics["mae"].tolist(),
            "val_mae_by_threshold": val_metrics["mae"].tolist(),
        }
        results["experiments"].append(result)
        print(
            f"  threshold={result['selected_threshold']:.2f}, "
            f"train MAE={result['train_selected']['mae']:.3f}, "
            f"val MAE={result['val_selected']['mae']:.3f}, "
            f"val MSE={result['val_selected']['mse']:.3f}",
            flush=True,
        )

        del checkpoint, model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
