"""Compare Wheat50 experiments and visualize per-spike counting behavior."""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import matplotlib
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torchvision.transforms import functional as transform_functional

matplotlib.use("Agg")
from matplotlib import pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import build_model
from util.repository import get_shared_repository_root


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SCALAR_TAGS = (
    ("loss/loss", "Total loss"),
    ("loss/loss_ce", "Classification loss"),
    ("loss/loss_point", "Weighted point loss"),
    ("metric/mae", "Validation MAE"),
    ("metric/mse", "Validation MSE"),
)
EXPERIMENTS = (
    {
        "key": "original-baseline",
        "label": "Original-image baseline",
        "run": "wheat50-p2pnet-baseline-bs16",
        "color": "#4c78a8",
    },
    {
        "key": "foreground-baseline",
        "label": "Foreground baseline",
        "run": "wheat50-foreground-p2pnet-bs16-e250",
        "color": "#f58518",
    },
    {
        "key": "copy-paste",
        "label": "Non-overlap Copy-Paste",
        "run": "wheat50-foreground-copypaste-bs16-e250",
        "color": "#54a24b",
    },
    {
        "key": "positive-crop-768",
        "label": "Positive crop 768",
        "run": "wheat50-foreground-positive-crop768-bs16-e250",
        "color": "#e45756",
    },
)


def get_args_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--category-name", default="wheat_2")
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--eval-frequency", default=5, type=int)
    parser.add_argument("--top-k", default=3, type=int)
    parser.add_argument("--gpu-id", default=0, type=int)
    parser.add_argument(
        "--output-dir",
        default=Path("figures/wheat50-model-comparison"),
        type=Path,
        help="output directory relative to the main worktree",
    )
    return parser


def resolve_output_directory(path):
    root = get_shared_repository_root(__file__)
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def exponential_smooth(values, weight=0.85):
    """Return a TensorBoard-style exponentially smoothed series."""
    if not values:
        return []
    smoothed = [float(values[0])]
    for value in values[1:]:
        smoothed.append(weight * smoothed[-1] + (1 - weight) * float(value))
    return smoothed


def load_tensorboard_scalars(repository_root):
    all_scalars = {}
    for experiment in EXPERIMENTS:
        run_directory = repository_root / "runs" / experiment["run"]
        accumulator = EventAccumulator(
            str(run_directory), size_guidance={"scalars": 0}
        )
        accumulator.Reload()
        available_tags = set(accumulator.Tags().get("scalars", []))
        missing = [tag for tag, _ in SCALAR_TAGS if tag not in available_tags]
        if missing:
            raise ValueError(
                f"Missing TensorBoard tags in {run_directory}: {missing}"
            )
        all_scalars[experiment["key"]] = {
            tag: accumulator.Scalars(tag) for tag, _ in SCALAR_TAGS
        }
    return all_scalars


def plot_tensorboard_comparison(all_scalars, output_path, eval_frequency):
    """Put every tracked TensorBoard scalar for all experiments in one figure."""
    figure, axes = plt.subplots(2, 3, figsize=(20, 11), constrained_layout=True)
    flat_axes = axes.ravel()
    for axis, (tag, title) in zip(flat_axes, SCALAR_TAGS):
        is_metric = tag.startswith("metric/")
        for experiment in EXPERIMENTS:
            events = all_scalars[experiment["key"]][tag]
            steps = np.asarray([event.step for event in events], dtype=float)
            values = np.asarray([event.value for event in events], dtype=float)
            if is_metric:
                steps = (steps + 1) * eval_frequency
                axis.plot(
                    steps,
                    values,
                    color=experiment["color"],
                    linewidth=1.8,
                    marker="o",
                    markersize=2.6,
                    label=experiment["label"],
                )
            else:
                axis.plot(
                    steps,
                    values,
                    color=experiment["color"],
                    linewidth=0.8,
                    alpha=0.22,
                )
                axis.plot(
                    steps,
                    exponential_smooth(values.tolist()),
                    color=experiment["color"],
                    linewidth=2,
                    label=experiment["label"],
                )
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(title)
        axis.grid(alpha=0.25)

    summary_axis = flat_axes[-1]
    summary_axis.axis("off")
    lines = ["Best values from TensorBoard", ""]
    for experiment in EXPERIMENTS:
        scalars = all_scalars[experiment["key"]]
        best_mae = min(event.value for event in scalars["metric/mae"])
        best_mse = min(event.value for event in scalars["metric/mse"])
        lines.append(
            f"{experiment['label']}:  MAE {best_mae:.4f}  |  MSE {best_mse:.4f}"
        )
    summary_axis.text(
        0.02,
        0.95,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=12,
        family="monospace",
    )
    handles, labels = flat_axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    figure.suptitle(
        "Wheat50 P2PNet TensorBoard comparison\n"
        "Loss curves: raw (faint) and exponential smoothing (solid)",
        fontsize=17,
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def load_coco_records(data_root, split, category_name):
    annotation_path = (
        data_root / "annotations" / f"instances_{split}2017.json"
    )
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = [
        category
        for category in coco["categories"]
        if category["name"] == category_name
    ]
    if len(categories) != 1:
        raise ValueError(
            f"Expected one category named {category_name!r}, found {len(categories)}"
        )
    category_id = categories[0]["id"]
    annotations_by_image = defaultdict(list)
    ignored_empty_annotations = []
    for annotation in coco["annotations"]:
        if annotation["category_id"] != category_id:
            continue
        if not (annotation.get("points") or []):
            ignored_empty_annotations.append(annotation["id"])
            continue
        annotations_by_image[annotation["image_id"]].append(annotation)
    image_root = data_root / f"{split}2017"
    records = []
    for image in coco["images"]:
        records.append(
            {
                **image,
                "path": image_root / image["file_name"],
                "annotations": annotations_by_image[image["id"]],
            }
        )
    return records, ignored_empty_annotations


def annotation_mask(annotation, height, width):
    segmentation = annotation.get("segmentation")
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        encoded = mask_utils.merge(rles)
    elif isinstance(segmentation, dict) and isinstance(
        segmentation.get("counts"), list
    ):
        encoded = mask_utils.frPyObjects(segmentation, height, width)
    else:
        encoded = segmentation
    mask = mask_utils.decode(encoded)
    if mask.ndim == 3:
        mask = np.any(mask, axis=2)
    return mask.astype(bool)


def prepare_instances(record):
    instances = []
    for annotation in record["annotations"]:
        points = np.asarray(annotation.get("points") or [], dtype=np.float32)
        points = points.reshape(-1, 2)
        if len(points):
            center = points.mean(axis=0)
        else:
            x, y, width, height = annotation["bbox"]
            center = np.asarray([x + width / 2, y + height / 2])
        instances.append(
            {
                "annotation_id": annotation["id"],
                "points": points,
                "mask": annotation_mask(
                    annotation, record["height"], record["width"]
                ),
                "center": center,
            }
        )
    return instances


def load_inference_model(checkpoint_path, device):
    args = SimpleNamespace(backbone="vgg16_bn", row=2, line=2)
    model = build_model(args)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


def load_image_tensor(path, device):
    with Image.open(path) as image:
        image = image.convert("RGB")
        tensor = transform_functional.to_tensor(image)
    tensor = transform_functional.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
    return tensor.unsqueeze(0).to(device)


@torch.inference_mode()
def predict_points(model, image_path, device, threshold):
    outputs = model(load_image_tensor(image_path, device))
    scores = torch.softmax(outputs["pred_logits"], dim=-1)[0, :, 1]
    keep = scores > threshold
    return (
        outputs["pred_points"][0, keep].cpu().numpy(),
        scores[keep].cpu().numpy(),
    )


def assign_predictions_to_instances(points, instances, width, height):
    """Assign predictions inside GT masks; resolve overlaps by nearest center."""
    counts = np.zeros(len(instances), dtype=np.int64)
    unassigned = 0
    if not len(points):
        return counts, unassigned
    for point in points:
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        if x < 0 or x >= width or y < 0 or y >= height:
            unassigned += 1
            continue
        candidates = [
            index for index, instance in enumerate(instances) if instance["mask"][y, x]
        ]
        if not candidates:
            unassigned += 1
            continue
        if len(candidates) > 1:
            candidates = [
                min(
                    candidates,
                    key=lambda index: np.linalg.norm(
                        point - instances[index]["center"]
                    ),
                )
            ]
        counts[candidates[0]] += 1
    return counts, unassigned


def calculate_summary(rows, image_errors, total_predictions, unassigned):
    true_counts = np.asarray([row["true_count"] for row in rows], dtype=float)
    predicted_counts = np.asarray(
        [row["predicted_count"] for row in rows], dtype=float
    )
    errors = predicted_counts - true_counts
    correlation = (
        float(np.corrcoef(true_counts, predicted_counts)[0, 1])
        if len(true_counts) > 1
        and np.std(true_counts) > 0
        and np.std(predicted_counts) > 0
        else None
    )
    return {
        "num_spikes": int(len(rows)),
        "count_MAE_spike": float(np.mean(np.abs(errors))),
        "count_RMSE_spike": float(np.sqrt(np.mean(np.square(errors)))),
        "pearson_r": correlation,
        "image_MAE": float(np.mean(np.abs(image_errors))),
        "image_RMSE": float(np.sqrt(np.mean(np.square(image_errors)))),
        "total_predictions": int(total_predictions),
        "unassigned_predictions": int(unassigned),
        "unassigned_fraction": (
            float(unassigned / total_predictions) if total_predictions else 0.0
        ),
    }


def run_inference(records, repository_root, device, threshold, top_image_ids):
    all_rows = {}
    summaries = {}
    top_predictions = defaultdict(dict)
    for experiment in EXPERIMENTS:
        checkpoint_path = (
            repository_root
            / "checkpoints"
            / experiment["run"]
            / "best_mae.pth"
        )
        print(f"Loading {experiment['label']} from {checkpoint_path}", flush=True)
        model = load_inference_model(checkpoint_path, device)
        rows = []
        image_errors = []
        total_predictions = 0
        total_unassigned = 0
        for index, record in enumerate(records, start=1):
            instances = prepare_instances(record)
            points, scores = predict_points(
                model, record["path"], device, threshold
            )
            predicted_counts, unassigned = assign_predictions_to_instances(
                points, instances, record["width"], record["height"]
            )
            total_predictions += len(points)
            total_unassigned += unassigned
            true_total = sum(len(instance["points"]) for instance in instances)
            image_errors.append(len(points) - true_total)
            for instance, predicted_count in zip(instances, predicted_counts):
                rows.append(
                    {
                        "experiment": experiment["key"],
                        "image_id": record["id"],
                        "file_name": record["file_name"],
                        "annotation_id": instance["annotation_id"],
                        "true_count": len(instance["points"]),
                        "predicted_count": int(predicted_count),
                        "absolute_error": abs(
                            int(predicted_count) - len(instance["points"])
                        ),
                    }
                )
            if record["id"] in top_image_ids:
                top_predictions[record["id"]][experiment["key"]] = {
                    "points": points,
                    "scores": scores,
                    "total": len(points),
                    "unassigned": unassigned,
                }
            if index % 20 == 0 or index == len(records):
                print(
                    f"  {experiment['label']}: {index}/{len(records)} images",
                    flush=True,
                )
        all_rows[experiment["key"]] = rows
        summaries[experiment["key"]] = calculate_summary(
            rows,
            np.asarray(image_errors, dtype=float),
            total_predictions,
            total_unassigned,
        )
        del model
        torch.cuda.empty_cache()
    return all_rows, summaries, top_predictions


def plot_spike_scatter(all_rows, summaries, output_path):
    all_counts = [
        value
        for rows in all_rows.values()
        for row in rows
        for value in (row["true_count"], row["predicted_count"])
    ]
    limit = max(all_counts) + 2
    figure, axes = plt.subplots(2, 2, figsize=(13, 12), constrained_layout=True)
    for axis, experiment in zip(axes.ravel(), EXPERIMENTS):
        rows = all_rows[experiment["key"]]
        true_counts = np.asarray([row["true_count"] for row in rows])
        predicted_counts = np.asarray([row["predicted_count"] for row in rows])
        axis.scatter(
            true_counts,
            predicted_counts,
            color=experiment["color"],
            alpha=0.42,
            s=28,
            edgecolors="none",
        )
        axis.plot([0, limit], [0, limit], "--", color="#333333", linewidth=1.2)
        axis.set_xlim(0, limit)
        axis.set_ylim(0, limit)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("Ground-truth spikelet count per spike")
        axis.set_ylabel("Predicted point count inside GT spike mask")
        summary = summaries[experiment["key"]]
        correlation = summary["pearson_r"]
        correlation_text = "n/a" if correlation is None else f"{correlation:.3f}"
        axis.set_title(
            f"{experiment['label']}\n"
            f"count_MAE_spike={summary['count_MAE_spike']:.3f}, "
            f"RMSE={summary['count_RMSE_spike']:.3f}, r={correlation_text}"
        )
        axis.grid(alpha=0.2)
    figure.suptitle(
        "Per-spike count agreement on the foreground validation set\n"
        "Predicted points are assigned using ground-truth instance masks",
        fontsize=16,
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def draw_overlay(record, predictions, label):
    image = cv2.imread(str(record["path"]), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(record["path"])
    instances = prepare_instances(record)
    for instance in instances:
        contours, _ = cv2.findContours(
            instance["mask"].astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(image, contours, -1, (255, 255, 0), 1)
        for x, y in instance["points"]:
            cv2.circle(
                image,
                (int(round(x)), int(round(y))),
                5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
    for x, y in predictions["points"]:
        center = (int(round(x)), int(round(y)))
        cv2.circle(
            image,
            center,
            3,
            (0, 220, 0),
            -1,
            cv2.LINE_AA,
        )
    true_total = sum(len(instance["points"]) for instance in instances)
    header = np.zeros((76, image.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        header,
        label,
        (18, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        header,
        f"spikes={len(instances)}  GT points={true_total}  "
        f"pred={predictions['total']}  outside masks={predictions['unassigned']}",
        (18, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([header, image])


def render_top_images(records, top_records, predictions, output_directory):
    top_directory = output_directory / "top-spike-images"
    top_directory.mkdir(parents=True, exist_ok=True)
    for rank, record in enumerate(top_records, start=1):
        panels = []
        stem = f"rank{rank:02d}_image{record['id']}"
        for experiment in EXPERIMENTS:
            panel = draw_overlay(
                record,
                predictions[record["id"]][experiment["key"]],
                experiment["label"],
            )
            panels.append(panel)
            cv2.imwrite(
                str(top_directory / f"{stem}_{experiment['key']}.png"), panel
            )
        composite = np.vstack(
            [np.hstack(panels[:2]), np.hstack(panels[2:])]
        )
        cv2.imwrite(str(top_directory / f"{stem}_comparison.png"), composite)


def write_csv(all_rows, output_path):
    fieldnames = (
        "experiment",
        "image_id",
        "file_name",
        "annotation_id",
        "true_count",
        "predicted_count",
        "absolute_error",
    )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for experiment in EXPERIMENTS:
            writer.writerows(all_rows[experiment["key"]])


def main():
    args = get_args_parser().parse_args()
    if not 0 <= args.threshold <= 1:
        raise ValueError("threshold must be between zero and one")
    if args.top_k <= 0:
        raise ValueError("top-k must be greater than zero")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for experiment inference")
    device = torch.device("cuda")

    repository_root = get_shared_repository_root(__file__)
    data_root = args.data_root.expanduser().resolve()
    output_directory = resolve_output_directory(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    print("Loading TensorBoard events", flush=True)
    all_scalars = load_tensorboard_scalars(repository_root)
    plot_tensorboard_comparison(
        all_scalars,
        output_directory / "tensorboard-comparison.png",
        args.eval_frequency,
    )

    records, ignored_empty_annotations = load_coco_records(
        data_root, args.split, args.category_name
    )
    top_records = sorted(
        records,
        key=lambda record: (
            -len(record["annotations"]),
            -sum(len(annotation.get("points") or []) for annotation in record["annotations"]),
            record["file_name"],
        ),
    )[: args.top_k]
    top_image_ids = {record["id"] for record in top_records}
    all_rows, summaries, top_predictions = run_inference(
        records, repository_root, device, args.threshold, top_image_ids
    )
    plot_spike_scatter(
        all_rows, summaries, output_directory / "count-mae-spike-scatter.png"
    )
    render_top_images(records, top_records, top_predictions, output_directory)
    write_csv(all_rows, output_directory / "per-spike-counts.csv")

    result = {
        "dataset_split": args.split,
        "score_threshold": args.threshold,
        "assignment": "prediction center inside GT instance mask",
        "ignored_empty_annotation_ids": ignored_empty_annotations,
        "summaries": summaries,
        "top_images": [
            {
                "rank": rank,
                "image_id": record["id"],
                "file_name": record["file_name"],
                "num_spikes": len(record["annotations"]),
                "num_spikelets": sum(
                    len(annotation.get("points") or [])
                    for annotation in record["annotations"]
                ),
            }
            for rank, record in enumerate(top_records, start=1)
        ],
    }
    (output_directory / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
