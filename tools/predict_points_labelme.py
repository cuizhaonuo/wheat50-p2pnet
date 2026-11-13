"""Run a trained P2PNet over a folder of images and write labelme point files.

Each image is passed through the network once and the resulting scores are
thresholded several times, so sweeping a range of thresholds costs one forward
pass rather than one per threshold. Every threshold gets its own output tree:

    <output_root>/point_<threshold>/<split>/<stem>.jpg    the input image
    <output_root>/point_<threshold>/<split>/<stem>.json   labelme points
    <output_root>/point_<threshold>/<split>/<stem>_pred.jpg   points drawn on it

The plain image sits beside the annotation so labelme opens the pair directly,
and the drawn version is kept separate so the predictions never become part of
what an annotator sees while correcting them.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from torchvision.transforms import functional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_model

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
LABELME_VERSION = "5.5.0"


def get_args_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", required=True, type=Path,
                        help="directory holding one subdirectory per split")
    parser.add_argument("--splits", nargs="+", default=["train2017", "val2017"])
    parser.add_argument("--weight-path", required=True, type=Path)
    parser.add_argument("--backbone", default="dinov3_convnext_base")
    parser.add_argument("--row", default=2, type=int)
    parser.add_argument("--line", default=2, type=int)
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    parser.add_argument("--label", default="wheat_2")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--radius", default=3, type=int,
                        help="marker radius in the drawn image; the points sit "
                             "about 16 pixels apart, so a large marker merges them")
    parser.add_argument("--visualization-only", action="store_true",
                        help="redraw the markers without touching annotations that "
                             "may already have been corrected by hand")
    parser.add_argument("--device", default="cuda")
    return parser


def load_model(args):
    model_args = SimpleNamespace(
        backbone=args.backbone, backbone_weights="", freeze_backbone=False,
        row=args.row, line=args.line, set_cost_class=1.0, set_cost_point=0.05,
        point_loss_coef=0.0002, eos_coef=0.5)
    model = build_model(model_args).to(args.device).eval()
    checkpoint = torch.load(args.weight_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    return model


def labelme_document(points, image_name, height, width, label):
    return {
        "version": LABELME_VERSION,
        "flags": {},
        "shapes": [
            {
                "label": label,
                "points": [[round(float(x), 2), round(float(y), 2)]],
                "group_id": None,
                "description": "",
                "shape_type": "point",
                "flags": {},
            }
            for x, y in points
        ],
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": int(height),
        "imageWidth": int(width),
    }


def main():
    args = get_args_parser().parse_args()
    model = load_model(args)
    thresholds = sorted(args.thresholds)
    grid = torch.tensor(thresholds, device=args.device).view(-1, 1)
    counts = {threshold: [] for threshold in thresholds}

    for split in args.splits:
        paths = sorted(p for p in (args.image_root / split).iterdir()
                       if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        directories = {}
        for threshold in thresholds:
            directory = args.output_root / f"point_{threshold:g}" / split
            directory.mkdir(parents=True, exist_ok=True)
            directories[threshold] = directory
        print(f"{split}: {len(paths)} images -> {args.output_root}", flush=True)

        for index, path in enumerate(paths, 1):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Could not read image: {path}")
            height, width = image.shape[:2]
            tensor = functional.normalize(
                functional.to_tensor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
                IMAGENET_MEAN, IMAGENET_STD)

            with torch.inference_mode():
                outputs = model(tensor.unsqueeze(0).to(args.device))
            scores = torch.softmax(outputs["pred_logits"], -1)[0, :, 1]
            coordinates = outputs["pred_points"][0]
            keep = scores.unsqueeze(0) > grid

            for row, threshold in enumerate(thresholds):
                selected = coordinates[keep[row]].cpu().numpy()
                counts[threshold].append(len(selected))
                directory = directories[threshold]

                image_name = path.with_suffix(".jpg").name
                if not args.visualization_only:
                    target = directory / image_name
                    if not target.exists():
                        cv2.imwrite(str(target), image,
                                    [cv2.IMWRITE_JPEG_QUALITY, 95])

                    document = labelme_document(
                        selected, image_name, height, width, args.label)
                    (directory / f"{path.stem}.json").write_text(
                        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")

                drawn = image.copy()
                for x, y in selected:
                    center = (int(round(float(x))), int(round(float(y))))
                    # a hollow ring with a white halo stays readable on the black
                    # background and on the spikes without hiding either
                    cv2.circle(drawn, center, args.radius + 1, (255, 255, 255), 1,
                               cv2.LINE_AA)
                    cv2.circle(drawn, center, args.radius, (0, 0, 255), 1,
                               cv2.LINE_AA)
                cv2.putText(drawn, f"{threshold:g}: {len(selected)}", (12, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
                cv2.imwrite(str(directory / f"{path.stem}_pred.jpg"), drawn,
                            [cv2.IMWRITE_JPEG_QUALITY, 90])

            if index % 50 == 0:
                print(f"  {index}/{len(paths)}", flush=True)

    print("\nthreshold  images  total points  mean per image")
    for threshold in thresholds:
        values = np.array(counts[threshold])
        print(f"{threshold:9g}  {len(values):6d}  {values.sum():12d}  "
              f"{values.mean():14.2f}")


if __name__ == "__main__":
    main()
