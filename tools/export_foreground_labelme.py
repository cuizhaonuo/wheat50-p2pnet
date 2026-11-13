"""Export one COCO category's foreground as a labelme dataset.

Every image keeps only the pixels inside its ``wheat_2`` instances and the rest
of the frame is painted black, so the network sees a spike on an empty
background instead of the surrounding canopy. The COCO annotations follow the
pixels: the other category disappears and what survives is rewritten as one
labelme file per image, stored next to its image so labelme opens the pair:

    <output_root>/<split>/<stem>.png     the masked image
    <output_root>/<split>/<stem>.json    its polygons and points

Masking never moves anything, so the coordinates are the source ones. Each
instance becomes a polygon shape plus one point shape per annotated point, and
they share a group_id so the points stay attached to the spike they belong to.
Images without any instance of the category are kept as an all-black frame with
an empty shape list, which preserves the split as the COCO source defines it.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extract_coco_foreground import (
    _load_coco,
    _target_category_id,
    polygon_segmentation_to_mask,
)

LABELME_VERSION = "5.5.0"


def get_args_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--category-name", default="wheat_2")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train2017", "val2017"],
        help="Image directories whose annotations are instances_<split>.json.",
    )
    parser.add_argument(
        "--shapes",
        default="both",
        choices=["both", "polygons", "points"],
        help="Which part of an instance to export as labelme shapes.",
    )
    return parser


def _shape(label, points, shape_type, group_id):
    return {
        "label": label,
        "points": [[round(float(x), 2), round(float(y), 2)] for x, y in points],
        "group_id": group_id,
        "description": "",
        "shape_type": shape_type,
        "flags": {},
    }


def instance_shapes(annotations, label, shapes):
    """Turn one image's COCO annotations into labelme shapes."""
    documents = []
    for group_id, annotation in enumerate(annotations, 1):
        if shapes in {"both", "polygons"}:
            for polygon in annotation["segmentation"]:
                coordinates = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
                documents.append(
                    _shape(label, coordinates, "polygon", group_id)
                )
        if shapes in {"both", "points"}:
            for point in annotation.get("points", []):
                documents.append(_shape(label, [point], "point", group_id))
    return documents


def labelme_document(shapes, image_name, height, width):
    return {
        "version": LABELME_VERSION,
        "flags": {},
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": int(height),
        "imageWidth": int(width),
    }


def process_split(source_root, output_root, split, category_name, shapes):
    """Process one COCO split and return generation statistics."""
    source_image_root = source_root / split
    annotation_path = source_root / "annotations" / f"instances_{split}.json"
    if not source_image_root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {source_image_root}")
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Annotation file does not exist: {annotation_path}")

    coco = _load_coco(annotation_path)
    category_id = _target_category_id(coco, category_name)
    annotations_by_image = defaultdict(list)
    instance_count = 0
    point_count = 0
    for annotation in coco.get("annotations", []):
        if annotation.get("category_id") != category_id:
            continue
        segmentation = annotation.get("segmentation")
        if not isinstance(segmentation, list) or not segmentation:
            raise ValueError(
                f"Annotation {annotation.get('id')} does not contain a "
                "non-empty polygon segmentation"
            )
        annotations_by_image[annotation["image_id"]].append(annotation)
        instance_count += 1
        point_count += len(annotation.get("points", []))

    output_image_root = output_root / split
    output_image_root.mkdir(parents=True, exist_ok=True)

    foreground_pixels = 0
    total_pixels = 0
    empty_image_count = 0
    points_outside_mask = 0

    image_records = coco.get("images", [])
    for index, image_record in enumerate(image_records, 1):
        image_id = image_record["id"]
        relative_name = Path(image_record["file_name"])
        source_path = source_image_root / relative_name
        if not source_path.is_file() and relative_name.parent != Path("."):
            # some annotation files carry a directory prefix the split does not
            # actually have, so fall back to the bare file name
            fallback = source_image_root / relative_name.name
            if fallback.is_file():
                relative_name = Path(relative_name.name)
                source_path = fallback
        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {source_path}")

        height, width = image.shape[:2]
        expected_size = (image_record.get("height"), image_record.get("width"))
        if None not in expected_size and expected_size != (height, width):
            raise ValueError(
                f"Image size mismatch for {source_path.name}: COCO has "
                f"{expected_size}, file has {(height, width)}"
            )

        image_annotations = annotations_by_image.get(image_id, [])
        foreground_mask = np.zeros((height, width), dtype=np.uint8)
        for annotation in image_annotations:
            foreground_mask |= polygon_segmentation_to_mask(
                annotation["segmentation"], height, width
            )

        foreground_pixel_count = int(foreground_mask.sum())
        if foreground_pixel_count == 0:
            empty_image_count += 1
        foreground_pixels += foreground_pixel_count
        total_pixels += height * width

        foreground_image = np.zeros_like(image)
        selected = foreground_mask.astype(bool)
        foreground_image[selected] = image[selected]

        for annotation in image_annotations:
            for x, y in annotation.get("points", []):
                column = int(round(float(x)))
                row = int(round(float(y)))
                inside = 0 <= row < height and 0 <= column < width
                if not inside or not selected[row, column]:
                    points_outside_mask += 1

        output_name = relative_name.with_suffix(".png")
        output_path = output_image_root / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(
            str(output_path), foreground_image, [cv2.IMWRITE_PNG_COMPRESSION, 3]
        ):
            raise OSError(f"Could not write image: {output_path}")

        document = labelme_document(
            instance_shapes(image_annotations, category_name, shapes),
            output_name.name,
            height,
            width,
        )
        output_path.with_suffix(".json").write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        if index % 50 == 0 or index == len(image_records):
            print(f"  {split}: {index}/{len(image_records)}", flush=True)

    return {
        "split": split,
        "images": len(image_records),
        "instances": instance_count,
        "points": point_count,
        "empty_images": empty_image_count,
        "points_outside_mask": points_outside_mask,
        "foreground_pixels": foreground_pixels,
        "total_pixels": total_pixels,
    }


def process_dataset(source_root, output_root, splits, category_name, shapes):
    """Atomically build the labelme foreground dataset at ``output_root``."""
    source_root = source_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source dataset does not exist: {source_root}")
    if output_root.exists():
        raise FileExistsError(f"Output path already exists: {output_root}")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.building-", dir=output_root.parent
        )
    )
    try:
        statistics = [
            process_split(
                source_root, temporary_root, split, category_name, shapes
            )
            for split in splits
        ]
        os.replace(temporary_root, output_root)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    return statistics


def main():
    args = get_args_parser().parse_args()
    statistics = process_dataset(
        args.source_root,
        args.output_root,
        args.splits,
        args.category_name,
        args.shapes,
    )
    for item in statistics:
        foreground_fraction = (
            item["foreground_pixels"] / item["total_pixels"]
            if item["total_pixels"]
            else 0.0
        )
        print(
            f"{item['split']}: images={item['images']}, "
            f"{args.category_name}_instances={item['instances']}, "
            f"points={item['points']}, "
            f"empty_images={item['empty_images']}, "
            f"points_outside_mask={item['points_outside_mask']}, "
            f"foreground_fraction={foreground_fraction:.4%}"
        )


if __name__ == "__main__":
    main()
