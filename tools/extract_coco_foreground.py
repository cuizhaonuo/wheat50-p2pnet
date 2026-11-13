"""Create a COCO dataset whose images retain only one category's masks."""

import argparse
import copy
import json
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def get_args_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Keep pixels inside all instances of one COCO category and set "
            "the remaining image area to black."
        )
    )
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--category-name", default="wheat_2")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train2017", "val2017"],
        help="Image directories whose annotations are instances_<split>.json.",
    )
    return parser


def polygon_segmentation_to_mask(segmentation, height, width):
    """Rasterize a COCO polygon segmentation to a binary uint8 mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    polygons = []
    for polygon in segmentation:
        coordinates = np.asarray(polygon, dtype=np.float32)
        if coordinates.size < 6 or coordinates.size % 2:
            raise ValueError("COCO polygons must contain at least three x/y pairs")
        coordinates = np.rint(coordinates.reshape(-1, 2)).astype(np.int32)
        polygons.append(coordinates)
    if polygons:
        cv2.fillPoly(mask, polygons, 1)
    return mask


def _load_coco(annotation_path):
    with annotation_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _target_category_id(coco, category_name):
    matching = [
        category["id"]
        for category in coco.get("categories", [])
        if category.get("name") == category_name
    ]
    if len(matching) != 1:
        raise ValueError(
            f"Expected exactly one category named {category_name!r}, "
            f"found {len(matching)}"
        )
    return matching[0]


def filter_coco_to_category(coco, category_id):
    """Keep all images but only one category and its annotations."""
    output_coco = copy.deepcopy(coco)
    output_coco["categories"] = [
        category
        for category in output_coco.get("categories", [])
        if category.get("id") == category_id
    ]
    output_coco["annotations"] = [
        annotation
        for annotation in output_coco.get("annotations", [])
        if annotation.get("category_id") == category_id
    ]
    return output_coco


def process_split(source_root, output_root, split, category_name):
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
    target_annotation_count = 0
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
        target_annotation_count += 1

    output_image_root = output_root / split
    output_annotation_root = output_root / "annotations"
    output_image_root.mkdir(parents=True, exist_ok=True)
    output_annotation_root.mkdir(parents=True, exist_ok=True)

    output_coco = filter_coco_to_category(coco, category_id)
    output_images_by_id = {
        image["id"]: image for image in output_coco.get("images", [])
    }
    foreground_pixels = 0
    total_pixels = 0
    empty_image_count = 0

    for image_record in coco.get("images", []):
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

        foreground_mask = np.zeros((height, width), dtype=np.uint8)
        for annotation in annotations_by_image.get(image_id, []):
            foreground_mask |= polygon_segmentation_to_mask(
                annotation["segmentation"], height, width
            )

        foreground_pixel_count = int(foreground_mask.sum())
        if foreground_pixel_count == 0:
            empty_image_count += 1
        foreground_pixels += foreground_pixel_count
        total_pixels += height * width

        foreground_image = np.zeros_like(image)
        foreground_image[foreground_mask.astype(bool)] = image[
            foreground_mask.astype(bool)
        ]

        output_name = relative_name.with_suffix(".png")
        output_path = output_image_root / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(
            str(output_path), foreground_image, [cv2.IMWRITE_PNG_COMPRESSION, 3]
        ):
            raise OSError(f"Could not write image: {output_path}")
        output_images_by_id[image_id]["file_name"] = output_name.as_posix()

    output_annotation_path = (
        output_annotation_root / f"instances_{split}.json"
    )
    with output_annotation_path.open("w", encoding="utf-8") as handle:
        json.dump(output_coco, handle, ensure_ascii=False)

    return {
        "split": split,
        "images": len(coco.get("images", [])),
        "target_annotations": target_annotation_count,
        "empty_images": empty_image_count,
        "foreground_pixels": foreground_pixels,
        "total_pixels": total_pixels,
    }


def process_dataset(source_root, output_root, splits, category_name):
    """Atomically build the foreground-only dataset at ``output_root``."""
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
                source_root,
                temporary_root,
                split,
                category_name,
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
    )
    for item in statistics:
        foreground_fraction = (
            item["foreground_pixels"] / item["total_pixels"]
            if item["total_pixels"]
            else 0.0
        )
        print(
            f"{item['split']}: images={item['images']}, "
            f"{args.category_name}_annotations={item['target_annotations']}, "
            f"empty_images={item['empty_images']}, "
            f"foreground_fraction={foreground_fraction:.4%}"
        )


if __name__ == "__main__":
    main()
