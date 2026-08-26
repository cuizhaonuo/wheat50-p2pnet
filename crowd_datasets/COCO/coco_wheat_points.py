"""COCO point dataset adapter for wheat spikelet counting."""

import json
import random
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as functional


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CocoWheatPoints(Dataset):
    """Read nested spikelet points from a COCO annotation file.

    Only annotations belonging to ``category_name`` are used. Their custom
    ``points`` arrays are flattened into the single point set expected by
    P2PNet. Images without target points are deliberately kept as negative
    samples.
    """

    def __init__(
        self,
        data_root,
        annotation_file,
        image_dir,
        category_name="wheat_2",
        train=False,
        crop_size=512,
        crops_per_image=1,
        scale_range=(0.7, 1.3),
        horizontal_flip_probability=0.5,
    ):
        self.data_root = Path(data_root).expanduser()
        annotation_file = Path(annotation_file)
        image_dir = Path(image_dir)
        self.annotation_path = (
            annotation_file
            if annotation_file.is_absolute()
            else self.data_root / annotation_file
        )
        self.image_root = (
            image_dir if image_dir.is_absolute() else self.data_root / image_dir
        )
        self.category_name = category_name
        self.train = train
        self.crop_size = int(crop_size)
        self.crops_per_image = int(crops_per_image)
        self.scale_range = tuple(float(value) for value in scale_range)
        self.horizontal_flip_probability = float(horizontal_flip_probability)

        self._validate_options()
        self.records = self._load_records()
        self.num_points = sum(len(record["points"]) for record in self.records)
        self.num_empty_images = sum(
            len(record["points"]) == 0 for record in self.records
        )

    def _validate_options(self):
        if not self.annotation_path.is_file():
            raise FileNotFoundError(
                f"COCO annotation file does not exist: {self.annotation_path}"
            )
        if not self.image_root.is_dir():
            raise FileNotFoundError(
                f"COCO image directory does not exist: {self.image_root}"
            )
        if self.crop_size <= 0:
            raise ValueError("crop_size must be greater than zero")
        if self.crops_per_image <= 0:
            raise ValueError("crops_per_image must be greater than zero")
        if len(self.scale_range) != 2:
            raise ValueError("scale_range must contain exactly two values")
        if not 0 < self.scale_range[0] <= self.scale_range[1]:
            raise ValueError("scale_range must satisfy 0 < min <= max")
        if not 0 <= self.horizontal_flip_probability <= 1:
            raise ValueError(
                "horizontal_flip_probability must be between zero and one"
            )

    def _load_records(self):
        with self.annotation_path.open(encoding="utf-8") as handle:
            coco = json.load(handle)

        matching_categories = [
            category
            for category in coco.get("categories", [])
            if category.get("name") == self.category_name
        ]
        if len(matching_categories) != 1:
            raise ValueError(
                f"Expected exactly one COCO category named "
                f"{self.category_name!r}, found {len(matching_categories)}"
            )
        self.category_id = matching_categories[0]["id"]

        images = coco.get("images", [])
        images_by_id = {image["id"]: image for image in images}
        if len(images_by_id) != len(images):
            raise ValueError("COCO image ids must be unique")

        points_by_image = defaultdict(list)
        empty_annotation_ids = []
        for annotation in coco.get("annotations", []):
            if annotation.get("category_id") != self.category_id:
                continue
            image_id = annotation.get("image_id")
            if image_id not in images_by_id:
                raise ValueError(
                    f"Annotation {annotation.get('id')} references unknown "
                    f"image id {image_id}"
                )

            annotation_points = annotation.get("points") or []
            if not annotation_points:
                empty_annotation_ids.append(annotation.get("id"))
                continue
            point_array = np.asarray(annotation_points, dtype=np.float32)
            if (
                point_array.ndim != 2
                or point_array.shape[1] != 2
                or not np.isfinite(point_array).all()
            ):
                raise ValueError(
                    f"Annotation {annotation.get('id')} has invalid points; "
                    "expected a finite N x 2 array"
                )
            points_by_image[image_id].append(point_array)

        if empty_annotation_ids:
            preview = ", ".join(str(value) for value in empty_annotation_ids[:8])
            warnings.warn(
                f"Ignored {len(empty_annotation_ids)} empty "
                f"{self.category_name!r} annotations in "
                f"{self.annotation_path.name}: {preview}",
                UserWarning,
                stacklevel=2,
            )
        self.empty_annotation_ids = tuple(empty_annotation_ids)

        records = []
        for image in images:
            image_path = self.image_root / image["file_name"]
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Image referenced by COCO does not exist: {image_path}"
                )
            point_arrays = points_by_image.get(image["id"], [])
            points = (
                np.concatenate(point_arrays, axis=0).astype(np.float32, copy=False)
                if point_arrays
                else np.empty((0, 2), dtype=np.float32)
            )
            records.append(
                {
                    "image_id": image["id"],
                    "file_name": image["file_name"],
                    "path": image_path,
                    "width": image.get("width"),
                    "height": image.get("height"),
                    "points": points,
                }
            )
        return records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record["path"]) as image:
            image = image.convert("RGB")

        expected_size = (record["width"], record["height"])
        if None not in expected_size and image.size != expected_size:
            raise ValueError(
                f"Image size mismatch for {record['file_name']}: "
                f"COCO has {expected_size}, file has {image.size}"
            )

        points = record["points"].copy()
        if not self.train:
            tensor = self._to_normalized_tensor(image)
            target = self._make_target(points, record["image_id"])
            return tensor, [target]

        image, points = self._random_scale(image, points)
        image, points = self._pad_to_crop_size(image, points)
        crop_tensors = []
        crop_targets = []
        for _ in range(self.crops_per_image):
            crop, crop_points = self._random_crop(image, points)
            if random.random() < self.horizontal_flip_probability:
                crop = functional.hflip(crop)
                crop_points = crop_points.copy()
                crop_points[:, 0] = self.crop_size - crop_points[:, 0]
            crop_tensors.append(self._to_normalized_tensor(crop))
            crop_targets.append(
                self._make_target(crop_points, record["image_id"])
            )
        return torch.stack(crop_tensors), crop_targets

    def _random_scale(self, image, points):
        scale = random.uniform(*self.scale_range)
        width, height = image.size
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        if (resized_width, resized_height) == image.size:
            return image, points

        image = functional.resize(
            image,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        points = points.copy()
        points[:, 0] *= resized_width / width
        points[:, 1] *= resized_height / height
        return image, points

    def _pad_to_crop_size(self, image, points):
        width, height = image.size
        padding_right = max(0, self.crop_size - width)
        padding_bottom = max(0, self.crop_size - height)
        if padding_right or padding_bottom:
            image = functional.pad(
                image,
                [0, 0, padding_right, padding_bottom],
                fill=0,
            )
        return image, points

    def _random_crop(self, image, points):
        width, height = image.size
        top = random.randint(0, height - self.crop_size)
        left = random.randint(0, width - self.crop_size)
        crop = functional.crop(
            image,
            top,
            left,
            self.crop_size,
            self.crop_size,
        )

        inside = (
            (points[:, 0] >= left)
            & (points[:, 0] < left + self.crop_size)
            & (points[:, 1] >= top)
            & (points[:, 1] < top + self.crop_size)
        )
        crop_points = points[inside].copy()
        crop_points[:, 0] -= left
        crop_points[:, 1] -= top
        return crop, crop_points

    @staticmethod
    def _make_target(points, image_id):
        point_tensor = torch.as_tensor(points, dtype=torch.float32).reshape(-1, 2)
        return {
            "point": point_tensor,
            "labels": torch.ones(len(point_tensor), dtype=torch.int64),
            "image_id": torch.tensor([image_id], dtype=torch.int64),
        }

    @staticmethod
    def _to_normalized_tensor(image):
        tensor = functional.to_tensor(image)
        return functional.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
