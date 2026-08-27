"""COCO point dataset adapter for wheat spikelet counting."""

import json
import random
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
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
        copy_paste_probability=0.0,
        copy_paste_max_instances=3,
        copy_paste_placement_attempts=20,
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
        self.copy_paste_probability = float(copy_paste_probability)
        self.copy_paste_max_instances = int(copy_paste_max_instances)
        self.copy_paste_placement_attempts = int(
            copy_paste_placement_attempts
        )

        self._validate_options()
        self.records = self._load_records()
        self.copy_paste_instances = [
            (record, instance)
            for record in self.records
            for instance in record["instances"]
            if len(instance["points"]) > 0
        ]
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
        if not 0 <= self.copy_paste_probability <= 1:
            raise ValueError(
                "copy_paste_probability must be between zero and one"
            )
        if self.copy_paste_max_instances <= 0:
            raise ValueError(
                "copy_paste_max_instances must be greater than zero"
            )
        if self.copy_paste_placement_attempts <= 0:
            raise ValueError(
                "copy_paste_placement_attempts must be greater than zero"
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
        instances_by_image = defaultdict(list)
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
                point_array = np.empty((0, 2), dtype=np.float32)
            else:
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

            polygons = self._parse_polygons(annotation.get("segmentation"))
            if polygons:
                instances_by_image[image_id].append(
                    {
                        "annotation_id": annotation.get("id"),
                        "polygons": polygons,
                        "points": point_array,
                    }
                )

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
                    "instances": instances_by_image.get(image["id"], []),
                }
            )
        return records

    @staticmethod
    def _parse_polygons(segmentation):
        if not isinstance(segmentation, list):
            return []
        polygons = []
        for polygon in segmentation:
            coordinates = np.asarray(polygon, dtype=np.float32)
            if (
                coordinates.ndim != 1
                or len(coordinates) < 6
                or len(coordinates) % 2
                or not np.isfinite(coordinates).all()
            ):
                continue
            polygons.append(coordinates.reshape(-1, 2))
        return polygons

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

        occupancy = self._rasterize_instances(record, image.size)
        image, points, occupancy = self._random_scale(image, points, occupancy)
        image, points, occupancy = self._pad_to_crop_size(
            image, points, occupancy
        )
        crop_tensors = []
        crop_targets = []
        for _ in range(self.crops_per_image):
            crop, crop_points, crop_occupancy = self._random_crop(
                image, points, occupancy
            )
            if random.random() < self.horizontal_flip_probability:
                crop = functional.hflip(crop)
                crop_occupancy = functional.hflip(crop_occupancy)
                crop_points = crop_points.copy()
                crop_points[:, 0] = self.crop_size - crop_points[:, 0]
            if (
                self.copy_paste_probability > 0
                and random.random() < self.copy_paste_probability
            ):
                crop, crop_points, crop_occupancy = self._copy_paste(
                    crop, crop_points, crop_occupancy
                )
            crop_tensors.append(self._to_normalized_tensor(crop))
            crop_targets.append(
                self._make_target(crop_points, record["image_id"])
            )
        return torch.stack(crop_tensors), crop_targets

    def _random_scale(self, image, points, occupancy):
        scale = random.uniform(*self.scale_range)
        width, height = image.size
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        if (resized_width, resized_height) == image.size:
            return image, points, occupancy

        image = functional.resize(
            image,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        occupancy = functional.resize(
            occupancy,
            [resized_height, resized_width],
            interpolation=InterpolationMode.NEAREST,
        )
        points = points.copy()
        points[:, 0] *= resized_width / width
        points[:, 1] *= resized_height / height
        return image, points, occupancy

    def _pad_to_crop_size(self, image, points, occupancy):
        width, height = image.size
        padding_right = max(0, self.crop_size - width)
        padding_bottom = max(0, self.crop_size - height)
        if padding_right or padding_bottom:
            image = functional.pad(
                image,
                [0, 0, padding_right, padding_bottom],
                fill=0,
            )
            occupancy = functional.pad(
                occupancy,
                [0, 0, padding_right, padding_bottom],
                fill=0,
            )
        return image, points, occupancy

    def _random_crop(self, image, points, occupancy):
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
        crop_occupancy = functional.crop(
            occupancy,
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
        return crop, crop_points, crop_occupancy

    @staticmethod
    def _rasterize_instances(record, image_size):
        mask = Image.new("L", image_size, 0)
        draw = ImageDraw.Draw(mask)
        for instance in record["instances"]:
            for polygon in instance["polygons"]:
                draw.polygon([tuple(point) for point in polygon], fill=255)
        return mask

    def _load_copy_paste_patch(self, record, instance):
        mask = Image.new("L", (record["width"], record["height"]), 0)
        draw = ImageDraw.Draw(mask)
        for polygon in instance["polygons"]:
            draw.polygon([tuple(point) for point in polygon], fill=255)
        bounding_box = mask.getbbox()
        if bounding_box is None:
            return None

        with Image.open(record["path"]) as donor_image:
            donor_image = donor_image.convert("RGB")
            patch = donor_image.crop(bounding_box)
        patch_mask = mask.crop(bounding_box)
        points = instance["points"].copy()
        points[:, 0] -= bounding_box[0]
        points[:, 1] -= bounding_box[1]
        return patch, patch_mask, points

    def _copy_paste(self, image, points, occupancy):
        """Paste complete non-overlapping polygon instances into one crop."""
        if not self.copy_paste_instances:
            return image, points, occupancy

        image = image.copy()
        occupancy = occupancy.copy()
        pasted_points = []
        for _ in range(self.copy_paste_max_instances):
            record, instance = random.choice(self.copy_paste_instances)
            loaded = self._load_copy_paste_patch(record, instance)
            if loaded is None:
                continue
            patch, patch_mask, instance_points = loaded
            patch_width, patch_height = patch.size
            if patch_width > self.crop_size or patch_height > self.crop_size:
                continue

            placed = False
            for _ in range(self.copy_paste_placement_attempts):
                top = random.randint(0, self.crop_size - patch_height)
                left = random.randint(0, self.crop_size - patch_width)
                translated_points = instance_points + np.asarray(
                    [left, top], dtype=np.float32
                )
                points_inside = (
                    (translated_points[:, 0] >= 0)
                    & (translated_points[:, 0] < self.crop_size)
                    & (translated_points[:, 1] >= 0)
                    & (translated_points[:, 1] < self.crop_size)
                ).all()
                if not points_inside:
                    continue

                occupied_region = occupancy.crop(
                    (left, top, left + patch_width, top + patch_height)
                )
                overlap = np.logical_and(
                    np.asarray(occupied_region) > 0,
                    np.asarray(patch_mask) > 0,
                ).any()
                if overlap:
                    continue

                image.paste(patch, (left, top), patch_mask)
                occupancy.paste(255, (left, top), patch_mask)
                pasted_points.append(translated_points)
                placed = True
                break
            if not placed:
                continue

        if pasted_points:
            points = np.concatenate([points, *pasted_points], axis=0)
        return image, points, occupancy

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
