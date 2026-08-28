import json
import random
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image, ImageDraw

from crowd_datasets.COCO.coco_wheat_points import CocoWheatPoints
from util.misc import collate_fn_crowd


class CocoWheatPointsTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        (self.root / "annotations").mkdir()
        (self.root / "train2017").mkdir()

        for image_id in (1, 2):
            Image.new("RGB", (8, 8), color=(image_id * 20, 40, 60)).save(
                self.root / "train2017" / f"{image_id}.jpg"
            )

        coco = {
            "images": [
                {"id": 1, "file_name": "1.jpg", "width": 8, "height": 8},
                {"id": 2, "file_name": "2.jpg", "width": 8, "height": 8},
            ],
            "categories": [
                {"id": 1, "name": "wheat"},
                {"id": 2, "name": "wheat_2"},
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "points": [[3, 3]],
                },
                {
                    "id": 2,
                    "image_id": 1,
                    "category_id": 2,
                    "points": [[1, 2], [6, 7]],
                },
                {
                    "id": 3,
                    "image_id": 1,
                    "category_id": 2,
                    "points": [],
                },
            ],
        }
        annotation_path = self.root / "annotations" / "instances_train2017.json"
        annotation_path.write_text(json.dumps(coco), encoding="utf-8")

    def tearDown(self):
        self.temporary_directory.cleanup()

    def build_dataset(self, **overrides):
        options = {
            "data_root": self.root,
            "annotation_file": "annotations/instances_train2017.json",
            "image_dir": "train2017",
            "category_name": "wheat_2",
            "train": False,
        }
        options.update(overrides)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return CocoWheatPoints(**options)

    def test_flattens_only_target_category_and_keeps_empty_images(self):
        dataset = self.build_dataset()

        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.num_points, 2)
        self.assertEqual(dataset.num_empty_images, 1)
        self.assertEqual(dataset.empty_annotation_ids, (3,))

        image, targets = dataset[0]
        self.assertEqual(tuple(image.shape), (3, 8, 8))
        torch.testing.assert_close(
            targets[0]["point"], torch.tensor([[1.0, 2.0], [6.0, 7.0]])
        )
        torch.testing.assert_close(targets[0]["labels"], torch.ones(2).long())

        _, empty_targets = dataset[1]
        self.assertEqual(tuple(empty_targets[0]["point"].shape), (0, 2))
        self.assertEqual(tuple(empty_targets[0]["labels"].shape), (0,))

    def test_training_augmentation_returns_p2pnet_batch_contract(self):
        dataset = self.build_dataset(
            train=True,
            crop_size=8,
            crops_per_image=1,
            scale_range=(1.0, 1.0),
            horizontal_flip_probability=1.0,
        )
        random.seed(0)
        images, targets = dataset[0]

        self.assertEqual(tuple(images.shape), (1, 3, 8, 8))
        self.assertEqual(len(targets), 1)
        torch.testing.assert_close(
            targets[0]["point"], torch.tensor([[7.0, 2.0], [2.0, 7.0]])
        )

        batched_images, batched_targets = collate_fn_crowd([dataset[0], dataset[1]])
        self.assertEqual(tuple(batched_images.shape), (2, 3, 128, 128))
        self.assertEqual(len(batched_targets), 2)

    def test_disabled_copy_paste_preserves_training_behavior(self):
        options = {
            "train": True,
            "crop_size": 8,
            "crops_per_image": 2,
            "scale_range": (1.0, 1.0),
            "horizontal_flip_probability": 0.5,
        }
        default_dataset = self.build_dataset(**options)
        disabled_dataset = self.build_dataset(
            **options,
            copy_paste_probability=0.0,
            copy_paste_max_instances=2,
            copy_paste_placement_attempts=4,
        )

        random.seed(17)
        default_images, default_targets = default_dataset[0]
        random.seed(17)
        disabled_images, disabled_targets = disabled_dataset[0]

        torch.testing.assert_close(default_images, disabled_images)
        for default_target, disabled_target in zip(
            default_targets, disabled_targets
        ):
            torch.testing.assert_close(
                default_target["point"], disabled_target["point"]
            )

    def test_copy_paste_translates_points_and_rejects_overlap(self):
        source_image = Image.new("RGB", (10, 10), color="black")
        ImageDraw.Draw(source_image).rectangle((1, 1, 3, 3), fill="red")
        source_image.save(self.root / "train2017" / "source.png")
        Image.new("RGB", (10, 10), color="black").save(
            self.root / "train2017" / "target.png"
        )
        coco = {
            "images": [
                {
                    "id": 10,
                    "file_name": "source.png",
                    "width": 10,
                    "height": 10,
                },
                {
                    "id": 11,
                    "file_name": "target.png",
                    "width": 10,
                    "height": 10,
                },
            ],
            "categories": [{"id": 2, "name": "wheat_2"}],
            "annotations": [
                {
                    "id": 10,
                    "image_id": 10,
                    "category_id": 2,
                    "segmentation": [[1, 1, 3, 1, 3, 3, 1, 3]],
                    "points": [[2, 2]],
                }
            ],
        }
        annotation_path = self.root / "annotations" / "copy_paste.json"
        annotation_path.write_text(json.dumps(coco), encoding="utf-8")
        dataset = self.build_dataset(
            annotation_file="annotations/copy_paste.json",
            train=True,
            crop_size=10,
            scale_range=(1.0, 1.0),
            horizontal_flip_probability=0.0,
            copy_paste_probability=1.0,
            copy_paste_max_instances=1,
            copy_paste_placement_attempts=2,
        )

        target_image = Image.new("RGB", (10, 10), color="black")
        original_occupancy = Image.new("L", (10, 10), 0)
        ImageDraw.Draw(original_occupancy).rectangle((0, 0, 4, 4), fill=255)
        original_points = np.asarray([[2, 2]], dtype=np.float32)
        with patch(
            "crowd_datasets.COCO.coco_wheat_points.random.randint",
            side_effect=[0, 0, 6, 6],
        ):
            pasted_image, pasted_points, pasted_occupancy = (
                dataset._copy_paste(
                    target_image, original_points, original_occupancy
                )
            )

        np.testing.assert_array_equal(
            pasted_points,
            np.asarray([[2, 2], [7, 7]], dtype=np.float32),
        )
        original_mask = np.asarray(original_occupancy) > 0
        result_mask = np.asarray(pasted_occupancy) > 0
        pasted_mask = np.logical_and(result_mask, ~original_mask)
        self.assertFalse(np.logical_and(original_mask, pasted_mask).any())
        self.assertEqual(int(pasted_mask.sum()), 9)
        self.assertEqual(pasted_image.getpixel((7, 7)), (255, 0, 0))

        with patch(
            "crowd_datasets.COCO.coco_wheat_points.random.random",
            side_effect=[0.5, 0.0],
        ), patch(
            "crowd_datasets.COCO.coco_wheat_points.random.randint",
            side_effect=[0, 0, 6, 6],
        ):
            _, targets = dataset[1]
        torch.testing.assert_close(
            targets[0]["point"], torch.tensor([[7.0, 7.0]])
        )


if __name__ == "__main__":
    unittest.main()
