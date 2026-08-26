import json
import random
import tempfile
import unittest
import warnings
from pathlib import Path

import torch
from PIL import Image

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


if __name__ == "__main__":
    unittest.main()
