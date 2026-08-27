import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tools.extract_coco_foreground import process_split


class ExtractCocoForegroundTest(unittest.TestCase):
    def test_keeps_target_polygon_and_blacks_other_pixels(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_root = root / "source"
            output_root = root / "output"
            (source_root / "annotations").mkdir(parents=True)
            (source_root / "train2017").mkdir()

            image = np.full((8, 8, 3), 100, dtype=np.uint8)
            cv2.imwrite(str(source_root / "train2017" / "positive.jpg"), image)
            cv2.imwrite(str(source_root / "train2017" / "empty.jpg"), image)
            coco = {
                "images": [
                    {
                        "id": 1,
                        "file_name": "positive.jpg",
                        "width": 8,
                        "height": 8,
                    },
                    {
                        "id": 2,
                        "file_name": "empty.jpg",
                        "width": 8,
                        "height": 8,
                    },
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 2,
                        "segmentation": [[1, 1, 5, 1, 5, 5, 1, 5]],
                    },
                    {
                        "id": 2,
                        "image_id": 1,
                        "category_id": 1,
                        "segmentation": [[6, 6, 7, 6, 7, 7, 6, 7]],
                    },
                ],
                "categories": [
                    {"id": 1, "name": "wheat"},
                    {"id": 2, "name": "wheat_2"},
                ],
            }
            annotation_path = (
                source_root / "annotations" / "instances_train2017.json"
            )
            annotation_path.write_text(json.dumps(coco), encoding="utf-8")

            statistics = process_split(
                source_root, output_root, "train2017", "wheat_2"
            )

            positive = cv2.imread(
                str(output_root / "train2017" / "positive.png")
            )
            empty = cv2.imread(str(output_root / "train2017" / "empty.png"))
            self.assertTrue(np.all(positive[0, 0] == 0))
            self.assertTrue(np.all(positive[3, 3] == 100))
            self.assertTrue(np.all(positive[6, 6] == 0))
            self.assertTrue(np.all(empty == 0))
            self.assertEqual(statistics["images"], 2)
            self.assertEqual(statistics["target_annotations"], 1)
            self.assertEqual(statistics["empty_images"], 1)

            output_coco = json.loads(
                (
                    output_root
                    / "annotations"
                    / "instances_train2017.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                [image["file_name"] for image in output_coco["images"]],
                ["positive.png", "empty.png"],
            )
            self.assertEqual(output_coco["annotations"], [coco["annotations"][0]])
            self.assertEqual(output_coco["categories"], [coco["categories"][1]])


if __name__ == "__main__":
    unittest.main()
