import unittest

import numpy as np
from PIL import Image

from models.ocr_text_analyzer import (
    DETECTION_LIMIT,
    ENABLE_MKLDNN,
    OCRTextAnalyzer,
    _layout_image_regions,
    _prediction_result,
    _prediction_text,
    _prepare_analysis_image,
    _sort_reading_order,
    _visual_image_regions,
)


class Prediction:
    def __init__(self, payload):
        self.json = payload


class OCRTextAnalyzerTest(unittest.TestCase):
    def test_pipeline_uses_configured_mkldnn_mode(self):
        captured = {}

        def create_pipeline(**kwargs):
            captured.update(kwargs)
            return object()

        OCRTextAnalyzer._create_pipeline(create_pipeline, "cpu")

        self.assertEqual(captured["enable_mkldnn"], ENABLE_MKLDNN)

    def test_prediction_text_orders_rows_and_columns(self):
        prediction = Prediction(
            {
                "res": {
                    "rec_texts": ["second column", "next row", "first column"],
                    "rec_boxes": [
                        [120, 10, 220, 30],
                        [10, 50, 100, 70],
                        [10, 11, 100, 31],
                    ],
                }
            }
        )

        self.assertEqual(
            _prediction_text(prediction),
            "first column\nsecond column\nnext row",
        )

    def test_prediction_text_skips_empty_values(self):
        prediction = Prediction(
            {
                "rec_texts": ["", "visible", None],
                "rec_boxes": [[0, 0, 1, 1], [0, 10, 10, 20], [0, 20, 10, 30]],
            }
        )

        self.assertEqual(_prediction_text(prediction), "visible")

    def test_prediction_result_reports_mean_confidence_and_content_box(self):
        prediction = Prediction(
            {
                "res": {
                    "rec_texts": ["title", "ingredient", ""],
                    "rec_scores": [0.98, 0.82, 0.01],
                    "rec_boxes": [[20, 10, 120, 30], [10, 40, 210, 70], [0, 0, 1, 1]],
                }
            }
        )

        text, confidence, minimum_confidence, content_box = _prediction_result(prediction)

        self.assertEqual(text, "title\ningredient")
        self.assertAlmostEqual(confidence, 0.9)
        self.assertAlmostEqual(minimum_confidence, 0.82)
        self.assertEqual(content_box, (10.0, 10.0, 210.0, 70.0))

    def test_prediction_result_allows_missing_scores_and_boxes(self):
        text, confidence, minimum_confidence, content_box = _prediction_result(Prediction({"rec_texts": ["recipe"]}))

        self.assertEqual(text, "recipe")
        self.assertIsNone(confidence)
        self.assertIsNone(minimum_confidence)
        self.assertIsNone(content_box)

    def test_reading_order_accepts_empty_results(self):
        self.assertEqual(_sort_reading_order([]), [])

    def test_prepare_analysis_image_preserves_small_images(self):
        prepared = _prepare_analysis_image(Image.new("RGBA", (800, 600), "white"))

        self.assertEqual(prepared.size, (800, 600))
        self.assertEqual(prepared.mode, "RGB")

    def test_prepare_analysis_image_limits_long_edge_without_distortion(self):
        prepared = _prepare_analysis_image(Image.new("RGB", (4000, 2000), "white"))

        self.assertEqual(prepared.size, (DETECTION_LIMIT, DETECTION_LIMIT // 2))

    def test_layout_regions_keep_recipe_images_with_nearby_text(self):
        prediction = Prediction({
            "res": {
                "boxes": [
                    {"label": "text", "score": 0.99, "coordinate": [0, 0, 500, 100]},
                    {"label": "image", "score": 0.91, "coordinate": [100, 200, 900, 700]},
                ]
            }
        })
        lines = [
            {"text": "Smaż cebulę przez 5 minut", "box": (120, 130, 850, 180)},
            {"text": "Następny krok", "box": (120, 720, 850, 760)},
        ]

        regions = _layout_image_regions(prediction, (1000, 1000), lines)

        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0]["bounding_box"], {"x": 0.1, "y": 0.2, "width": 0.8, "height": 0.5})
        self.assertEqual(regions[0]["confidence"], 0.91)
        self.assertIn("Smaż cebulę", regions[0]["context_text"])

    def test_layout_regions_reject_small_chrome_and_deduplicate_overlaps(self):
        prediction = Prediction({
            "boxes": [
                {"label": "header_image", "score": 0.99, "coordinate": [0, 0, 1000, 100]},
                {"label": "image", "score": 0.99, "coordinate": [0, 0, 1000, 1000]},
                {"label": "image", "score": 0.90, "coordinate": [100, 200, 800, 700]},
                {"label": "figure", "score": 0.80, "coordinate": [110, 210, 790, 690]},
                {"label": "image", "score": 0.95, "coordinate": [0, 0, 80, 80]},
            ]
        })

        regions = _layout_image_regions(prediction, (1000, 1000), [])

        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0]["confidence"], 0.9)

    def test_visual_regions_find_textured_photo_below_document_text(self):
        image = np.full((1000, 1000, 3), 245, dtype=np.uint8)
        random = np.random.default_rng(42)
        image[550:950, 100:900] = random.integers(0, 255, (400, 800, 3), dtype=np.uint8)
        lines = [
            {"text": "Step one", "box": (100, 100, 900, 150)},
            {"text": "Cook until golden", "box": (100, 220, 900, 270)},
        ]

        regions = _visual_image_regions(image, lines)

        self.assertEqual(len(regions), 1)
        self.assertGreater(regions[0]["bounding_box"]["y"], 0.45)
        self.assertGreater(regions[0]["bounding_box"]["width"], 0.7)

    def test_visual_regions_ignore_blank_document(self):
        image = np.full((1000, 1000, 3), 245, dtype=np.uint8)

        self.assertEqual(_visual_image_regions(image, []), [])

    def test_visual_regions_bridge_textured_desaturated_part_of_one_photo(self):
        image = np.full((1000, 1000, 3), 245, dtype=np.uint8)
        random = np.random.default_rng(7)
        image[550:950, 100:480] = random.integers(0, 255, (400, 380, 3), dtype=np.uint8)
        image[550:950, 520:900] = random.integers(0, 255, (400, 380, 3), dtype=np.uint8)
        grayscale = random.integers(0, 255, (400, 40, 1), dtype=np.uint8)
        image[550:950, 480:520] = np.repeat(grayscale, 3, axis=2)

        regions = _visual_image_regions(image, [])

        self.assertEqual(len(regions), 1)
        self.assertGreater(regions[0]["bounding_box"]["width"], 0.7)


if __name__ == "__main__":
    unittest.main()
