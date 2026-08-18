import logging
import os
import threading
from typing import List

import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

DEVICE_MODE = os.getenv("OCR_DEVICE", "auto").strip().lower()
RUNTIME_PROFILE = os.getenv("OCR_RUNTIME_PROFILE", "cpu").strip().lower()
CPU_THREADS = int(os.getenv("OCR_CPU_THREADS", "12"))
DETECTION_MODEL = os.getenv("OCR_DETECTION_MODEL", "PP-OCRv6_small_det")
RECOGNITION_MODEL = os.getenv("OCR_RECOGNITION_MODEL", "PP-OCRv6_small_rec")
DETECTION_LIMIT = int(os.getenv("OCR_DETECTION_LIMIT", "1920"))
LAYOUT_ENABLED = os.getenv("OCR_LAYOUT_ENABLED", "true").strip().lower() in {"1", "true", "yes"}
LAYOUT_MODEL = os.getenv("OCR_LAYOUT_MODEL", "PP-DocLayout-S")
LAYOUT_THRESHOLD = float(os.getenv("OCR_LAYOUT_THRESHOLD", "0.55"))
MKLDNN_MODE = os.getenv("OCR_ENABLE_MKLDNN", "false").strip().lower()

if DEVICE_MODE not in {"auto", "cpu", "cuda"}:
    raise RuntimeError("OCR_DEVICE must be auto, cpu or cuda")
if CPU_THREADS < 1 or CPU_THREADS > 64:
    raise RuntimeError("OCR_CPU_THREADS must be between 1 and 64")
if DETECTION_LIMIT < 64 or DETECTION_LIMIT > 4096:
    raise RuntimeError("OCR_DETECTION_LIMIT must be between 64 and 4096")
if not 0.1 <= LAYOUT_THRESHOLD <= 0.95:
    raise RuntimeError("OCR_LAYOUT_THRESHOLD must be between 0.1 and 0.95")
if MKLDNN_MODE not in {"1", "true", "yes", "0", "false", "no"}:
    raise RuntimeError("OCR_ENABLE_MKLDNN must be true or false")

ENABLE_MKLDNN = MKLDNN_MODE in {"1", "true", "yes"}


class OCRTextAnalyzer:
    def __init__(self):
        self.ocr, self.device = self._initialize_ocr()
        self.layout = self._initialize_layout(self.device) if LAYOUT_ENABLED else None
        self.lock = threading.Lock()

    def _initialize_ocr(self):
        import paddle
        from paddleocr import PaddleOCR

        gpu_available = paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
        if DEVICE_MODE == "cuda" and not gpu_available:
            raise RuntimeError("OCR_DEVICE is cuda but PaddlePaddle cannot use a CUDA device")

        use_gpu = DEVICE_MODE != "cpu" and gpu_available
        device = "gpu:0" if use_gpu else "cpu"
        logger.info(
            "initializing PaddleOCR device=%s runtime_profile=%s detection_model=%s recognition_model=%s",
            device,
            RUNTIME_PROFILE,
            DETECTION_MODEL,
            RECOGNITION_MODEL,
        )
        try:
            pipeline = self._create_pipeline(PaddleOCR, device)
        except RuntimeError:
            if DEVICE_MODE != "auto" or device == "cpu":
                raise
            logger.exception("PaddleOCR GPU initialization failed; retrying on CPU")
            device = "cpu"
            pipeline = self._create_pipeline(PaddleOCR, device)
        logger.info("PaddleOCR initialized device=%s", device)
        return pipeline, device

    @staticmethod
    def _initialize_layout(device):
        from paddleocr import LayoutDetection

        logger.info("initializing layout detection device=%s model=%s", device, LAYOUT_MODEL)
        return LayoutDetection(model_name=LAYOUT_MODEL, device=device, enable_mkldnn=ENABLE_MKLDNN)

    @staticmethod
    def _create_pipeline(paddle_ocr, device):
        return paddle_ocr(
            device=device,
            text_detection_model_name=DETECTION_MODEL,
            text_recognition_model_name=RECOGNITION_MODEL,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_det_limit_side_len=DETECTION_LIMIT,
            text_det_limit_type="max",
            text_recognition_batch_size=4,
            cpu_threads=CPU_THREADS,
            enable_mkldnn=ENABLE_MKLDNN,
        )

    def extract_text_from_image(self, image: Image.Image) -> str:
        return self.extract_text_from_images([image])[0]

    def extract_text_from_images(self, images: List[Image.Image]) -> List[str]:
        return [analysis["extracted_text"] for analysis in self.analyze_images(images)]

    def analyze_images(self, images: List[Image.Image]) -> List[dict]:
        if not images:
            return []

        analysis_images = [_prepare_analysis_image(image) for image in images]
        inputs = [np.asarray(image) for image in analysis_images]
        with self.lock:
            predictions = list(self.ocr.predict(inputs))
            layout_predictions = (
                list(self.layout.predict(inputs, batch_size=1, threshold=LAYOUT_THRESHOLD, layout_nms=True))
                if self.layout is not None
                else [None] * len(inputs)
            )
        if len(predictions) != len(images):
            raise RuntimeError("PaddleOCR returned an unexpected result count")
        if len(layout_predictions) != len(images):
            raise RuntimeError("layout detection returned an unexpected result count")
        results = []
        for image, image_input, prediction, layout_prediction in zip(analysis_images, inputs, predictions, layout_predictions):
            text, _, _, _, lines = _prediction_document(prediction)
            results.append({
                "extracted_text": text,
                "image_regions": _merge_image_regions(
                    _layout_image_regions(layout_prediction, image.size, lines),
                    _visual_image_regions(image_input, lines),
                ),
            })
        return results


def _prepare_analysis_image(image: Image.Image) -> Image.Image:
    prepared = ImageOps.exif_transpose(image).convert("RGB")
    longest_side = max(prepared.size)
    if longest_side <= DETECTION_LIMIT:
        return prepared

    scale = DETECTION_LIMIT / longest_side
    return prepared.resize(
        (
            max(1, round(prepared.width * scale)),
            max(1, round(prepared.height * scale)),
        ),
        Image.Resampling.LANCZOS,
    )


def _prediction_text(prediction) -> str:
    return _prediction_result(prediction)[0]


def _prediction_result(prediction):
    text, confidence, minimum_confidence, content_box, _ = _prediction_document(prediction)
    return text, confidence, minimum_confidence, content_box


def _prediction_document(prediction):
    payload = prediction.json
    if callable(payload):
        payload = payload()
    result = payload.get("res", payload)
    texts = result.get("rec_texts", [])
    scores = result.get("rec_scores", [])
    boxes = result.get("rec_boxes", [])
    if hasattr(boxes, "tolist"):
        boxes = boxes.tolist()

    lines = []
    accepted_scores = []
    accepted_boxes = []
    accepted_lines = []
    for index, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            continue
        box = boxes[index] if index < len(boxes) else None
        if isinstance(box, (list, tuple)) and len(box) == 4:
            x_min, y_min, x_max, y_max = [float(value) for value in box]
            lines.append(((y_min + y_max) / 2, (x_min + x_max) / 2, y_max - y_min, text.strip()))
            accepted_boxes.append((x_min, y_min, x_max, y_max))
            accepted_lines.append({"text": text.strip(), "box": (x_min, y_min, x_max, y_max)})
        else:
            lines.append((float(index), 0.0, 1.0, text.strip()))

        if index < len(scores):
            try:
                accepted_scores.append(float(scores[index]))
            except (TypeError, ValueError):
                pass

    text = "\n".join(item[3] for item in _sort_reading_order(lines))
    confidence = sum(accepted_scores) / len(accepted_scores) if accepted_scores else None
    minimum_confidence = min(accepted_scores) if accepted_scores else None
    content_box = _content_box(accepted_boxes)
    return text, confidence, minimum_confidence, content_box, accepted_lines


def _layout_image_regions(prediction, image_size, text_lines):
    if prediction is None:
        return []
    payload = prediction.json
    if callable(payload):
        payload = payload()
    result = payload.get("res", payload)
    boxes = result.get("boxes", [])
    width, height = image_size
    candidates = []
    for box in boxes:
        if not isinstance(box, dict) or str(box.get("label", "")).strip().lower() not in {"image", "figure"}:
            continue
        coordinate = box.get("coordinate")
        try:
            score = float(box.get("score", 0))
            x_min, y_min, x_max, y_max = [float(value) for value in coordinate]
        except (TypeError, ValueError):
            continue
        x_min = max(0.0, min(x_min, float(width)))
        y_min = max(0.0, min(y_min, float(height)))
        x_max = max(x_min, min(x_max, float(width)))
        y_max = max(y_min, min(y_max, float(height)))
        region_width = x_max - x_min
        region_height = y_max - y_min
        if region_width >= width * 0.94 and region_height >= height * 0.94:
            continue
        if region_width < width * 0.15 or region_height < height * 0.1:
            continue
        if region_width * region_height < width * height * 0.025:
            continue
        absolute_box = (x_min, y_min, x_max, y_max)
        candidates.append({
            "bounding_box": {
                "x": x_min / width,
                "y": y_min / height,
                "width": region_width / width,
                "height": region_height / height,
            },
            "confidence": score,
            "label": str(box["label"]).strip().lower(),
            "context_text": _nearby_text(absolute_box, text_lines, height),
            "_absolute_box": absolute_box,
        })

    candidates.sort(key=lambda candidate: (-candidate["confidence"], candidate["bounding_box"]["y"]))
    unique = []
    for candidate in candidates:
        if any(_intersection_over_union(candidate["_absolute_box"], kept["_absolute_box"]) >= 0.75 for kept in unique):
            continue
        unique.append(candidate)
    unique.sort(key=lambda candidate: (candidate["bounding_box"]["y"], candidate["bounding_box"]["x"]))
    for candidate in unique:
        candidate.pop("_absolute_box")
    return unique[:32]


def _visual_image_regions(image, text_lines):
    height, width = image.shape[:2]
    tile_size = max(24, round(min(width, height) / 24))
    rows = (height + tile_size - 1) // tile_size
    columns = (width + tile_size - 1) // tile_size
    text_tiles = np.zeros((rows, columns), dtype=bool)
    for line in text_lines:
        x_min, y_min, x_max, y_max = line["box"]
        padding = tile_size * 0.35
        column_start = max(0, int((x_min - padding) // tile_size))
        column_end = min(columns, int((x_max + padding) // tile_size) + 1)
        row_start = max(0, int((y_min - padding) // tile_size))
        row_end = min(rows, int((y_max + padding) // tile_size) + 1)
        text_tiles[row_start:row_end, column_start:column_end] = True

    visual_tiles = np.zeros((rows, columns), dtype=bool)
    tile_scores = np.zeros((rows, columns), dtype=float)
    for row in range(rows):
        for column in range(columns):
            if text_tiles[row, column]:
                continue
            tile = image[
                row * tile_size:min((row + 1) * tile_size, height),
                column * tile_size:min((column + 1) * tile_size, width),
            ].astype(np.float32)
            if tile.shape[0] < tile_size * 0.5 or tile.shape[1] < tile_size * 0.5:
                continue
            maximum = tile.max(axis=2)
            minimum = tile.min(axis=2)
            saturation = np.mean((maximum - minimum) / np.maximum(maximum, 1.0))
            grayscale = tile.mean(axis=2)
            variation = float(grayscale.std())
            horizontal_edges = np.abs(np.diff(grayscale, axis=1)).mean() if tile.shape[1] > 1 else 0
            vertical_edges = np.abs(np.diff(grayscale, axis=0)).mean() if tile.shape[0] > 1 else 0
            edge_strength = float((horizontal_edges + vertical_edges) / 2)
            score = min(1.0, saturation * 2.2 + variation / 90 + edge_strength / 55)
            tile_scores[row, column] = score
            visual_tiles[row, column] = saturation >= 0.11 and (variation >= 12 or edge_strength >= 5)

    bridged_tiles = visual_tiles.copy()
    for row in range(rows):
        for column in range(columns):
            if visual_tiles[row, column] or text_tiles[row, column] or tile_scores[row, column] < 0.25:
                continue
            horizontal_bridge = column > 0 and column + 1 < columns and \
                visual_tiles[row, column - 1] and visual_tiles[row, column + 1]
            vertical_bridge = row > 0 and row + 1 < rows and \
                visual_tiles[row - 1, column] and visual_tiles[row + 1, column]
            if horizontal_bridge or vertical_bridge:
                bridged_tiles[row, column] = True
    visual_tiles = bridged_tiles

    regions = []
    visited = np.zeros_like(visual_tiles)
    for start_row in range(rows):
        for start_column in range(columns):
            if not visual_tiles[start_row, start_column] or visited[start_row, start_column]:
                continue
            stack = [(start_row, start_column)]
            visited[start_row, start_column] = True
            component = []
            while stack:
                row, column = stack.pop()
                component.append((row, column))
                for next_row, next_column in ((row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1)):
                    if 0 <= next_row < rows and 0 <= next_column < columns and \
                            visual_tiles[next_row, next_column] and not visited[next_row, next_column]:
                        visited[next_row, next_column] = True
                        stack.append((next_row, next_column))

            row_min = min(item[0] for item in component)
            row_max = max(item[0] for item in component) + 1
            column_min = min(item[1] for item in component)
            column_max = max(item[1] for item in component) + 1
            bounding_tiles = (row_max - row_min) * (column_max - column_min)
            if len(component) / bounding_tiles < 0.38:
                continue
            x_min = column_min * tile_size
            y_min = row_min * tile_size
            x_max = min(width, column_max * tile_size)
            y_max = min(height, row_max * tile_size)
            region_width = x_max - x_min
            region_height = y_max - y_min
            area = region_width * region_height
            if region_width < width * 0.15 or region_height < height * 0.1 or area < width * height * 0.025:
                continue
            if region_width >= width * 0.94 and region_height >= height * 0.94:
                continue
            if region_width >= width * 0.94 and region_height >= height * 0.72:
                continue
            score = float(np.mean([tile_scores[row, column] for row, column in component]))
            regions.append({
                "bounding_box": {
                    "x": x_min / width,
                    "y": y_min / height,
                    "width": region_width / width,
                    "height": region_height / height,
                },
                "confidence": min(0.89, max(0.5, score)),
                "label": "image",
                "context_text": _nearby_text((x_min, y_min, x_max, y_max), text_lines, height),
            })
    return regions


def _merge_image_regions(*groups):
    candidates = [candidate for group in groups for candidate in group]
    candidates.sort(key=lambda candidate: -candidate["confidence"])
    unique = []
    for candidate in candidates:
        box = _absolute_normalized_box(candidate["bounding_box"])
        if any(_intersection_over_union(box, _absolute_normalized_box(kept["bounding_box"])) >= 0.65 for kept in unique):
            continue
        unique.append(candidate)
    unique.sort(key=lambda candidate: (candidate["bounding_box"]["y"], candidate["bounding_box"]["x"]))
    return unique[:32]


def _absolute_normalized_box(box):
    return (box["x"], box["y"], box["x"] + box["width"], box["y"] + box["height"])


def _nearby_text(image_box, text_lines, image_height):
    x_min, y_min, x_max, y_max = image_box
    maximum_distance = max(80.0, image_height * 0.22)
    nearby = []
    for line in text_lines:
        line_x_min, line_y_min, line_x_max, line_y_max = line["box"]
        center_x = (line_x_min + line_x_max) / 2
        center_y = (line_y_min + line_y_max) / 2
        if x_min <= center_x <= x_max and y_min <= center_y <= y_max:
            continue
        vertical_distance = max(y_min - line_y_max, line_y_min - y_max, 0.0)
        horizontal_overlap = max(0.0, min(x_max, line_x_max) - max(x_min, line_x_min))
        if vertical_distance > maximum_distance or horizontal_overlap <= 0:
            continue
        nearby.append((vertical_distance, abs(center_y - ((y_min + y_max) / 2)), line_y_min, line["text"]))
    nearby.sort()
    selected = sorted(nearby[:4], key=lambda item: item[2])
    return "\n".join(item[3] for item in selected)


def _intersection_over_union(first, second):
    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    if intersection == 0:
        return 0.0
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (first_area + second_area - intersection)


def _content_box(boxes):
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _sort_reading_order(lines):
    if not lines:
        return []

    ordered = sorted(lines, key=lambda item: (item[0], item[1]))
    rows = []
    for line in ordered:
        tolerance = max(line[2] * 0.6, 8.0)
        if not rows or abs(line[0] - rows[-1][0][0]) > tolerance:
            rows.append([line])
        else:
            rows[-1].append(line)

    result = []
    for row in rows:
        result.extend(sorted(row, key=lambda item: item[1]))
    return result


_ocr_analyzer = None
_ocr_analyzer_lock = threading.Lock()


def get_ocr_analyzer() -> OCRTextAnalyzer:
    global _ocr_analyzer
    if _ocr_analyzer is None:
        with _ocr_analyzer_lock:
            if _ocr_analyzer is None:
                _ocr_analyzer = OCRTextAnalyzer()
    return _ocr_analyzer


def analyze_image_with_ocr(image: Image.Image) -> str:
    return get_ocr_analyzer().extract_text_from_image(image)


def analyze_images_batch_with_ocr(images: list) -> List[str]:
    return get_ocr_analyzer().extract_text_from_images(images)


def analyze_images_batch_with_layout(images: list) -> List[dict]:
    return get_ocr_analyzer().analyze_images(images)
