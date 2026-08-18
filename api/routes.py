import logging
from flask import request, jsonify

from models.ocr_text_analyzer import analyze_image_with_ocr, analyze_images_batch_with_layout
from utils.image_loader import (
    InvalidOCRImage,
    MAXIMUM_IMAGE_FILES,
    decode_base64_image,
    load_image_path,
    valid_image_batch,
)

logger = logging.getLogger(__name__)


def health():
    return jsonify({'status': 'ok', 'service': 'ocr-text-analysis'})


def ready(models_ready: bool):
    if models_ready:
        return jsonify({'status': 'ready', 'model': 'ocr'})
    else:
        return jsonify({'status': 'not_ready', 'message': 'OCR model still loading'}), 503


def analyze():
    try:
        data = request.get_json()
        if not data or 'image_path' not in data:
            return jsonify({'error': 'No image_path provided'}), 400

        image_path = data['image_path']

        if request.environ.get('werkzeug.socket'):
            try:
                request.environ.get('werkzeug.socket').getpeername()
            except:
                logger.info("Client disconnected before OCR processing")
                return jsonify({'error': 'Client disconnected'}), 499

        image = load_image_path(image_path)

        extracted_text = analyze_image_with_ocr(image)

        return jsonify({
            'extracted_text': extracted_text
        })

    except BrokenPipeError:
        logger.info(f"Client disconnected during OCR processing")
        return jsonify({'error': 'Client disconnected'}), 499
    except (InvalidOCRImage, FileNotFoundError):
        return jsonify({'error': 'Image path or image data is invalid'}), 400
    except Exception:
        logger.exception("OCR analysis failed")
        return jsonify({'error': 'OCR analysis failed'}), 500


def analyze_batch():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        stream_mode = 'images' in data
        path_mode = 'image_paths' in data

        if not stream_mode and not path_mode:
            return jsonify({'error': 'Either image_paths or images must be provided'}), 400

        if stream_mode and path_mode:
            return jsonify({'error': 'Provide either image_paths OR images, not both'}), 400

        candidates = data['images'] if stream_mode else data['image_paths']
        if not valid_image_batch(candidates):
            return jsonify({'error': f'Batch must contain between 1 and {MAXIMUM_IMAGE_FILES} images'}), 400

        if request.environ.get('werkzeug.socket'):
            try:
                request.environ.get('werkzeug.socket').getpeername()
            except:
                logger.info(f"Client disconnected before batch OCR processing")
                return jsonify({'error': 'Client disconnected'}), 499

        images, error_map = _load_images_for_batch(data, stream_mode)

        if not images:
            return jsonify({'error': 'No valid images to process'}), 400

        batch_results = analyze_images_batch_with_layout(images)

        total_count = len(data.get('images', [])) if stream_mode else len(data.get('image_paths', []))
        results = _build_batch_results(batch_results, error_map, total_count)

        return jsonify({
            'results': results,
            'processed_count': len([r for r in results if 'error' not in r]),
            'failed_count': len(error_map),
            'total_count': total_count,
            'mode': 'stream' if stream_mode else 'path'
        })

    except BrokenPipeError:
        logger.info(f"Client disconnected during batch OCR processing")
        return jsonify({'error': 'Client disconnected'}), 499
    except (InvalidOCRImage, ValueError):
        return jsonify({'error': 'Image batch is invalid'}), 400
    except Exception:
        logger.exception("Batch OCR analysis failed")
        return jsonify({'error': 'Batch OCR analysis failed'}), 500


def _load_images_for_batch(data, stream_mode):
    images = []
    error_map = {}

    if stream_mode:
        images_data = data['images']
        if not isinstance(images_data, list):
            raise ValueError('images must be a list')
        for i, img_data in enumerate(images_data):
            if not isinstance(img_data, dict) or 'data' not in img_data:
                error_map[i] = "Invalid image data format"
                continue

            try:
                image = decode_base64_image(img_data['data'])
                images.append(image)
            except (InvalidOCRImage, TypeError):
                logger.warning("Rejected invalid image at batch position %s", i)
                error_map[i] = "Image data is invalid"
    else:
        image_paths = data['image_paths']

        if not isinstance(image_paths, list):
            raise ValueError('image_paths must be a list')
        for i, image_path in enumerate(image_paths):
            try:
                image = load_image_path(image_path)
                images.append(image)
            except (InvalidOCRImage, FileNotFoundError, TypeError):
                logger.warning("Rejected invalid image path at batch position %s", i)
                error_map[i] = "Image path or image data is invalid"

    return images, error_map


def _build_batch_results(batch_results, error_map, total_count):
    results = []
    valid_idx = 0

    for i in range(total_count):
        if i in error_map:
            results.append({
                'error': error_map[i],
                'extracted_text': '',
                'image_regions': []
            })
        else:
            analysis = batch_results[valid_idx]
            results.append({
                'extracted_text': analysis['extracted_text'],
                'image_regions': analysis['image_regions']
            })
            valid_idx += 1

    return results
