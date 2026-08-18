import argparse
import atexit
import gc
import logging
import os
import signal
import sys

from flask import Flask
from werkzeug.exceptions import RequestEntityTooLarge

from api import routes
from models.ocr_text_analyzer import get_ocr_analyzer
from utils.image_loader import MAXIMUM_REQUEST_BYTES


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAXIMUM_REQUEST_BYTES
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
models_ready = False


@app.errorhandler(RequestEntityTooLarge)
def request_too_large(_error):
    return {"error": "Request body is too large"}, 413


@app.route("/health", methods=["GET"])
def health():
    return routes.health()


@app.route("/ready", methods=["GET"])
def ready():
    return routes.ready(models_ready)


@app.route("/analyze", methods=["POST"])
def analyze():
    return routes.analyze()


@app.route("/analyze_batch", methods=["POST"])
def analyze_batch():
    return routes.analyze_batch()


def cleanup_resources():
    logger.info("Cleaning up OCR resources...")
    gc.collect()
    logger.info("Cleanup complete")


def signal_handler(signum, _frame):
    logger.info("Received signal %s, shutting down gracefully...", signum)
    cleanup_resources()
    sys.exit(0)


def initialize_models():
    global models_ready

    if models_ready:
        return

    logger.info("Initializing OCR model...")
    get_ocr_analyzer()
    models_ready = True
    logger.info("OCR model ready for inference")


def parse_arguments():
    parser = argparse.ArgumentParser(description="OCR Text Analysis API Server")
    parser.add_argument(
        "--listen",
        default=os.getenv("OCR_HTTP_ADDRESS", "0.0.0.0:8080"),
        help="Listen address in host:port format",
    )
    parser.add_argument("--tls-cert", help="Path to a TLS certificate")
    parser.add_argument("--tls-key", help="Path to the TLS private key")
    return parser.parse_args()


def parse_listen_address(value):
    if ":" not in value:
        return value, 5001

    host, port_value = value.rsplit(":", 1)
    try:
        return host, int(port_value)
    except ValueError as error:
        raise ValueError("listen address must use host:port format") from error


def tls_context(arguments):
    if bool(arguments.tls_cert) != bool(arguments.tls_key):
        raise ValueError("--tls-cert and --tls-key must be provided together")
    if not arguments.tls_cert:
        return None
    if not os.path.isfile(arguments.tls_cert):
        raise ValueError(f"TLS certificate not found: {arguments.tls_cert}")
    if not os.path.isfile(arguments.tls_key):
        raise ValueError(f"TLS key not found: {arguments.tls_key}")
    return arguments.tls_cert, arguments.tls_key


def main():
    arguments = parse_arguments()
    try:
        host, port = parse_listen_address(arguments.listen)
        ssl_context = tls_context(arguments)
    except ValueError as error:
        logger.error("Invalid server configuration: %s", error)
        return 1

    atexit.register(cleanup_resources)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    initialize_models()

    protocol = "https" if ssl_context else "http"
    logger.info("Starting OCR API on %s://%s:%s", protocol, host, port)
    app.run(host=host, port=port, threaded=True, ssl_context=ssl_context)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
