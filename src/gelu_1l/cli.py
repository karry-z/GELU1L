import argparse
import logging
from pathlib import Path

from gelu_1l.constants import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CACHE_DIR,
    DEFAULT_SAE_RUN,
    DEFAULT_SEQ_LEN,
)

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--sae-run", type=int, default=DEFAULT_SAE_RUN)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level), format=LOG_FORMAT)


def feature_dir(artifacts_dir: Path, feature_id: int) -> Path:
    return artifacts_dir / f"feature_{feature_id}"
