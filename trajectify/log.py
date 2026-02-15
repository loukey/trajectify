"""Centralised logging configuration."""

import logging
import sys

_FORMAT = "%(asctime)s %(levelname)-8s [%(name)s] %(message)s"


def get_logger(name: str = "trajectify") -> logging.Logger:
    _logger = logging.getLogger(name)
    if not _logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
        _logger.addHandler(handler)
        _logger.setLevel(logging.INFO)
    return _logger


logger = get_logger()
