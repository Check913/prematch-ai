# -*- coding: utf-8 -*-
"""Configuration des logs (console + fichier rotatif)."""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"


def setup_logging(level: str = "INFO", log_dir: Path | None = None,
                  filename: str = "prematch.log") -> logging.Logger:
    """Initialise le logger racine (idempotent)."""
    root = logging.getLogger()
    if getattr(root, "_prematch_configured", False):
        return root

    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    formatter = logging.Formatter(_FORMAT)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / filename, maxBytes=5 * 1024 * 1024, backupCount=5,
            encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Les libs tierces sont tres bavardes en DEBUG.
    for noisy in ("urllib3", "requests", "httpx", "apscheduler",
                  "telegram", "telegram.ext", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    root._prematch_configured = True  # type: ignore[attr-defined]
    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
