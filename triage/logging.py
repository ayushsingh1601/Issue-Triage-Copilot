"""Silence noisy third-party loggers (openai, mcp, chromadb, httpx) in CLIs."""
from __future__ import annotations

import logging

_NOISY_LOGGERS = [
    "openai",
    "httpx",
    "httpx2",
    "httpcore",
    "httpcore2",
    "mcp",
    "chromadb",
    "chroma",
]


def silence_libraries() -> None:
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)
