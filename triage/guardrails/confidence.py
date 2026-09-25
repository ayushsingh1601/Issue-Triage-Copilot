"""Confidence gating for the human-in-the-loop edge."""
from __future__ import annotations


def needs_human_review(confidence: float, threshold: float = 0.5) -> bool:
    return confidence < threshold
