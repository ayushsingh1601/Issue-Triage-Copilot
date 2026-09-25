"""Helpers for extracting JSON from LLM and MCP tool responses."""
from __future__ import annotations

import json
import re
from typing import Any


def extract_json(text: str) -> str:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    return text


def parse_json_documents(text: str) -> list[Any]:
    decoder = json.JSONDecoder()
    documents: list[Any] = []
    index = 0
    while index < len(text):
        while index < len(text) and text[index] in " \t\n\r":
            index += 1
        if index >= len(text):
            break
        document, end = decoder.raw_decode(text, index)
        documents.append(document)
        index = end
    return documents


def content_text(result: Any) -> str:
    if isinstance(result, list):
        parts = [block["text"] for block in result if isinstance(block, dict) and "text" in block]
        if parts:
            return "\n".join(parts)
    return str(result)
