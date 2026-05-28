"""YAML extraction helper for rubric-agent LLM responses.

Inline-ported from ``agento_next/core/yaml_utils.py``.

Only the bits used by ``critical_point_classifier.py`` are kept —
``extract_yaml_block`` and ``ParseResult``.  ``LiteralStr`` / dump
helpers are not needed in webeval and have been omitted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Generic, TypeVar

import yaml

T = TypeVar("T")


YAML_BLOCK_PATTERN = re.compile(r"```ya?ml\s*\n(.*?)\n```", re.DOTALL)
YAML_BLOCK_UNCLOSED_PATTERN = re.compile(
    r"```ya?ml\s*\n(.+?)(?:\n```|$)", re.DOTALL
)
YAML_UNFENCED_PATTERN = re.compile(
    r"^yaml\s*\n(prompts:\s*\n.*)", re.DOTALL | re.MULTILINE
)
YAML_RAW_PATTERN = re.compile(
    r"^(prompts:\s*\n.*)", re.DOTALL | re.MULTILINE
)


@dataclass(frozen=True)
class ParseResult(Generic[T]):
    """Result of parsing LLM output."""

    success: bool
    data: T | None
    error: str | None = None
    raw_response: str = ""


def extract_yaml_block(response: str) -> ParseResult[dict]:
    """Extract and parse a YAML block from an LLM response.

    Searches for a fenced ``yaml`` code block, then falls back to
    unfenced and raw patterns. As a last resort, tries to parse the
    entire response as YAML.
    """
    match = YAML_BLOCK_PATTERN.search(response)
    if not match:
        match = YAML_BLOCK_UNCLOSED_PATTERN.search(response)
    if not match:
        match = YAML_UNFENCED_PATTERN.search(response)
    if not match:
        match = YAML_RAW_PATTERN.search(response)

    yaml_text = match.group(1) if match else None

    # Last resort: try parsing the entire response as YAML.
    if yaml_text is None:
        try:
            data = yaml.safe_load(response)
            if isinstance(data, dict):
                return ParseResult(success=True, data=data, raw_response=response)
        except yaml.YAMLError:
            pass
        return ParseResult(
            success=False,
            data=None,
            error="No YAML code block found in response.",
            raw_response=response,
        )

    try:
        data = yaml.safe_load(yaml_text)
        if not isinstance(data, dict):
            return ParseResult(
                success=False,
                data=None,
                error=(
                    f"Top-level YAML content must be a mapping/dictionary, "
                    f"got {type(data).__name__ if data is not None else 'null'}."
                ),
                raw_response=response,
            )
        return ParseResult(success=True, data=data, raw_response=response)
    except yaml.YAMLError as e:
        return ParseResult(
            success=False,
            data=None,
            error=f"Invalid YAML syntax: {e}",
            raw_response=response,
        )
