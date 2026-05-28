"""LLM-call helpers shared between the rubric / human-feedback agents and
the user-simulator retry gates.

Kept as a leaf module (no ``agento_next`` deps) so consumers in both
``agents/rubric_agent`` and ``data_gen`` can import from here without
re-introducing the ``human_feedback_agent`` ↔ ``retry_feedback`` cycle
that ``__init__.py`` was emptied to break.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


async def llm_call_expect_json(
    client: Any,
    messages: List[Any],
    required_keys: List[str],
    *,
    max_retries: int = 5,
    gate_name: str = "LLM call",
    json_output: bool = False,
    append_error_nudge: bool = False,
) -> Dict[str, Any]:
    """Call ``client.create(messages=...)`` and parse the response as a JSON
    object containing every name in ``required_keys``.

    Retries up to ``max_retries`` times on any of:

    - the transport raising,
    - the response failing to parse as JSON,
    - the parsed value not being a dict,
    - any name in ``required_keys`` missing from the parsed dict.

    When ``append_error_nudge=True``, an in-place ``{"role": "user",
    "content": "Error: …"}`` turn is appended to ``messages`` after each
    failed attempt so the LLM gets a chance to correct itself in the
    next call.  This matches the original hand-rolled retry pattern in
    ``mm_rubric_agent.py``.  Off by default (the user-simulator gates
    rely on the prompt + ``json_output=True`` to be retry-safe).

    Logs a warning for each failed attempt with diagnostic context and
    raises :class:`RuntimeError` once all retries are exhausted.  Callers
    that previously caught :class:`Exception` to fall back to a sentinel
    value (``None``, ``llm_error`` status, …) keep that contract by
    catching the post-retry ``RuntimeError`` themselves; callers that let
    exceptions propagate get a clean signal in
    ``failed/<task_id>/exception.txt``.
    """
    last_error: Optional[str] = None

    def _log_and_nudge(err: str) -> None:
        nonlocal last_error
        last_error = err
        logger.warning(
            "%s attempt %d/%d: %s",
            gate_name,
            attempt,
            max_retries,
            err,
        )
        if append_error_nudge:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Error: {err}. Please ensure your output follows "
                        "the exact JSON format specified with all required "
                        "fields."
                    ),
                }
            )

    for attempt in range(1, max_retries + 1):
        try:
            create_kwargs: Dict[str, Any] = {"messages": messages}
            if json_output:
                create_kwargs["json_output"] = True
            result = await client.create(**create_kwargs)
            response_text = (result.content.content or "").strip()
        except Exception as e:
            # Don't nudge on transport errors — the LLM never saw the
            # prompt, so an "Error: ConnectionError" turn is just noise.
            last_error = f"client.create raised {type(e).__name__}: {e}"
            logger.warning(
                "%s attempt %d/%d: %s",
                gate_name,
                attempt,
                max_retries,
                last_error,
            )
            continue

        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError as e:
            _log_and_nudge(
                f"JSONDecodeError: {e}; response[:300]={response_text[:300]!r}"
            )
            continue

        if not isinstance(parsed, dict):
            _log_and_nudge(
                f"Expected JSON object, got {type(parsed).__name__}: "
                f"{str(parsed)[:200]!r}"
            )
            continue

        missing = [k for k in required_keys if k not in parsed]
        if missing:
            _log_and_nudge(
                f"Missing required keys {missing}; got keys {list(parsed.keys())}"
            )
            continue

        return parsed

    raise RuntimeError(
        f"{gate_name} failed after {max_retries} attempts. Last error: {last_error}"
    )
