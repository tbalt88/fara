"""Shared formatting + LLM helpers for the rubric-agent package.

Both :class:`MMRubricAgent` (Steps 0-8) and :class:`VerifierAgent`
(Steps 9-10) consume these helpers. They were originally instance/static
methods on ``MMRubricAgent`` and were duplicated on ``VerifierAgent``;
carved out here so the two agents stay decoupled without duplicating
code. Mirrors agento_next's ``rubric_agent/formatting.py``.
"""

from __future__ import annotations

import base64
import io
import json
from typing import Any, Dict, List, Optional

from PIL import Image

from .data_point import StepSummary, UserMessageType


def truncate_observation(text: str, max_chars: int = 1000) -> str:
    """Truncate text keeping first and last halves with a marker between.

    Inlined from agento_next's ``agents/utils.truncate_observation`` so
    this module stays self-contained.
    """
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return (
        text[:half]
        + f"\n... [truncated {len(text) - max_chars} chars] ...\n"
        + text[-half:]
    )


def format_action_history(
    summaries: List[StepSummary], max_url_chars: int = 150
) -> str:
    """Format step summaries into the ``State N / Action N`` text format."""
    lines: List[str] = []
    for s in summaries:
        for msg_type, msg_content in s.user_messages_before:
            if msg_type == UserMessageType.CRITICAL_POINT_RESPONSE:
                lines.append(f"[User Response] {msg_content}")
            elif msg_type == UserMessageType.FOLLOWUP_TASK:
                lines.append(f"[Follow-up Task] {msg_content}")

        url_shortened = s.url.split("?")[0].split("#")[0]
        if len(url_shortened) > max_url_chars:
            url_shortened = url_shortened[:max_url_chars] + "..."

        state_str = f"{url_shortened}, state_description: {s.state_description}"
        action_str = f"{s.action_name}({json.dumps(s.action_args, indent=4)})"

        idx = s.index
        entry = f"State {idx}: {state_str}\nAction {idx}: {action_str}"

        if s.previous_error:
            entry += f"\nError! The above Action {idx} encountered an Error: {s.previous_error}"

        if s.action_name == "run_command" and s.tool_output:
            entry += f"\nCommand Output: {truncate_observation(s.tool_output)}"

        lines.append(entry)

    return "\n".join(lines)


def get_init_url_context(init_url: Optional[str]) -> str:
    if not init_url:
        return ""
    if init_url.lower() in [
        "",
        "bing.com",
        "https://bing.com",
        "https://bing.com/",
        "http://bing.com",
        "https://www.bing.com",
        "http://www.bing.com",
    ]:
        return ""
    return (
        f"\n\nIMPORTANT: The agent MAY have started on the URL: {init_url}\n"
        "This starting URL may be considered part of the task context. "
        "The agent should NOT be penalized for using or assuming information "
        "that is implicit in this starting URL."
    )


def build_scored_rubric_summary(rubric: dict) -> str:
    lines = []
    for j, item in enumerate(rubric["items"]):
        lines.append(
            f'--- Criterion {j}: "{item.get("criterion", f"Criterion {j}")}" ---'
        )
        lines.append(f"Description: {item.get('description', '')}")
        if item.get("reality_notes"):
            lines.append(f"Reality Notes: {item['reality_notes']}")
        if item.get("condition"):
            lines.append(f"Condition: {item['condition']}")
            lines.append(f"Condition Met: {item.get('is_condition_met', 'unknown')}")
        lines.append(f"Max Points: {item.get('max_points', 0)}")
        lines.append(
            f"Baseline Score (action-only): {item.get('earned_points', 'N/A')}/{item.get('max_points', 0)}"
        )
        lines.append(
            f"Final Score (post-image): {item.get('post_image_earned_points', 'N/A')}/{item.get('max_points', 0)}"
        )
        lines.append(
            f'Final Justification: "{item.get("post_image_justification", "N/A")}"'
        )
        if item.get("penalty"):
            lines.append("[PENALTY CRITERION]")
        lines.append("")
    lines.append(
        f"Total: {rubric.get('total_earned_points', 'N/A')}/{rubric.get('total_max_points', 'N/A')}"
    )
    return "\n".join(lines)


def build_all_screenshot_evidence_text(
    rubric: dict,
    evidence_by_criterion: Dict[int, List[Dict[str, Any]]],
    total_screenshots: int,
) -> str:
    lines = []
    for c_idx, criterion in enumerate(rubric["items"]):
        lines.append(
            f'## Criterion {c_idx}: "{criterion.get("criterion", f"Criterion {c_idx}")}"'
        )
        analyses = evidence_by_criterion.get(c_idx, [])
        if not analyses:
            lines.append("No screenshot evidence available for this criterion.")
            lines.append("")
            continue
        for analysis in sorted(analyses, key=lambda x: x.get("screenshot_idx", 0)):
            sn = analysis.get("screenshot_idx", 0)
            lines.append(f"### Screenshot {sn + 1} of {total_screenshots} Analysis:")
            lines.append(f"**Evidence:** {analysis.get('screenshot_evidence', 'N/A')}")
            lines.append(f"**Analysis:** {analysis.get('criterion_analysis', 'N/A')}")
            lines.append(f"**Discrepancies:** {analysis.get('discrepancies', 'N/A')}")
            lines.append(
                f"**Environment Issues Confirmed:** {analysis.get('environment_issues_confirmed', False)}"
            )
            lines.append("")
        lines.append("")
    return "\n".join(lines)


async def call_llm(
    messages: List[Dict[str, Any]],
    client: Any,
    json_output: bool = False,
) -> str:
    """Call an LLM client and return the text content.

    Wrapper-agnostic: handles both webeval-native ``CreateResult`` (whose
    ``.content`` is already a string) and aztool-style wrappers that
    nest a second ``.content`` inside it. ``client.supports_json`` is
    treated as optional — older clients may not expose it.
    """
    supports_json = True
    fn = getattr(client, "supports_json", None)
    if callable(fn):
        try:
            supports_json = bool(fn())
        except TypeError:
            supports_json = bool(fn)
    result = await client.create(
        messages=messages,
        json_output=json_output if supports_json else False,
    )
    content = result.content
    if hasattr(content, "content"):
        content = content.content
    assert isinstance(content, str)
    return content


def encode_image_b64(image: Image.Image, quality: int = 95) -> str:
    """Encode *image* as a base64-encoded JPEG string.

    ``quality`` defaults to 95 (near-lossless). ``subsampling=0`` is
    pinned so chroma is not subsampled - this preserves the sub-pixel
    UI affordances (focus rings, tiny carets) that the 6.4/6.5
    grounding prompts evaluate.
    """
    if image.mode == "RGBA":
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality, subsampling=0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
