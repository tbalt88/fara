"""Task Classification — Unified Verification Check (Step 10).

Standalone module for classifying tasks before execution.  The unified
:func:`classify_task` function classifies the task along two axes:

  1. **Task Ambiguity** (Category 7)
  2. **Invalid Task**   (Category 8)

Only the task description, starting URL/app, and current date are
required — no screenshots, action history, or rubric context.
"""

import json
import logging
from datetime import datetime, timezone
from string import Template
from typing import Any, Dict, List, Optional

from pydantic import ConfigDict

from .base import Agent, AgentConfig, RunContext
from .data_point import DataPoint, TaskAgentResult
from .prompts import CHECK_VALID_TASK_PROMPT

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_MESSAGES: List[Dict[str, str]] = [
    {"role": "system", "content": "You are a helpful AI assistant."}
]

MAX_LLM_RETRIES = 5

# Required top-level fields and their expected types in the verification JSON.
_REQUIRED_FIELDS: Dict[str, type] = {
    "reasoning_is_ambiguous": str,
    "is_ambiguous": bool,
    "ambiguity_codes": list,
    "reasoning_is_invalid": str,
    "is_invalid": bool,
    "invalid_task_codes": list,
}


# ---------------------------------------------------------------------------
# DataPoint helpers
# ---------------------------------------------------------------------------
def _extract_start_url_from_environment_config(data_point: DataPoint) -> str:
    """Extract the configured starting URL/page from task.environment_config."""
    env_cfg = data_point.task.environment_config or {}
    for key in ("init_url", "start_page", "start_url"):
        url = env_cfg.get(key)
        if url:
            return str(url)
    return ""


def extract_initial_url(data_point: DataPoint) -> str:
    """Extract the starting URL, preferring task config over solver log events."""
    configured_url = _extract_start_url_from_environment_config(data_point)
    if configured_url:
        return configured_url
    for event in data_point.solver_log.events:
        url = getattr(event, "url", "") or ""
        if url:
            return url
    return "N/A"


def extract_apps(data_point: DataPoint) -> List[str]:
    """Extract the application name(s) from environment_config.apps."""
    env_cfg = data_point.task.environment_config or {}
    apps = env_cfg.get("apps")
    if apps:
        if isinstance(apps, list):
            return [str(a) for a in apps]
        return [str(apps)]
    url = extract_initial_url(data_point)
    if url and url != "N/A":
        return ["Edge"]
    return ["N/A"]


def extract_app(data_point: DataPoint) -> str:
    """Extract the application name(s) as a comma-separated string.

    .. deprecated:: Use :func:`extract_apps` instead.
    """
    return ", ".join(extract_apps(data_point))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
async def _call_llm(
    messages: list[dict],
    client: Any,  # ChatCompletionClient
    json_output: bool = False,
) -> str:
    """Call an LLM client and return the text content."""
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


def _validate_verification_result(result: dict) -> None:
    """Raise ``ValueError`` if *result* is missing or mis-typed fields."""
    for field, expected_type in _REQUIRED_FIELDS.items():
        if field not in result:
            raise ValueError(f"Missing required field: {field}")
        if not isinstance(result[field], expected_type):
            raise ValueError(
                f"{field} must be {expected_type.__name__}, "
                f"got {type(result[field]).__name__}"
            )
    for rf in ("reasoning_is_ambiguous", "reasoning_is_invalid"):
        if not result[rf]:
            raise ValueError(f"{rf} must be a non-empty string")


# ---------------------------------------------------------------------------
# TaskAgent — agent wrapper
# ---------------------------------------------------------------------------
class TaskAgentConfig(AgentConfig):
    """Configuration for the task verification classification agent."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    name: str = "task_agent"
    client: Any = None  # ChatCompletionClient


class TaskAgent(Agent):
    """Agent that performs task verification classification."""

    config: TaskAgentConfig

    @classmethod
    def _get_config_class(cls) -> type[AgentConfig]:
        return TaskAgentConfig

    async def run(
        self, run_context: RunContext, input: Any = None
    ) -> list[TaskAgentResult]:
        dp = run_context.data_point
        task_desc = dp.task.instruction
        url = extract_initial_url(dp)
        apps = extract_apps(dp)
        result = await classify_task(
            task_desc,
            url,
            self.config.client,
            apps=apps,
        )
        return [result]

    async def classify(
        self,
        task: str,
        url: str,
        *,
        apps: List[str] | None = None,
        date: str | None = None,
    ) -> TaskAgentResult:
        """Classify a task without a DataPoint / RunContext."""
        return await classify_task(
            task,
            url,
            self.config.client,
            apps=apps,
            date=date,
        )


# ---------------------------------------------------------------------------
# Step 10: Unified task verification classification (bare function)
# ---------------------------------------------------------------------------
async def classify_task(
    task: str,
    url: str,
    client: Any,  # ChatCompletionClient
    *,
    apps: List[str] | None = None,
    date: str | None = None,
    system_messages: Optional[List[Dict[str, str]]] = None,
) -> TaskAgentResult:
    """Unified task verification classification across ambiguity and validity."""
    if apps is None:
        apps = ["N/A"]
    if date is None:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    apps_str = ", ".join(apps) if apps else "N/A"

    sys_msgs = (
        system_messages if system_messages is not None else DEFAULT_SYSTEM_MESSAGES
    )
    prompt = Template(CHECK_VALID_TASK_PROMPT).substitute(
        task_definition=task,
        url=url or "N/A",
        apps=apps_str,
        date=date,
    )
    messages = list(sys_msgs) + [{"role": "user", "content": prompt}]

    retries_left = MAX_LLM_RETRIES
    last_error = None
    while retries_left > 0:
        try:
            response_text = await _call_llm(messages, client, json_output=True)
            result = json.loads(response_text)
            _validate_verification_result(result)
            is_flagged = result.get("is_ambiguous") or result.get("is_invalid")
            logger.info(
                "Task verification result: is_ambiguous=%s, is_invalid=%s",
                result["is_ambiguous"],
                result["is_invalid"],
            )
            return TaskAgentResult(
                verifier_name="task_verification",
                score=0.0 if is_flagged else 1.0,
                reasoning="FLAGGED" if is_flagged else "OK",
                **result,
            )
        except Exception as e:
            last_error = str(e)
            attempt = MAX_LLM_RETRIES - retries_left + 1
            logger.error(
                f"Error in task verification classification (attempt {attempt}): {e}"
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Error: {e}. Please ensure your output follows the exact "
                        "JSON format specified with all required fields."
                    ),
                }
            )
            retries_left -= 1

    logger.warning(
        "Failed task verification classification after %d attempts. Last error: %s",
        MAX_LLM_RETRIES,
        last_error,
    )
    error_msg = f"Failed after {MAX_LLM_RETRIES} attempts. Last error: {last_error}"
    return TaskAgentResult(
        verifier_name="task_verification",
        score=None,
        reasoning=error_msg,
        reasoning_is_ambiguous=error_msg,
        is_ambiguous=None,
        reasoning_is_invalid=error_msg,
        is_invalid=None,
    )
