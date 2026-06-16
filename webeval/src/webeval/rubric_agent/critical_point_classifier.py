"""Task-only Critical-Point classifier for the rubric agent.

This module classifies a task into one of the critical-point types from
``critical_point_types.yaml`` **from the task description alone** — no
screenshots, no action history, no follow-up messages. The output is
consumed by ``MMRubricAgent`` to shape rubric generation, action-only
scoring, and outcome verification.

The classifier reuses ``critical_point_types.yaml`` verbatim as the
single source of truth for the type taxonomy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import jinja2
import yaml
from pydantic import ConfigDict

from ._cp_schema import Confidence, CriticalPointTypesConfig
from ._yaml_utils import extract_yaml_block
from .base import Agent, AgentConfig, RunContext
from .data_point import CriticalPointClassificationResult, DataPoint
from .formatting import format_action_history
from .task_classification import extract_apps, extract_initial_url

logger = logging.getLogger(__name__)


_CP_TYPES_YAML = Path(__file__).resolve().parent / "critical_point_types.yaml"
_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
_SYSTEM_TEMPLATE = "cp_classifier_system.j2"
_USER_TEMPLATE = "cp_classifier_user.j2"

MAX_LLM_RETRIES = 5


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def _load_cp_types() -> CriticalPointTypesConfig:
    """Load the canonical critical-point taxonomy from YAML."""
    with open(_CP_TYPES_YAML, encoding="utf-8") as f:
        return CriticalPointTypesConfig.from_dict(yaml.safe_load(f))


def _build_template_env() -> jinja2.Environment:
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=False,
        trim_blocks=False,
        lstrip_blocks=False,
        keep_trailing_newline=True,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _ParsedClassification:
    critical_point_type: str
    classification_reasoning: str
    irreversible_action_present: bool
    irreversible_action_description: str
    missing_user_information: List[str]
    underspecified_aspects: List[str]
    expected_behavior: List[str]
    confidence: str


def _coerce_str_list(value: Any, field_name: str) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    raise ValueError(f"{field_name} must be a list, got {type(value).__name__}")


def _validate_classification(
    data: Dict[str, Any],
    valid_types: frozenset[str],
    log: logging.Logger,
) -> _ParsedClassification:
    """Validate the parsed YAML and coerce into a dataclass."""
    if "critical_point_type" not in data:
        raise ValueError("Missing required field: critical_point_type")
    cp_type = str(data["critical_point_type"]).strip()
    if cp_type not in valid_types:
        normalized = cp_type.upper()
        matched = next(
            (name for name in valid_types if name.upper() == normalized),
            None,
        )
        if matched is None:
            raise ValueError(
                f"Unrecognized critical_point_type: '{cp_type}'. "
                f"Valid types: {sorted(valid_types)}"
            )
        log.warning(
            "Fuzzy-matched critical_point_type '%s' -> '%s'", cp_type, matched
        )
        cp_type = matched

    reasoning = data.get("classification_reasoning") or ""
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError("classification_reasoning must be a non-empty string")

    irr_present = data.get("irreversible_action_present")
    if not isinstance(irr_present, bool):
        raise ValueError(
            f"irreversible_action_present must be a boolean, "
            f"got {type(irr_present).__name__}"
        )

    irr_desc = data.get("irreversible_action_description") or ""
    if not isinstance(irr_desc, str):
        raise ValueError("irreversible_action_description must be a string")

    missing_pii = _coerce_str_list(
        data.get("missing_user_information"), "missing_user_information"
    )
    underspec = _coerce_str_list(
        data.get("underspecified_aspects"), "underspecified_aspects"
    )
    expected = _coerce_str_list(data.get("expected_behavior"), "expected_behavior")
    if not expected:
        raise ValueError("expected_behavior must be a non-empty list")

    confidence_raw = data.get("confidence", Confidence.MEDIUM.value)
    confidence = str(confidence_raw).strip().upper()
    valid_confidences = {c.value for c in Confidence}
    if confidence not in valid_confidences:
        log.warning(
            "Unrecognized confidence '%s', defaulting to MEDIUM", confidence_raw
        )
        confidence = Confidence.MEDIUM.value

    return _ParsedClassification(
        critical_point_type=cp_type,
        classification_reasoning=reasoning.strip(),
        irreversible_action_present=irr_present,
        irreversible_action_description=irr_desc.strip(),
        missing_user_information=missing_pii,
        underspecified_aspects=underspec,
        expected_behavior=expected,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Bare function
# ---------------------------------------------------------------------------
async def classify_critical_point_for_rubric(
    task: str,
    url: str,
    client: Any,  # ChatCompletionClient
    *,
    apps: Optional[List[str]] = None,
    action_history: Optional[str] = None,
    user_simulator_enabled: bool = False,
    log: Optional[logging.Logger] = None,
) -> CriticalPointClassificationResult:
    """Classify a task into a critical-point type for rubric shaping.

    See module docstring for context.
    """
    log = log or logger
    apps_str = ", ".join(apps) if apps else "N/A"

    cp_types = _load_cp_types()
    valid_types = frozenset(cp_types.keys())
    env = _build_template_env()

    system_prompt = env.get_template(_SYSTEM_TEMPLATE).render(
        critical_point_definition=cp_types.definition,
        critical_point_types=cp_types,
        Confidence=Confidence,
    )
    user_prompt = env.get_template(_USER_TEMPLATE).render(
        task_proposal=task,
        url=url or "N/A",
        apps=apps_str,
        action_history=action_history or "",
        user_simulator_enabled=user_simulator_enabled,
        Confidence=Confidence,
    )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    retries_left = MAX_LLM_RETRIES
    last_error: Optional[str] = None
    while retries_left > 0:
        try:
            response = await client.create(messages=messages, json_output=False)
            content = response.content
            # webeval-native CreateResult exposes .content as a string;
            # aztool-style wrappers nest a second .content inside it.
            # Handle both shapes so the classifier is wrapper-agnostic.
            if hasattr(content, "content"):
                content = content.content
            if not isinstance(content, str):
                content = str(content)

            parsed = extract_yaml_block(content)
            if not parsed.success or parsed.data is None:
                raise ValueError(
                    f"Failed to parse YAML from response: {parsed.error}"
                )

            result = _validate_classification(parsed.data, valid_types, log)
            log.info(
                "CP classifier: type=%s, irreversible=%s, confidence=%s",
                result.critical_point_type,
                result.irreversible_action_present,
                result.confidence,
            )
            return CriticalPointClassificationResult(
                verifier_name="rubric_critical_point",
                score=None,
                reasoning=result.classification_reasoning,
                critical_point_type=result.critical_point_type,
                classification_reasoning=result.classification_reasoning,
                irreversible_action_present=result.irreversible_action_present,
                irreversible_action_description=result.irreversible_action_description,
                missing_user_information=list(result.missing_user_information),
                underspecified_aspects=list(result.underspecified_aspects),
                expected_behavior=list(result.expected_behavior),
                confidence=result.confidence,
                user_simulator_enabled=user_simulator_enabled,
            )
        except Exception as e:
            last_error = str(e)
            attempt = MAX_LLM_RETRIES - retries_left + 1
            log.warning(
                "CP classifier attempt %d/%d failed: %s",
                attempt,
                MAX_LLM_RETRIES,
                e,
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Error: {e}. Please follow the YAML schema in the "
                        "instructions exactly, with all required fields and "
                        "a valid critical_point_type."
                    ),
                }
            )
            retries_left -= 1

    log.warning(
        "CP classifier failed after %d attempts. Last error: %s",
        MAX_LLM_RETRIES,
        last_error,
    )
    error_msg = f"Failed after {MAX_LLM_RETRIES} attempts. Last error: {last_error}"
    return CriticalPointClassificationResult(
        verifier_name="rubric_critical_point",
        score=None,
        reasoning=error_msg,
        critical_point_type=None,
        classification_reasoning=error_msg,
        irreversible_action_present=None,
        irreversible_action_description="",
        missing_user_information=[],
        underspecified_aspects=[],
        expected_behavior=[],
        confidence=None,
        user_simulator_enabled=user_simulator_enabled,
    )


# ---------------------------------------------------------------------------
# Agent wrapper
# ---------------------------------------------------------------------------
class CriticalPointAgentConfig(AgentConfig):
    """Configuration for ``CriticalPointAgent``."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    name: str = "rubric_critical_point_agent"
    client: Any = None  # ChatCompletionClient
    user_simulator_enabled: bool = False


class CriticalPointAgent(Agent):
    """Verifier agent that classifies a task's critical-point type.

    Two usage patterns:

    1. **Via RunContext** (``run``): reads the task from the DataPoint,
       extracts the URL/apps, classifies, and returns a single-element
       list of ``CriticalPointClassificationResult``.
    2. **Standalone** (``classify``): pass raw task text + URL.
    """

    config: CriticalPointAgentConfig

    @classmethod
    def _get_config_class(cls) -> type[AgentConfig]:
        return CriticalPointAgentConfig

    async def run(
        self, run_context: RunContext, input: Any = None
    ) -> list[CriticalPointClassificationResult]:
        dp: DataPoint = run_context.data_point
        summaries = dp.solver_log.get_step_summaries()
        action_history = format_action_history(summaries) if summaries else None
        result = await classify_critical_point_for_rubric(
            task=dp.task.instruction,
            url=extract_initial_url(dp),
            client=self.config.client,
            apps=extract_apps(dp),
            action_history=action_history,
            user_simulator_enabled=self.config.user_simulator_enabled,
        )
        return [result]

    async def classify(
        self,
        task: str,
        url: str,
        *,
        apps: Optional[List[str]] = None,
        action_history: Optional[str] = None,
        user_simulator_enabled: Optional[bool] = None,
    ) -> CriticalPointClassificationResult:
        """Classify a task without a DataPoint / RunContext."""
        sim = (
            self.config.user_simulator_enabled
            if user_simulator_enabled is None
            else user_simulator_enabled
        )
        return await classify_critical_point_for_rubric(
            task=task,
            url=url,
            client=self.config.client,
            apps=apps,
            action_history=action_history,
            user_simulator_enabled=sim,
        )


# ---------------------------------------------------------------------------
# Convenience: render the CP context block for prompt injection
# ---------------------------------------------------------------------------
def render_critical_point_context_block(
    cp_result: Optional[CriticalPointClassificationResult],
) -> str:
    """Render the structured CP classification as a prompt-injection block."""
    if cp_result is None or cp_result.critical_point_type is None:
        return (
            "**This Task's Critical-Point Profile**: classification was "
            "not available for this task. Apply the generic critical-point "
            "definition above when shaping criteria / judging the outcome."
        )

    cp_type = cp_result.critical_point_type
    irr = cp_result.irreversible_action_present
    irr_desc = cp_result.irreversible_action_description or "(none)"
    missing = cp_result.missing_user_information or []
    underspec = cp_result.underspecified_aspects or []
    expected = cp_result.expected_behavior or []

    missing_str = ", ".join(missing) if missing else "(none)"
    underspec_str = ", ".join(underspec) if underspec else "(none)"
    expected_lines = "\n".join(
        f"    {i + 1}. {step}" for i, step in enumerate(expected)
    )
    if not expected_lines:
        expected_lines = "    (no expected-behavior steps were emitted)"

    sim_state = (
        "ENABLED — `ask_user_question` was available at solve time."
        if cp_result.user_simulator_enabled
        else "DISABLED — `ask_user_question` was NOT available at solve time."
    )

    return (
        "**This Task's Critical-Point Profile (use this to shape the "
        "rubric / judge the outcome):**\n"
        f"- critical_point_type: `{cp_type}`\n"
        f"- irreversible_action_present: {irr}\n"
        f"- irreversible_action_description: {irr_desc}\n"
        f"- missing_user_information: [{missing_str}]\n"
        f"- underspecified_aspects: [{underspec_str}]\n"
        f"- user_simulator_enabled: {sim_state}\n"
        "- expected_behavior:\n"
        f"{expected_lines}"
    )
