"""Steps 9–10 of the rubric verification pipeline (renamed
``HumanFeedbackAgent`` → :class:`VerifierAgent`, Step 11 dropped).

In agento_next, :class:`HumanFeedbackAgent` was composed into
:class:`MMRubricAgent` via multiple inheritance. Here it is a standalone
:class:`Agent` subclass that consumes a scored rubric (the output of
``MMRubricAgent``) and runs Steps 9a/9b/10:

- **Step 9a** :meth:`_first_point_of_failure_analysis` and its
  programmatic-detection helpers (6.1/6.2 tool-interaction,
  6.4/6.5 visual grounding, 9.1/9.2 harness analysis).
- **Step 9b** :meth:`_classify_task_with_trajectory` —
  trajectory-informed task verification.
- **Step 10** :meth:`_classify_task` — wraps
  :func:`task_classification.classify_task`.

Step 11 (synthetic human-voice feedback) has been removed entirely.

Primary entry point: :meth:`VerifierAgent.verify`. Pass the scored rubric
dict (Steps 0–8 output from :class:`MMRubricAgent`), the
``outcome_verification`` dict from the same run, and the original input
dict, and it returns a sub-dict containing
``step9_first_point_of_failure``, ``step9b_task_verification_with_trajectory``,
and ``step10_task_verification``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional, Set, Tuple

import imagehash
from PIL import Image, ImageDraw
from pydantic import ConfigDict

from .base import Agent, AgentConfig, RunContext
from .error_taxonomy_loader import extract_subcategory, get_harness_code_info
from .formatting import (
    build_all_screenshot_evidence_text,
    build_scored_rubric_summary,
    call_llm,
    encode_image_b64,
    get_init_url_context,
)

from .prompts import (
    CHECK_VALID_TASK_WITH_TRAJECTORY_PROMPT,
    FINE_GRAINED_GROUNDING_PROMPT,
    FIRST_POINT_OF_FAILURE_PROMPT,
    GROUNDING_CROP_DECISION_PROMPT,
    _GROUNDING_ACCURACY_WIDE_ONLY,
    _GROUNDING_ACCURACY_WITH_ZOOM,
    _GROUNDING_ERROR_CODES,
    _GROUNDING_NO_POST_IMAGE_DESCRIPTION,
    _GROUNDING_POST_IMAGE_DESCRIPTION,
    _GROUNDING_PREAMBLE_WIDE_ONLY,
    _GROUNDING_PREAMBLE_WITH_ZOOM,
)
from .task_classification import (
    _REQUIRED_FIELDS,
    _validate_verification_result,
    classify_task,
)
from .llm_helpers import llm_call_expect_json

logger = logging.getLogger(__name__)


def _get_coordinate_actions(action_definitions: Dict[str, Set[str]]) -> Set[str]:
    """Return action names whose parameters include ``x`` and ``y``.

    Inlined from agento_next's ``computer_agent.tools.get_coordinate_actions``
    so this module is self-contained.
    """
    return {name for name, args in action_definitions.items() if {"x", "y"} <= args}


# ---------------------------------------------------------------------------
# Grounding-check error metadata (module-level for class-attribute init)
# ---------------------------------------------------------------------------
_GROUNDING_CATEGORY_MAP: dict[str, str] = {
    "3": "Execution & Strategy",
    "6": "Tool Interaction",
    "9": "Harness Analysis",
}
_GROUNDING_IMPACT_MAP: dict[str, str] = {
    "6.5": (
        "The action references an object or element that does not "
        "exist on the current screenshot."
    ),
    "6.4": (
        "The action may have interacted with the wrong UI element "
        "due to spatial imprecision."
    ),
    "3.5": (
        "The agent targeted the correct UI element but the action "
        "produced no observable effect, leaving the relevant sub-goal "
        "incomplete."
    ),
    "9.1": (
        "The grounding error is genuine — the agent saw the same "
        "visual context in the previous screenshot and still missed "
        "the target."
    ),
    "9.2": (
        "The grounding error was a harness artifact — the screenshot "
        "used for the grounding check differed from what the agent "
        "actually saw at decision time."
    ),
}


class VerifierAgentConfig(AgentConfig):
    """Configuration for :class:`VerifierAgent` (Steps 9a/9b/10)."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    name: str = "verifier_agent"

    # LLM clients — callers pass concrete ChatCompletionClient instances.
    # Same naming convention as :class:`MMRubricAgentConfig` so a single
    # pair of clients can be threaded through both agents.
    o4mini_client: Any = None
    gpt5_client: Any = None

    # Pipeline knobs (subset of MMRubricAgentConfig's relevant to 9a/9b/10).
    max_iters: int = 5
    enable_fine_grained_grounding_detection: bool = False

    # JPEG quality for the base64-encoded images sent to grounding (6.4/6.5)
    # prompts. Default 95 (near-lossless) preserves sub-pixel UI affordances
    # (focus rings, tiny carets) that the 6.4/6.5 grounding prompts evaluate.
    grounding_image_quality: int = 95

    # Action definitions for failure-point analysis (Step 9a).
    # Maps action_name -> set(arg_names). When None, the LLM judgment is
    # still produced but programmatic 6.1/6.2 tool-interaction checks are
    # skipped.
    action_definitions: Optional[Dict[str, Set[str]]] = None


class VerifierAgent(Agent):
    """Steps 9–10 of the multimodal rubric pipeline.

    Consumes the scored rubric + outcome verification produced by
    :class:`MMRubricAgent` and produces the failure-analysis /
    task-verification sub-results.
    """

    DEFAULT_SYSTEM_MESSAGES = [
        {"role": "system", "content": "You are a helpful AI assistant."}
    ]

    config: VerifierAgentConfig  # type: narrow from AgentConfig

    @classmethod
    def _get_config_class(cls) -> type[AgentConfig]:
        return VerifierAgentConfig

    @property
    def _o4mini_client(self) -> Any:
        return self.config.o4mini_client

    @property
    def _gpt5_client(self) -> Any:
        return self.config.gpt5_client

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------
    async def verify(
        self,
        rubric_dict: dict,
        outcome_dict: dict,
        input_dict: dict,
        evidence_by_criterion: Optional[Dict[int, List[Dict]]] = None,
        total_screenshots: int = 0,
        run_context: Optional[RunContext] = None,
    ) -> dict:
        """Run Steps 9a, 9b, and 10 on top of a scored rubric.

        Args:
            rubric_dict: Scored rubric (output of
                :meth:`MMRubricAgent._generate_reply`).
            outcome_dict: ``outcome_verification`` dict from the same
                run (``rubric_dict["outcome_verification"]``).
            input_dict: The input dict that was fed into
                :meth:`MMRubricAgent._generate_reply` — used for
                ``task``, ``action_history``, ``init_url``, ``apps``,
                ``predicted_output``, ``step_actions``,
                ``screenshots_dir``.
            evidence_by_criterion: Mapping of criterion-index to list of
                per-screenshot analysis dicts. When omitted, the method
                tries to read it from
                ``rubric_dict["intermediate_mm_rubric_steps"]
                ["step4_evidence_by_criterion"]``.
            total_screenshots: Total screenshots considered for
                evidence summaries. Read from
                ``rubric_dict["intermediate_mm_rubric_steps"]
                ["step1_num_screenshots"]`` when not provided.

        Returns:
            Dict with three keys:
                - ``step9_first_point_of_failure``
                - ``step9b_task_verification_with_trajectory``
                - ``step10_task_verification``
        """
        task: str = input_dict["task"]
        action_history: str = input_dict["action_history"]
        predicted_output: str = input_dict.get("predicted_output", "")
        init_url: str = input_dict.get("init_url", "")
        apps_list: list = input_dict.get("apps", [])
        step_actions: Optional[List[Dict[str, Any]]] = input_dict.get("step_actions")
        screenshots_dir: Optional[str] = input_dict.get("screenshots_dir")
        action_definitions = (
            input_dict.get("action_definitions") or self.config.action_definitions
        )

        init_url_context = get_init_url_context(init_url)
        apps_str = ", ".join(apps_list) if apps_list else "N/A"

        intermediate = (
            rubric_dict.get("intermediate_mm_rubric_steps") or {}
        )

        if evidence_by_criterion is None:
            raw_evidence = intermediate.get("step4_evidence_by_criterion", {}) or {}
            evidence_by_criterion = {int(k): v for k, v in raw_evidence.items()}
        if not total_screenshots:
            total_screenshots = int(intermediate.get("step1_num_screenshots", 0) or 0)

        # Step 9a — points of failure
        step9 = await self._first_point_of_failure_analysis(
            rubric_dict,
            evidence_by_criterion,
            task,
            init_url_context,
            action_history,
            predicted_output,
            outcome_result=outcome_dict,
            total_screenshots=total_screenshots,
            action_definitions=action_definitions,
            step_actions=step_actions,
            screenshots_dir=screenshots_dir,
        )

        # Step 9b — trajectory-informed task verification
        step9b = await self._classify_task_with_trajectory(
            rubric_dict,
            evidence_by_criterion,
            task,
            init_url_context,
            action_history,
            predicted_output,
            outcome_result=outcome_dict,
            total_screenshots=total_screenshots,
            apps=apps_str,
        )

        # Step 10 — unified task verification
        step10 = await self._classify_task(task, init_url, apps=apps_list)

        return {
            "step9_first_point_of_failure": step9,
            "step9b_task_verification_with_trajectory": step9b,
            "step10_task_verification": step10,
        }

    # ------------------------------------------------------------------
    # Step 9a class-level constants
    # ------------------------------------------------------------------
    _STEP_NUMBERS_RE = re.compile(r"^\d+(-\d+)?(,\d+)*$")

    # Error codes whose detection is owned by programmatic/visual detectors.
    # LLM-emitted failure_points with these codes are stripped before the
    # programmatic detectors re-inject their own (more reliable) versions.
    #   6.1 — _detect_tool_interaction_errors
    #   6.2 — _detect_tool_interaction_errors
    #   6.4 — _detect_fine_grained_grounding_errors (visual)
    #   6.5 — _detect_fine_grained_grounding_errors (visual)
    _PROGRAMMATIC_ERROR_CODES: frozenset[str] = frozenset({"6.1", "6.2", "6.4", "6.5"})

    # Metadata for error codes emitted by _detect_fine_grained_grounding_errors.
    # error_type (name) is pulled from the taxonomy .md via extract_subcategory;
    # error_category and impact are grounding-check-specific (module-level maps).
    # 9.x harness codes use get_harness_code_info() for their label.
    _GROUNDING_ERROR_INFO: dict[str, dict[str, str]] = {
        **{
            code: {
                "error_category": _GROUNDING_CATEGORY_MAP[code.split(".")[0]],
                "error_type": extract_subcategory(code)[0],
                "impact": _GROUNDING_IMPACT_MAP[code],
            }
            for code in _GROUNDING_ERROR_CODES
        },
        **{
            code: {
                "error_category": _GROUNDING_CATEGORY_MAP[code.split(".")[0]],
                "error_type": get_harness_code_info(code)["harness_label"],
                "impact": _GROUNDING_IMPACT_MAP[code],
            }
            for code in ("9.1", "9.2")
        },
    }

    # Maximum dhash Hamming distance for two zoom-in crops to be treated
    # as visually identical, allowing the harness recheck LLM call to be
    # skipped when a 6.4 error is detected.  The crops are 500×500 px
    # centred on the click coordinate.  dhash(hash_size=8) → 64 bits;
    # threshold 3 ≈ 5% tolerance — enough for rendering jitter without
    # conflating genuinely different content.
    _HARNESS_HASH_THRESHOLD: int = 3
    _HARNESS_HASH_SIZE: int = 8

    # Step 9a: Points of Failure Analysis
    # ------------------------------------------------------------------
    async def _first_point_of_failure_analysis(
        self,
        rubric: dict,
        evidence_by_criterion: Dict[int, List[Dict]],
        task: str,
        init_url_context: str,
        action_history: str,
        predicted_output: str,
        outcome_result: dict,
        total_screenshots: int = 0,
        action_definitions: Optional[Dict[str, Set[str]]] = None,
        step_actions: Optional[List[Dict[str, Any]]] = None,
        screenshots_dir: Optional[str] = None,
    ) -> dict:
        """Step 9a: Failure Point Analysis — identify all failure points in the
        trajectory.  The first (earliest) point of failure is computed
        programmatically from the LLM's ``failure_points`` list.

        Tool interaction errors 6.1 (Invalid invocation) and 6.2
        (Hallucinated action) are also detected programmatically from
        ``step_actions`` when available, and injected into the result.

        Fine-grained grounding errors (6.4) and grounding intent-action
        mismatches (6.5) are detected by visual
        verification of coordinate-bearing actions when ``screenshots_dir``
        is provided.

        Uses 1 gpt-5 call (with up to 5 retry attempts on validation errors).

        Args:
            action_definitions: Mapping of ``{action_name: set(arg_names)}``
                describing the agent's available tools.  If ``None`` and
                no ``self.config.action_definitions`` is configured,
                programmatic 6.1/6.2 tool-interaction checks are skipped
                (the LLM judgment is still produced).

        Returns:
            Dict with ``reasoning``, ``has_failure``, ``failure_points``,
            ``first_failure_step``, ``first_failure_summary``.
        """
        if action_definitions is None:
            action_definitions = self.config.action_definitions
        if action_definitions is None:
            # No fallback tool registry in fara — skip programmatic
            # 6.1/6.2 checks rather than crash. The LLM judgment in
            # ``failure_points`` is still produced.
            action_definitions = {}

        rubric_summary = build_scored_rubric_summary(rubric)
        evidence_summary = build_all_screenshot_evidence_text(
            rubric, evidence_by_criterion, total_screenshots
        )

        outcome_success = outcome_result.get("output_success")
        if outcome_success is True:
            outcome_label = "SUCCESS"
        elif outcome_success is False:
            outcome_label = "FAILURE"
        else:
            outcome_label = "UNKNOWN"
        outcome_text = (
            f"Task outcome: {outcome_label}\n"
            f"Primary intent: {outcome_result.get('primary_intent', 'N/A')}\n"
            f"Reasoning: {outcome_result.get('reasoning', 'N/A')}"
        )

        # Build prompt variables from action_definitions
        action_space_str = ", ".join(f"`{a}`" for a in action_definitions)
        action_defs_lines = []
        for act_name in sorted(action_definitions):
            args_str = ", ".join(sorted(action_definitions[act_name]))
            action_defs_lines.append(f"  - `{act_name}({args_str})`")
        action_definitions_text = "\n".join(action_defs_lines)

        prompt = Template(FIRST_POINT_OF_FAILURE_PROMPT).substitute(
            task_definition=task,
            init_url_context=init_url_context,
            action_history=action_history,
            predicted_output=predicted_output or "N/A",
            rubric_summary=rubric_summary,
            evidence_summary=evidence_summary,
            outcome_verification=outcome_text,
            action_space=action_space_str,
            action_definitions_text=action_definitions_text,
        )
        messages = self.DEFAULT_SYSTEM_MESSAGES + [{"role": "user", "content": prompt}]

        max_iters = self.config.max_iters
        last_error = None
        while max_iters > 0:
            try:
                response_text = await call_llm(
                    messages, self._gpt5_client, json_output=True
                )
                result = json.loads(response_text)

                # -- Validate top-level fields --
                if "reasoning" not in result:
                    raise ValueError("Missing required field: reasoning")
                if not isinstance(result["reasoning"], str) or not result["reasoning"]:
                    raise ValueError("reasoning must be a non-empty string")
                if "has_failure" not in result:
                    raise ValueError("Missing required field: has_failure")
                if not isinstance(result["has_failure"], bool):
                    raise ValueError(
                        f"has_failure must be a boolean, got {type(result['has_failure']).__name__}"
                    )
                if "failure_points" not in result:
                    raise ValueError("Missing required field: failure_points")
                if not isinstance(result["failure_points"], list):
                    raise ValueError(
                        f"failure_points must be a list, got {type(result['failure_points']).__name__}"
                    )

                # -- Validate each failure point --
                for i, fp in enumerate(result["failure_points"]):
                    required_fields = [
                        "step_numbers",
                        "error_code",
                        "error_category",
                        "error_type",
                        "what_happened",
                        "agent_reasoning",
                        "evidence",
                        "impact",
                    ]
                    missing = [f for f in required_fields if f not in fp]
                    if missing:
                        raise ValueError(
                            f"failure_points[{i}] missing fields: {', '.join(missing)}"
                        )

                    # Validate step_numbers format: "INT", "INT-INT", or "INT,INT,..."
                    sn = str(fp["step_numbers"]).replace(" ", "")
                    if not self._STEP_NUMBERS_RE.match(sn):
                        raise ValueError(
                            f'failure_points[{i}].step_numbers must be "INT", '
                            f'"INT-INT", or "INT,INT,..." (e.g. "5", "5-7", or '
                            f'"5,8,12"), got "{fp["step_numbers"]}". '
                            f"Never use N/A or descriptive text."
                        )
                    fp["step_numbers"] = sn

                # -- Strip LLM-emitted codes owned by programmatic detectors --
                # The LLM sees the full taxonomy for context (helps it
                # correctly classify 6.5 vs neighbours), but its 6.1/6.2/6.4/6.5
                # outputs are unreliable compared to the dedicated detectors
                # that re-inject them below.
                # When fine-grained grounding detection is disabled, keep
                # LLM-emitted 6.4/6.5 codes (they won't be re-injected).
                strip_codes = self._PROGRAMMATIC_ERROR_CODES
                if not self.config.enable_fine_grained_grounding_detection:
                    strip_codes = strip_codes - {"6.4", "6.5"}
                result["failure_points"] = [
                    fp
                    for fp in result["failure_points"]
                    if fp.get("error_code") not in strip_codes
                ]

                # -- Inject programmatic 6.1/6.2 errors --
                if step_actions is not None and action_definitions:
                    prog_fps = self._detect_tool_interaction_errors(
                        step_actions, action_definitions
                    )
                    if prog_fps:
                        existing = {
                            (fp.get("step_numbers"), fp.get("error_code"))
                            for fp in result["failure_points"]
                        }
                        for pfp in prog_fps:
                            key = (pfp["step_numbers"], pfp["error_code"])
                            if key not in existing:
                                result["failure_points"].append(pfp)
                        result["failure_points"].sort(
                            key=lambda fp: self._parse_first_step_number(
                                fp.get("step_numbers", "")
                            )
                        )
                        if result["failure_points"]:
                            result["has_failure"] = True

                # -- Inject visual grounding errors (6.4, 6.5)
                #    and harness analysis results (9.1, 9.2) --
                if (
                    self.config.enable_fine_grained_grounding_detection
                    and step_actions is not None
                    and screenshots_dir
                ):
                    try:
                        grounding_fps = (
                            await self._detect_fine_grained_grounding_errors(
                                step_actions,
                                screenshots_dir,
                                action_definitions=action_definitions,
                            )
                        )
                        if grounding_fps:
                            existing = {
                                (fp.get("step_numbers"), fp.get("error_code"))
                                for fp in result["failure_points"]
                            }
                            for gfp in grounding_fps:
                                key = (gfp["step_numbers"], gfp["error_code"])
                                if key in existing:
                                    # Same error at same step found by both
                                    # LLM rubric and grounding check —
                                    # replace with grounding version (has
                                    # visual evidence) and mark as
                                    # step-level error.
                                    ec = gfp["error_code"]
                                    sn = gfp["step_numbers"]
                                    gfp["step_level_error"] = True
                                    result["failure_points"] = [
                                        (
                                            gfp
                                            if fp.get("step_numbers") == sn
                                            and fp.get("error_code") == ec
                                            else fp
                                        )
                                        for fp in result["failure_points"]
                                    ]
                                elif key not in existing:
                                    # New error found only by grounding
                                    # check — keep the marker.
                                    result["failure_points"].append(gfp)
                            result["failure_points"].sort(
                                key=lambda fp: self._parse_first_step_number(
                                    fp.get("step_numbers", "")
                                )
                            )
                            if result["failure_points"]:
                                result["has_failure"] = True
                    except (OSError, json.JSONDecodeError) as e:
                        logger.warning(
                            "Fine-grained grounding detection failed: %s",
                            e,
                            exc_info=True,
                        )

                # -- Compute first_failure_step programmatically --
                first_failure_step, first_failure_summary = self._compute_first_failure(
                    result["failure_points"]
                )
                result["first_failure_step"] = first_failure_step
                result["first_failure_summary"] = first_failure_summary

                logger.info(
                    "Points of failure result: has_failure=%s, "
                    "first_failure_step=%s, num_failure_points=%d",
                    result["has_failure"],
                    result["first_failure_step"],
                    len(result["failure_points"]),
                )
                return result
            except Exception as e:
                last_error = str(e)
                logger.error(
                    "Error in points of failure analysis (attempt %d): %s",
                    self.config.max_iters + 1 - max_iters,
                    e,
                )
                messages.append(
                    {
                        "role": "user",
                        "content": f"Error: {e}. Please ensure your output follows the exact JSON format specified with all required fields.",
                    }
                )
                max_iters -= 1

        logger.warning(
            "Failed points of failure analysis after %d attempts. Last error: %s",
            self.config.max_iters,
            last_error,
        )
        return {
            "reasoning": f"Failed after {self.config.max_iters} attempts. Last error: {last_error}",
            "has_failure": False,
            "failure_points": [],
            "first_failure_step": None,
            "first_failure_summary": "",
        }

    @staticmethod
    def _parse_first_step_number(step_numbers: str) -> int:
        """Parse the minimum step number from a ``step_numbers`` string.

        Handles formats: ``"5"``, ``"5-7"``, ``"5,8,12"``, ``"8,5"``, ``"3-7,12"``.
        For ranges, takes the min of endpoints.  For comma-separated lists,
        takes the global minimum across all entries.
        Returns a large sentinel value if parsing fails.
        """
        try:
            step_numbers = step_numbers.strip()
            values: list[int] = []
            for token in step_numbers.split(","):
                token = token.strip()
                if "-" in token:
                    values.extend(int(p.strip()) for p in token.split("-"))
                else:
                    values.append(int(token))
            return min(values) if values else 999999
        except (ValueError, IndexError):
            return 999999

    @staticmethod
    def _compute_first_failure(
        failure_points: List[Dict],
    ) -> Tuple[Optional[int], str]:
        """Compute ``first_failure_step`` and ``first_failure_summary`` from
        the LLM's ``failure_points`` list.

        Priority: first failure of any kind by step number (the LLM no longer
        outputs severity tiers, so we simply pick the earliest failure point).
        If no failures at all, returns ``(None, "")``.
        """
        if not failure_points:
            return None, ""

        def sort_key(fp: Dict) -> int:
            return VerifierAgent._parse_first_step_number(
                fp.get("step_numbers", "")
            )

        sorted_fps = sorted(failure_points, key=sort_key)

        fp = sorted_fps[0]
        step = VerifierAgent._parse_first_step_number(fp.get("step_numbers", ""))
        summary = (
            f"[{fp.get('error_code', '')}] {fp.get('error_type', '')}: "
            f"{fp.get('what_happened', '')}"
        )
        return step if step != 999999 else None, summary

    @staticmethod
    def _detect_tool_interaction_errors(
        step_actions: List[Dict[str, Any]],
        action_definitions: Dict[str, Set[str]],
    ) -> List[Dict]:
        """Programmatically detect 6.1 (Invalid invocation) and 6.2
        (Hallucinated action) errors by comparing each step's action
        name and argument keys against ``action_definitions``.

        Returns a list of failure-point dicts matching the schema used
        by the LLM's ``failure_points`` list, with an extra
        ``"programmatic": True`` flag.
        """
        errors: List[Dict] = []
        valid_action_names = set(action_definitions.keys())

        for sa in step_actions:
            step = sa["step_number"]
            name = sa["action_name"]
            # ``thoughts`` is added to action arguments by Fara/GPT54 system
            # logging code (e.g. gpt54_agent_browser._build_trajectory) but
            # isn't part of any action's formal tool schema. ``action`` is
            # the union-tool-dict discriminator (always equal to the function
            # name) — also auto-injected, never a content arg. ``_call_id``
            # is webeval's internal correlation key. Exclude all three from
            # 6.1 validation.
            args_keys = set(sa["action_args_keys"]) - {"_call_id", "thoughts", "action"}

            if not name:
                continue

            if name not in valid_action_names:
                errors.append(
                    {
                        "step_numbers": str(step),
                        "error_code": "6.2",
                        "error_category": "Tool Interaction",
                        "error_type": "Hallucinated action",
                        "what_happened": (
                            f"The agent invoked `{name}` which does not exist "
                            f"in the available action space "
                            f"[{', '.join(sorted(valid_action_names))}]."
                        ),
                        "agent_reasoning": "",
                        "evidence": (
                            f"Action `{name}` is not defined in the tool schema."
                        ),
                        "impact": "The action could not be executed as intended.",
                        "programmatic": True,
                    }
                )
            else:
                expected_args = action_definitions[name]
                unknown_args = args_keys - expected_args
                if unknown_args:
                    errors.append(
                        {
                            "step_numbers": str(step),
                            "error_code": "6.1",
                            "error_category": "Tool Interaction",
                            "error_type": "Invalid invocation",
                            "what_happened": (
                                f"The agent called `{name}` with unknown "
                                f"argument(s): {', '.join(sorted(unknown_args))}. "
                                f"Valid arguments are: "
                                f"{', '.join(sorted(expected_args))}."
                            ),
                            "agent_reasoning": "",
                            "evidence": (
                                f"Arguments {sorted(unknown_args)} are not in "
                                f"the schema for `{name}`."
                            ),
                            "impact": (
                                "The action may not execute correctly due to "
                                "invalid arguments."
                            ),
                            "programmatic": True,
                        }
                    )

        return errors

    # ------------------------------------------------------------------
    # 6.4/6.5: Fine-grained grounding error detection (visual)
    # ------------------------------------------------------------------
    async def _detect_fine_grained_grounding_errors(
        self,
        step_actions: List[Dict[str, Any]],
        screenshots_dir: str,
        action_definitions: Optional[Dict[str, Set[str]]] = None,
    ) -> List[Dict]:
        """Visually verify that coordinate-bearing actions land on the intended target.

        For each action whose tool definition includes ``x`` and ``y``
        parameters (derived dynamically from *action_definitions*), loads
        the pre-action screenshot, overlays concentric circles at the
        emitted (x, y), and asks the LLM whether the point is on the
        intended element.

        The check proceeds in two LLM calls per step:

        1. **Crop decision** (``GROUNDING_CROP_DECISION_PROMPT``): the raw
           (unannotated) pre-action screenshot and the agent's intent are
           sent to the LLM, which returns ``{"should_crop": bool}``.  This
           controls whether zoom-in crops are included in the main call.

        2. **Main grounding evaluation** (``FINE_GRAINED_GROUNDING_PROMPT``):
           sends a variable number of images depending on the crop
           decision and post-screenshot availability:

           - **Wide view** — full pre-action screenshot with concentric
             circles drawn at (x, y).
           - **Zoom marked** (if should_crop) — 500×500 crop centred on
             (x, y) with concentric circles.
           - **Zoom unmarked** (if should_crop) — same 500×500 crop
             without annotations, for clean reference.
           - **Post-action screenshot** (if ``post_screenshot_path``
             exists) — the screenshot taken after the action executed.

           Image count: 1–4 (wide only through wide + 2 zooms + post).

        The LLM classifies the action and returns **all** applicable error
        codes (multiple may apply simultaneously), or an empty list if correct:

        - **6.5** (Grounding intent-action mismatch) — target not present on
          the screenshot, or the agent references an element that does not exist.
        - **6.4** (Fine-grained grounding error) — target is present and
          unambiguous but coordinates miss it spatially.

        When both 6.5 and 6.4 are returned, 6.5 takes precedence as the
        primary failure point. The 6.4 still triggers a harness recheck,
        yielding 6.5 + 9.1 or 6.5 + 9.2.

        **Harness recheck (9.x classification):**

        When a 6.4 error is detected, the harness recheck first compares the
        zoom-in crops (500×500 region around the click coordinate) from the
        current pre-action screenshot and the previous action's post-action
        screenshot using perceptual hashing (dhash).  If the crops are
        identical (Hamming distance ≤ ``_HARNESS_HASH_THRESHOLD``), the
        second LLM call is skipped entirely — no harness error is emitted.

        Otherwise the grounding prompt is re-run using the *previous*
        action's post-action screenshot as the base image (this was the image
        the agent actually saw before predicting the action, prior to a
        harness bug fix). The recheck result is classified as:

        - **9.1** (Harness + Grounding Error) — 6.4 persists with the
          previous screenshot (with or without 6.5), indicating a genuine
          grounding miss.
        - **9.2** (Harness only) — 6.4 disappears with the previous
          screenshot (only 6.5 or neither returned), indicating the error
          was a harness artifact.

        These 9.x results are tracked separately and do **not** contribute to
        the trajectory's main error taxonomy.

        Returns:
            List of failure-point dicts (schema matches
            ``_detect_tool_interaction_errors``).  Includes both grounding
            errors (6.x) and harness analysis results (9.x) as first-class
            failure points.
        """
        if not step_actions or not screenshots_dir:
            return []

        if action_definitions is None:
            action_definitions = self.config.action_definitions or {}
        coordinate_actions = _get_coordinate_actions(action_definitions)
        if not coordinate_actions:
            return []

        # Build mapping: step_number → previous action's post_screenshot_path.
        # This is what the agent actually saw before the harness bug fix.
        prev_post_screenshot_map: Dict[int, str] = {}
        for i, sa in enumerate(step_actions):
            if i > 0:
                prev_post_screenshot_map[sa["step_number"]] = step_actions[i - 1].get(
                    "post_screenshot_path", ""
                )

        # Filter to coordinate-bearing actions with valid x, y.
        coord_steps = []
        for sa in step_actions:
            if sa["action_name"] not in coordinate_actions:
                continue
            args = sa.get("action_args", {})
            x, y = args.get("x"), args.get("y")
            if x is None or y is None:
                continue
            try:
                x, y = int(x), int(y)
            except (TypeError, ValueError):
                continue
            coord_steps.append((sa, x, y))

        if not coord_steps:
            return []

        async def _check_one(
            sa: Dict[str, Any], x: int, y: int
        ) -> Tuple[Optional[Dict], Optional[Dict]]:
            step = sa["step_number"]
            screenshot_file = sa.get("screenshot_path", "")
            if not screenshot_file:
                return None, None

            screenshot_path = Path(screenshots_dir) / screenshot_file
            if not screenshot_path.exists():
                logger.warning(
                    "Grounding check: screenshot not found for step %d: %s",
                    step,
                    screenshot_path,
                )
                return None, None

            try:
                img = Image.open(screenshot_path).convert("RGB").copy()

                # Skip if coordinates are outside the image bounds.
                w, h = img.size
                if x < 0 or x >= w or y < 0 or y >= h:
                    logger.warning(
                        "Grounding check: coords (%d, %d) out of bounds "
                        "for %dx%d image at step %d",
                        x,
                        y,
                        w,
                        h,
                        step,
                    )
                    return None, None

                wide_view, zoom_marked, zoom_unmarked = self._create_grounding_images(
                    img, x, y
                )
                wide_b64 = encode_image_b64(wide_view, self.config.grounding_image_quality)

                intent = sa.get("reasoning", "") or "N/A"
                action_type = sa["action_name"]

                # -- Crop decision: ask LLM whether zoom images help --
                # Uses the raw (unannotated) screenshot and only the
                # agent intent — no circles or action type.
                should_crop = True  # default to cropping
                try:
                    crop_prompt = GROUNDING_CROP_DECISION_PROMPT.format(
                        intent=intent,
                    )
                    raw_b64 = encode_image_b64(img, self.config.grounding_image_quality)
                    crop_messages = self.DEFAULT_SYSTEM_MESSAGES + [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": crop_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{raw_b64}"
                                    },
                                },
                            ],
                        }
                    ]
                    crop_response = await call_llm(
                        crop_messages, self._gpt5_client, json_output=True
                    )
                    crop_result = json.loads(crop_response)
                    should_crop = bool(crop_result.get("should_crop", True))
                except Exception as e:
                    logger.debug(
                        "Grounding check: crop decision failed for step %d, "
                        "defaulting to crop: %s",
                        step,
                        e,
                    )

                # Load post-action screenshot if available.
                post_screenshot_file = sa.get("post_screenshot_path", "")
                has_post_image = False
                post_b64 = None
                if post_screenshot_file:
                    post_path = Path(screenshots_dir) / post_screenshot_file
                    if post_path.exists():
                        post_img = Image.open(post_path).convert("RGB").copy()
                        post_b64 = encode_image_b64(post_img, self.config.grounding_image_quality)
                        has_post_image = True
                    else:
                        logger.debug(
                            "Grounding check: post-screenshot not found for "
                            "step %d: %s",
                            step,
                            post_path,
                        )

                # Build the main grounding prompt with the appropriate
                # preamble and accuracy instructions based on whether
                # zoom crops are included.
                if should_crop:
                    image_preamble = _GROUNDING_PREAMBLE_WITH_ZOOM
                    accuracy_instructions = _GROUNDING_ACCURACY_WITH_ZOOM
                    post_image_number = 4  # wide + 2 zooms + post
                else:
                    image_preamble = _GROUNDING_PREAMBLE_WIDE_ONLY
                    accuracy_instructions = _GROUNDING_ACCURACY_WIDE_ONLY
                    post_image_number = 2  # wide + post

                if has_post_image:
                    post_image_description = _GROUNDING_POST_IMAGE_DESCRIPTION.format(
                        post_image_number=post_image_number
                    )
                else:
                    post_image_description = _GROUNDING_NO_POST_IMAGE_DESCRIPTION

                prompt_text = FINE_GRAINED_GROUNDING_PROMPT.format(
                    image_preamble=image_preamble,
                    accuracy_instructions=accuracy_instructions,
                    intent=intent,
                    action_type=action_type,
                    post_image_description=post_image_description,
                )

                content_parts = [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{wide_b64}"},
                    },
                ]
                if should_crop:
                    zoom_b64 = encode_image_b64(zoom_marked, self.config.grounding_image_quality)
                    zoom_clean_b64 = encode_image_b64(zoom_unmarked, self.config.grounding_image_quality)
                    content_parts.extend(
                        [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{zoom_b64}"
                                },
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{zoom_clean_b64}"
                                },
                            },
                        ]
                    )
                if has_post_image:
                    content_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{post_b64}"},
                        }
                    )

                messages = self.DEFAULT_SYSTEM_MESSAGES + [
                    {"role": "user", "content": content_parts}
                ]

                response_text = await call_llm(
                    messages, self._gpt5_client, json_output=True
                )
                result = json.loads(response_text)

                # Parse the errors list from the LLM response.
                raw_errors = result.get("errors", [])
                if not isinstance(raw_errors, list):
                    raw_errors = []
                reasoning = result.get("Reasoning", "")

                # Build a dict keyed by error_code for easy lookup.
                errors_by_code: Dict[str, Dict] = {}
                for err in raw_errors:
                    code = str(err.get("error_code", "")).strip()
                    if code in {c for c in _GROUNDING_ERROR_CODES}:
                        errors_by_code[code] = err

                # Resolution precedence among grounding-judge codes:
                #   6.5 (target absent) > 6.4 (spatial miss) > 3.5
                #   (correct grounding but no effect → incomplete
                #   sub-goal). The judge prompt instructs that 3.5 is
                #   mutually exclusive with 6.4 (coords-correct + no
                #   effect routes to 3.5), so in practice 3.5 only
                #   surfaces when neither 6.4 nor 6.5 applies. If both
                #   3.5 and 6.5 are returned (rare), 6.5 wins as the
                #   primary FP because it points at a deeper grounding
                #   failure. Harness recheck still only fires on 6.4.
                has_64 = "6.4" in errors_by_code
                has_65 = "6.5" in errors_by_code
                has_35 = "3.5" in errors_by_code

                if has_65:
                    primary_code = "6.5"
                    primary_err = errors_by_code["6.5"]
                elif has_64:
                    primary_code = "6.4"
                    primary_err = errors_by_code["6.4"]
                elif has_35:
                    primary_code = "3.5"
                    primary_err = errors_by_code["3.5"]
                else:
                    primary_code = None
                    primary_err = None

                what_happened = (
                    primary_err.get("what_happened", "") if primary_err else ""
                )

                # -- Harness recheck for 6.4 errors --
                # Triggers whenever 6.4 is among the detected errors
                # (even if 6.5 is the primary code).
                harness_result = None
                if has_64:
                    prev_post_file = prev_post_screenshot_map.get(step, "")
                    if prev_post_file:
                        prev_post_path = Path(screenshots_dir) / prev_post_file
                        if prev_post_path.exists():
                            try:
                                prev_img = (
                                    Image.open(prev_post_path).convert("RGB").copy()
                                )
                                pw, ph = prev_img.size
                                if 0 <= x < pw and 0 <= y < ph:
                                    # Fast path: compare the zoom-in crop
                                    # around (x, y) from both the current
                                    # pre-action screenshot and the previous
                                    # post-action screenshot.  If the local
                                    # region is perceptually identical there
                                    # is no harness timing discrepancy —
                                    # skip the second LLM call entirely.
                                    crop_a = self._crop_around(img, x, y)
                                    crop_b = self._crop_around(prev_img, x, y)
                                    crops_match, hash_dist = self._images_are_identical(
                                        crop_a,
                                        crop_b,
                                        threshold=self._HARNESS_HASH_THRESHOLD,
                                        hash_size=self._HARNESS_HASH_SIZE,
                                    )
                                    if crops_match:
                                        # Images are essentially the same —
                                        # no harness timing discrepancy to
                                        # investigate.  Skip the second LLM
                                        # call entirely; no harness error.
                                        harness_result = None
                                        logger.info(
                                            "Harness recheck step %d: "
                                            "image hash match — skipping "
                                            "second LLM call "
                                            "(distance=%d, threshold=%d)",
                                            step,
                                            hash_dist,
                                            self._HARNESS_HASH_THRESHOLD,
                                        )
                                    else:
                                        # Crops differ — full LLM recheck.
                                        # Always include zoom crops from
                                        # the previous post-image regardless
                                        # of the first call's crop decision.
                                        (
                                            prev_wide,
                                            prev_zoom_marked,
                                            prev_zoom_unmarked,
                                        ) = self._create_grounding_images(
                                            prev_img, x, y
                                        )
                                        prev_wide_b64 = encode_image_b64(prev_wide, self.config.grounding_image_quality)

                                        # Build a fresh prompt for the
                                        # recheck — always with zoom
                                        # preamble since crops are always
                                        # included here.
                                        recheck_post_num = 4  # wide + 2 zooms + post
                                        if has_post_image:
                                            recheck_post_desc = _GROUNDING_POST_IMAGE_DESCRIPTION.format(
                                                post_image_number=recheck_post_num
                                            )
                                        else:
                                            recheck_post_desc = (
                                                _GROUNDING_NO_POST_IMAGE_DESCRIPTION
                                            )

                                        recheck_prompt_text = FINE_GRAINED_GROUNDING_PROMPT.format(
                                            image_preamble=_GROUNDING_PREAMBLE_WITH_ZOOM,
                                            accuracy_instructions=_GROUNDING_ACCURACY_WITH_ZOOM,
                                            intent=intent,
                                            action_type=action_type,
                                            post_image_description=recheck_post_desc,
                                        )

                                        prev_content_parts: list = [
                                            {
                                                "type": "text",
                                                "text": recheck_prompt_text,
                                            },
                                            {
                                                "type": "image_url",
                                                "image_url": {
                                                    "url": f"data:image/jpeg;base64,{prev_wide_b64}"
                                                },
                                            },
                                            {
                                                "type": "image_url",
                                                "image_url": {
                                                    "url": f"data:image/jpeg;base64,{encode_image_b64(prev_zoom_marked, self.config.grounding_image_quality)}"
                                                },
                                            },
                                            {
                                                "type": "image_url",
                                                "image_url": {
                                                    "url": f"data:image/jpeg;base64,{encode_image_b64(prev_zoom_unmarked, self.config.grounding_image_quality)}"
                                                },
                                            },
                                        ]
                                        if has_post_image:
                                            prev_content_parts.append(
                                                {
                                                    "type": "image_url",
                                                    "image_url": {
                                                        "url": f"data:image/jpeg;base64,{post_b64}"
                                                    },
                                                }
                                            )

                                        prev_messages = self.DEFAULT_SYSTEM_MESSAGES + [
                                            {
                                                "role": "user",
                                                "content": prev_content_parts,
                                            }
                                        ]
                                        p2_response = await call_llm(
                                            prev_messages,
                                            self._gpt5_client,
                                            json_output=True,
                                        )
                                        p2_result = json.loads(p2_response)
                                        p2_raw_errors = p2_result.get("errors", [])
                                        if not isinstance(p2_raw_errors, list):
                                            p2_raw_errors = []
                                        p2_error_codes = {
                                            str(e.get("error_code", "")).strip()
                                            for e in p2_raw_errors
                                        }
                                        p2_reasoning = p2_result.get("Reasoning", "")

                                        p2_has_64 = "6.4" in p2_error_codes
                                        p2_codes_str = (
                                            ", ".join(sorted(p2_error_codes)) or "none"
                                        )

                                        # Recheck classification:
                                        #   6.4 present (with or without 6.5) → 9.1 (harness + grounding error)
                                        #   6.4 absent  (only 6.5 or neither)  → 9.2 (harness only)
                                        if p2_has_64:
                                            _info = get_harness_code_info("9.1")
                                            harness_result = {
                                                "step_number": step,
                                                "harness_code": "9.1",
                                                "harness_label": _info["harness_label"],
                                                "harness_description": _info[
                                                    "description"
                                                ],
                                                "initial_error_code": "6.4",
                                                "initial_reasoning": reasoning,
                                                "recheck_error_code": p2_codes_str,
                                                "recheck_reasoning": (
                                                    f"LLM recheck with the "
                                                    f"previous post-action "
                                                    f"screenshot still "
                                                    f"classifies this as 6.4 "
                                                    f"(got {p2_codes_str}). "
                                                    f"The grounding error is "
                                                    f"genuine — it persists on "
                                                    f"the previous screenshot. "
                                                    f"Recheck reasoning: "
                                                    f"{p2_reasoning}"
                                                ),
                                            }
                                        else:
                                            _info = get_harness_code_info("9.2")
                                            harness_result = {
                                                "step_number": step,
                                                "harness_code": "9.2",
                                                "harness_label": _info["harness_label"],
                                                "harness_description": _info[
                                                    "description"
                                                ],
                                                "initial_error_code": "6.4",
                                                "initial_reasoning": reasoning,
                                                "recheck_error_code": p2_codes_str,
                                                "recheck_reasoning": (
                                                    f"LLM recheck with the "
                                                    f"previous post-action "
                                                    f"screenshot no longer "
                                                    f"classifies this as 6.4 "
                                                    f"(got {p2_codes_str}). "
                                                    f"The grounding error was "
                                                    f"a harness artifact — the "
                                                    f"agent saw a different "
                                                    f"image at decision time. "
                                                    f"Recheck reasoning: "
                                                    f"{p2_reasoning}"
                                                ),
                                            }
                                        logger.info(
                                            "Harness recheck step %d: %s (%s)",
                                            step,
                                            harness_result["harness_code"],
                                            harness_result["harness_label"],
                                        )
                            except Exception as e:
                                logger.debug(
                                    "Harness recheck failed for step %d: %s",
                                    step,
                                    e,
                                )

                if primary_code and primary_code in self._GROUNDING_ERROR_INFO:
                    info = self._GROUNDING_ERROR_INFO[primary_code]
                    grounding_fp = {
                        "step_numbers": str(step),
                        "error_code": primary_code,
                        "error_category": info["error_category"],
                        "error_type": info["error_type"],
                        "what_happened": what_happened
                        or (
                            f"The agent's `{action_type}` at coordinates "
                            f"({x}, {y}). Agent intent: {intent}"
                        ),
                        "agent_reasoning": intent,
                        "evidence": reasoning,
                        "impact": info["impact"],
                        "programmatic": True,
                    }

                    # Convert harness result into a proper failure-point dict.
                    harness_fp = None
                    if harness_result is not None:
                        h_code = harness_result["harness_code"]
                        h_info = self._GROUNDING_ERROR_INFO[h_code]
                        harness_fp = {
                            "step_numbers": str(step),
                            "error_code": h_code,
                            "error_category": h_info["error_category"],
                            "error_type": h_info["error_type"],
                            "what_happened": harness_result.get(
                                "recheck_reasoning", ""
                            ),
                            "agent_reasoning": intent,
                            "evidence": harness_result.get(
                                "initial_reasoning", reasoning
                            ),
                            "impact": h_info["impact"],
                            "programmatic": True,
                            "harness_metadata": harness_result,
                        }
                    return grounding_fp, harness_fp
            except Exception as e:
                logger.warning(
                    "Grounding check: failed for step %d: %s",
                    step,
                    e,
                )
            return None, None

        raw_results = await asyncio.gather(
            *[_check_one(sa, x, y) for sa, x, y in coord_steps]
        )
        grounding_results = [r[0] for r in raw_results if r[0] is not None]
        harness_fps = [r[1] for r in raw_results if r[1] is not None]

        # If any 6.5 (target-absent) was found, drop 6.4 entries that
        # lack a harness recheck companion — 6.5 is the more specific
        # grounding-failure mode and subsumes generic 6.4 misses across
        # the trajectory. 3.5 is a different failure mode (correct
        # grounding, no effect) and does NOT suppress 6.4. 6.4 entries
        # that triggered a harness recheck (9.1/9.2) are kept so both
        # the grounding error and the harness classification are visible.
        has_65 = any(r["error_code"] == "6.5" for r in grounding_results)
        if has_65:
            harness_steps = {r["step_numbers"] for r in harness_fps}
            grounding_results = [
                r
                for r in grounding_results
                if r["error_code"] != "6.4" or r["step_numbers"] in harness_steps
            ]

        # Merge harness failure points into the main results list.
        all_fps = grounding_results + harness_fps
        return all_fps

    # ------------------------------------------------------------------
    # Step 9b: Post-execution Task Verification (trajectory-informed)
    # ------------------------------------------------------------------
    async def _classify_task_with_trajectory(
        self,
        rubric: dict,
        evidence_by_criterion: Dict[int, List[Dict]],
        task: str,
        init_url_context: str,
        action_history: str,
        predicted_output: str,
        outcome_result: dict,
        total_screenshots: int = 0,
        apps: str = "N/A",
    ) -> dict:
        """Step 9b: Trajectory-informed task verification.

        Uses the same ambiguity / validity axes as Step 10
        (``CHECK_VALID_TASK_PROMPT``), but enriched with the full trajectory
        context (action history, scored rubric, screenshot evidence, and
        outcome verification).  This allows the LLM to use execution evidence
        to make a more informed judgment about whether the *task itself* was
        ambiguous or invalid.

        Uses 1 o4-mini call (with up to 5 retry attempts on validation errors).

        Returns:
            Dict matching the ``TaskAgentResult`` schema, including
            ``is_ambiguous``, ``is_invalid``, etc.
        """
        rubric_summary = build_scored_rubric_summary(rubric)
        evidence_summary = build_all_screenshot_evidence_text(
            rubric, evidence_by_criterion, total_screenshots
        )

        outcome_success = outcome_result.get("output_success")
        if outcome_success is True:
            outcome_label = "SUCCESS"
        elif outcome_success is False:
            outcome_label = "FAILURE"
        else:
            outcome_label = "UNKNOWN"
        outcome_text = (
            f"Task outcome: {outcome_label}\n"
            f"Primary intent: {outcome_result.get('primary_intent', 'N/A')}\n"
            f"Reasoning: {outcome_result.get('reasoning', 'N/A')}"
        )

        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        prompt = Template(CHECK_VALID_TASK_WITH_TRAJECTORY_PROMPT).substitute(
            task_definition=task,
            init_url_context=init_url_context,
            apps=apps,
            date=date,
            action_history=action_history,
            predicted_output=predicted_output or "N/A",
            rubric_summary=rubric_summary,
            evidence_summary=evidence_summary,
            outcome_verification=outcome_text,
        )
        messages = self.DEFAULT_SYSTEM_MESSAGES + [{"role": "user", "content": prompt}]

        try:
            result = await llm_call_expect_json(
                self._o4mini_client,
                messages,
                required_keys=list(_REQUIRED_FIELDS.keys()),
                max_retries=self.config.max_iters,
                gate_name="step9b_classify_task_with_trajectory",
                json_output=True,
            )
            _validate_verification_result(result)
        except (RuntimeError, ValueError) as e:
            logger.warning(
                "Failed trajectory-informed task verification after %d attempts: %s",
                self.config.max_iters,
                e,
            )
            return {
                "reasoning_is_ambiguous": (
                    f"Failed after {self.config.max_iters} attempts. Last error: {e}"
                ),
                "is_ambiguous": None,
                "ambiguity_codes": [],
                "reasoning_is_invalid": (
                    f"Failed after {self.config.max_iters} attempts. Last error: {e}"
                ),
                "is_invalid": None,
                "invalid_task_codes": [],
            }

        logger.info(
            "Step 9b task verification result: is_ambiguous=%s, is_invalid=%s",
            result["is_ambiguous"],
            result["is_invalid"],
        )
        return result

    # ------------------------------------------------------------------
    # Step 10: Unified Task Verification (CHECK_VALID_TASK_PROMPT)
    # ------------------------------------------------------------------
    async def _classify_task(
        self,
        task: str,
        url: str,
        apps: list[str] | None = None,
    ) -> dict:
        """Step 10: Delegates to :func:`task_classification.classify_task`.

        Returns the ``TaskAgentResult`` as a plain dict so it can be stored
        directly in the rubric JSON.
        """
        result = await classify_task(
            task,
            url,
            self._o4mini_client,
            apps=apps,
            system_messages=self.DEFAULT_SYSTEM_MESSAGES,
        )
        return result.model_dump()

    # ------------------------------------------------------------------
    # Grounding-check image helpers (used for 6.4/6.5 detection)
    # ------------------------------------------------------------------

    @staticmethod
    def _crop_around(
        image: Image.Image, x: int, y: int, crop_half: int = 250
    ) -> Image.Image:
        """Extract a crop centred on (*x*, *y*), clamped to image bounds."""
        w, h = image.size
        left = max(0, x - crop_half)
        upper = max(0, y - crop_half)
        right = min(w, x + crop_half)
        lower = min(h, y + crop_half)
        return image.crop((left, upper, right, lower))

    @staticmethod
    def _images_are_identical(
        img_a: Image.Image,
        img_b: Image.Image,
        *,
        threshold: int = 0,
        hash_size: int = 8,
    ) -> Tuple[bool, int]:
        """Check whether two images are perceptually identical via dhash.

        Returns ``(is_match, hamming_distance)``.  *is_match* is ``True``
        when the Hamming distance is ``<= threshold`` **and** the images
        have the same dimensions.
        """
        if img_a.size != img_b.size:
            return False, -1
        h_a = imagehash.dhash(img_a, hash_size=hash_size)
        h_b = imagehash.dhash(img_b, hash_size=hash_size)
        distance = h_a - h_b
        return distance <= threshold, distance

    @staticmethod
    def _draw_concentric_circles(
        image: Image.Image,
        x: int,
        y: int,
        inner_radius: int,
        outer_radius: int,
        inner_color: str = "lime",
        outer_color: str = "red",
    ) -> Image.Image:
        """Draw concentric circles on a copy of *image* at (*x*, *y*).

        Returns a new image; the original is not modified.
        """
        img = image.copy()
        draw = ImageDraw.Draw(img)
        # Outer circle (red)
        draw.ellipse(
            [x - outer_radius, y - outer_radius, x + outer_radius, y + outer_radius],
            outline=outer_color,
            width=3,
        )
        # Inner circle (lime green)
        draw.ellipse(
            [x - inner_radius, y - inner_radius, x + inner_radius, y + inner_radius],
            outline=inner_color,
            width=1,
        )
        return img

    @staticmethod
    def _create_grounding_images(
        screenshot: Image.Image, x: int, y: int
    ) -> Tuple[Image.Image, Image.Image, Image.Image]:
        """Create annotated wide-view and zoom-in images for grounding checks.

        Returns:
            (wide_view, zoom_marked, zoom_unmarked) —
            wide_view has inner_r=2, outer_r=15.
            zoom_marked is a 500×500 crop centred on (x, y) with inner_r=2, outer_r=15.
            zoom_unmarked is the same crop without any annotations.
        """
        wide_view = VerifierAgent._draw_concentric_circles(
            screenshot, x, y, inner_radius=2, outer_radius=15
        )

        # Compute crop box clamped to image bounds.
        w, h = screenshot.size
        crop_half = 250
        left = max(0, x - crop_half)
        upper = max(0, y - crop_half)
        right = min(w, x + crop_half)
        lower = min(h, y + crop_half)
        cropped = screenshot.crop((left, upper, right, lower))

        # Unmarked zoom-in (clean crop without annotations).
        zoom_unmarked = cropped.copy()

        # Coordinates relative to the crop for circle drawing.
        cx = x - left
        cy = y - upper
        zoom_marked = VerifierAgent._draw_concentric_circles(
            cropped, cx, cy, inner_radius=2, outer_radius=15
        )
        return wide_view, zoom_marked, zoom_unmarked

