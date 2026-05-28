"""Universal Verifier (MMRubricAgent) + companion VerifierAgent.

Self-contained multimodal rubric verification pipeline used by the fara
``webeval`` package to score agent trajectories. Uses the
:class:`webeval.oai_clients.ChatCompletionClient` interface.

- :class:`MMRubricAgent` runs Steps 0–8: rubric generation → action-only
  scoring → screenshot evidence analysis → multimodal rescoring →
  outcome verification + critical-point classification + CP violation
  check.
- :class:`VerifierAgent` runs Steps 9a/9b/10: failure-point analysis,
  trajectory-informed task verification, unified task verification —
  consumes the scored rubric produced by ``MMRubricAgent``.
"""

from .mm_rubric_agent import MMRubricAgent, MMRubricAgentConfig
from .verifier_agent import VerifierAgent, VerifierAgentConfig
from .data_point import (
    Action,
    ComputerObservation,
    CriticalPointClassificationResult,
    DataPoint,
    DataPointMetadata,
    MMRubricOutcomeResult,
    MMRubricResult,
    Outcome,
    SolverLog,
    SolverStatus,
    Task,
    UserMessage,
    UserMessageType,
    VerificationResult,
)

__all__ = [
    "MMRubricAgent",
    "MMRubricAgentConfig",
    "VerifierAgent",
    "VerifierAgentConfig",
    "DataPoint",
    "DataPointMetadata",
    "Task",
    "SolverLog",
    "SolverStatus",
    "Outcome",
    "Action",
    "ComputerObservation",
    "UserMessage",
    "UserMessageType",
    "VerificationResult",
    "MMRubricResult",
    "MMRubricOutcomeResult",
    "CriticalPointClassificationResult",
]
