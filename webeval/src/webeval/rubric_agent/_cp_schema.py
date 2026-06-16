"""Critical-point taxonomy schema (inline-ported from agento_next).

The original lives at
``agento_next/agents/critical_point_compliance/schema.py``. Only the
two types consumed by ``critical_point_classifier.py`` are kept here:

- :class:`Confidence` — judge confidence enum with descriptions.
- :class:`CriticalPointTypesConfig` — YAML-loaded type taxonomy.

The form-field / compliance dataclasses (``ClassificationResult``,
``ComplianceResult``, etc.) and the ``ImageObj``-using
``ComplianceInput`` are intentionally excluded — they pull
``aztool.clients.messages`` and would re-introduce the external
dependency this fork is designed to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Confidence(str, Enum):
    """Judge confidence levels, with human-readable descriptions.

    Each member carries a short value (``"HIGH"``, etc.) and a longer
    ``description`` suitable for inclusion in LLM prompts.
    """

    HIGH = (
        "HIGH",
        (
            "The screenshots clearly show what the agent did, and the "
            "compliance verdict is unambiguous."
        ),
    )
    MEDIUM = (
        "MEDIUM",
        (
            "Some agent actions are unclear from the screenshots "
            "(e.g., fields are partially visible, scrolled off-screen), "
            "or the expected behavior has some room for interpretation."
        ),
    )
    LOW = (
        "LOW",
        (
            "The screenshots are unclear, the agent's actions are hard "
            "to trace, or the expected behavior is ambiguous enough that "
            "the verdict could go either way."
        ),
    )

    def __new__(cls, value: str, description: str = "") -> "Confidence":
        obj = str.__new__(cls, value)
        obj._value_ = value
        return obj

    def __init__(self, value: str, description: str = "") -> None:
        self._description = description

    @property
    def description(self) -> str:
        """Human-readable explanation of this confidence level."""
        return self._description


@dataclass(frozen=True)
class CriticalPointType:
    """Configuration for a single critical point type."""

    name: str
    description: str
    expected_behavior: str = ""
    examples: tuple[str, ...] = ()


@dataclass(frozen=True)
class CriticalPointTypesConfig:
    """Loaded configuration for all critical point types.

    Typically deserialized from ``critical_point_types.yaml``.
    """

    definition: str = ""
    types: dict[str, CriticalPointType] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CriticalPointTypesConfig":
        definition = data.get("definition", "")
        types_data = data.get("types", {})
        types = {
            name: CriticalPointType(
                name=name,
                description=config.get("description", ""),
                expected_behavior=config.get("expected_behavior", ""),
                examples=tuple(config.get("examples", [])),
            )
            for name, config in types_data.items()
        }
        return cls(definition=definition, types=types)

    def items(self) -> Any:
        """Iterates over types as ``(name, dict)`` pairs for template use."""
        return {
            name: {
                "description": t.description,
                "expected_behavior": t.expected_behavior,
                "examples": list(t.examples),
            }
            for name, t in self.types.items()
        }.items()

    def keys(self) -> set[str]:
        """Returns the set of type names."""
        return set(self.types.keys())
