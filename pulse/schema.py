"""Pulse Unified DDM — typed I/O schema.

One unified model invocation reads a `UnifiedInput` (state + typed questions +
outcome contract + eligible plans) and returns a `UnifiedOutput` (typed answers,
one per question, + ranked execution policy + abstention). Internally the model
encodes several sequences (state+question+choice, state+plan); "one invocation"
means one logical call, not one encoder forward pass.

No torch here, so this stays importable anywhere (planner, gateway, tests).

Honesty: the plan-ranking outputs keep validator-pass, ground-truth-correct,
correction, and escalation probabilities SEPARATE. There is no single field that
means "accepted & correct."

Status: EXPERIMENTAL BASELINE. Nothing here is trained yet.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# --------------------------------------------------------------------------- #
# Decision side — a GENERAL typed-question model (reads each question)
# --------------------------------------------------------------------------- #
class QuestionType(str, Enum):
    """Canonical validated decision types (single source of truth; experts.py imports
    this). Abstention is NOT a type here — it is a separate calibrated gate/action."""
    CHOICE = "choice"            # pick exactly one of `choices`
    BOOLEAN = "boolean"          # no / yes (a 2-choice question)
    MULTI_LABEL = "multi_label"  # pick any subset of `choices`
    ORDINAL = "ordinal"          # pick one of ORDERED `choices` (cumulative link)
    SCORE = "score"              # calibrated scalar in [0, 1]

    @classmethod
    def coerce(cls, value) -> "QuestionType":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value))
        except ValueError as e:
            raise ValueError(f"unknown decision type {value!r}; expected "
                             f"{[t.value for t in cls]}") from e


# Types that carry a candidate list (SCORE does not).
CHOICES_TYPES = {QuestionType.CHOICE, QuestionType.BOOLEAN,
                 QuestionType.MULTI_LABEL, QuestionType.ORDINAL}


@dataclass
class TypedQuestion:
    """A bounded question put to the model about a state. The model conditions
    on `text`, `qtype`, and `choices` — it is not a fixed positional head."""
    key: str
    qtype: QuestionType
    text: str = ""                                     # natural-language question
    choices: list[str] = field(default_factory=list)  # required for CHOICE
    description: str = ""

    def resolved_choices(self) -> list[str]:
        if self.qtype is QuestionType.BOOLEAN:
            return ["no", "yes"]
        return self.choices

    def __post_init__(self) -> None:
        self.qtype = QuestionType.coerce(self.qtype)
        if self.qtype in CHOICES_TYPES and self.qtype is not QuestionType.BOOLEAN \
                and len(self.choices) < 2:
            raise ValueError(f"{self.qtype.value} question {self.key!r} needs >= 2 choices")
        if self.qtype is QuestionType.SCORE and self.choices:
            raise ValueError(f"SCORE question {self.key!r} takes no choices")


@dataclass
class DecisionAnswer:
    """The model's answer to one typed question, with calibrated probability."""
    key: str
    selected: Optional[str] = None          # CHOICE / BOOLEAN label
    score: Optional[float] = None           # SCORE, in [0, 1]
    probabilities: dict[str, float] = field(default_factory=dict)  # over valid choices
    confidence: float = 0.0                 # max class prob, post-calibration
    abstained: bool = False                 # below contract min_confidence / OOD


# --------------------------------------------------------------------------- #
# Policy / plan side (makes it a router, not a classifier)
# --------------------------------------------------------------------------- #
class Tier(str, Enum):
    L0 = "L0"; L1 = "L1"; L2 = "L2"; L3 = "L3"; L4 = "L4"


@dataclass
class PlanStep:
    tier: Tier
    executor: str                                    # "pulse-decision", "claude", "rules"
    verification: list[str] = field(default_factory=list)


@dataclass
class ExecutionPlan:
    """One candidate policy: primary attempt + verification + ordered fallback."""
    plan_id: str
    steps: list[PlanStep]

    @property
    def primary(self) -> PlanStep:
        return self.steps[0]


@dataclass
class OutcomeContract:
    output_schema: str
    min_confidence: float = 0.0
    max_cost_usd: float = float("inf")
    max_latency_ms: float = float("inf")
    allowed_tiers: list[Tier] = field(default_factory=lambda: list(Tier))
    requires_human_approval: bool = False


@dataclass
class UnifiedInput:
    state: str
    questions: list[TypedQuestion]
    contract: OutcomeContract
    eligible_plans: list[ExecutionPlan]   # already filtered by hard policy upstream


@dataclass
class RankedPlan:
    """Plan-head outputs, kept SEPARATE by construction (no combined field)."""
    plan_id: str
    policy_score: float                       # ranking score (higher = preferred)
    validator_pass_prob: float                # P(declared validators pass)
    ground_truth_correct_prob: Optional[float]  # only when independently labeled
    correction_prob: float                    # P(accepted result later corrected)
    escalation_prob: float                    # P(primary needs a stronger tier)
    quality_lower_bound: float                # calibrated LCB used for the gate
    expected_cost_usd: float
    expected_latency_ms: float


@dataclass
class UnifiedOutput:
    decisions: list[DecisionAnswer]
    ranked_plans: list[RankedPlan]            # sorted best-first
    selected_plan_id: Optional[str]           # None => abstain / escalate
    abstained: bool
    model_version: str
    calibration_profile: str
    dataset_fingerprint: str = ""             # revision of data the model saw
    status: str = "experimental_not_measured"
