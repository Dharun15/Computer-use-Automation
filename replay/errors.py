"""
Replay result contract -- Phase 7 taxonomy.

Every replay ends in exactly one of four states:

  SUCCESS             -- outputs returned, checkpoint verified.
  BUSINESS_OUTCOME    -- the workflow completed and reached a real,
                         expected non-happy-path result the caller needs to
                         know about (no such member, invalid input format).
                         This is NOT a crash -- `outcome_code` says which
                         known outcome it was, and `outputs` may still be
                         partially populated.
  HUMAN_INTERVENTION  -- automation cannot safely continue on its own (an
                         unrecognized/unresolved dialog blocking progress)
                         and has paused, handing off to a human operator on
                         the SAME session (see handoff/manager.py). This is
                         a paused, resumable state, not a terminal failure.
  HARD_FAILURE        -- something the workflow doesn't know how to
                         recover from and a human wasn't explicitly
                         summoned for (permission denied, application
                         error, an unresolvable target, an exhausted retry
                         budget). `failed_step_id`/`expected`/`observed`
                         give a debuggable trail.

`recovery_attempts` records every RECOVERABLE_ERROR retry that was tried
along the way (e.g. a transient timeout), whether or not the run ultimately
succeeded -- this is what lets an evidence reviewer see "yes, replay did
retry twice before continuing" rather than a run just silently taking
longer than expected.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ReplayStatus(str, Enum):
    SUCCESS = "success"
    BUSINESS_OUTCOME = "business_outcome"
    HUMAN_INTERVENTION = "human_intervention"
    HARD_FAILURE = "hard_failure"


class OutcomeCode(str, Enum):
    """Known, named outcomes -- extend this as new business/hard-failure
    patterns are recognized. An unrecognized failure gets `UNKNOWN`, never
    silently reclassified as something more specific than we actually know."""

    MEMBER_NOT_FOUND = "MEMBER_NOT_FOUND"
    INVALID_INPUT = "INVALID_INPUT"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    APPLICATION_ERROR = "APPLICATION_ERROR"
    UNRESOLVED_DIALOG = "UNRESOLVED_DIALOG"
    RETRY_BUDGET_EXHAUSTED = "RETRY_BUDGET_EXHAUSTED"
    CHECKPOINT_NOT_SATISFIED = "CHECKPOINT_NOT_SATISFIED"
    UNKNOWN = "UNKNOWN"


class RecoveryAttempt(BaseModel):
    step_id: str
    attempt: int
    reason: str
    outcome: str  # "retried_ok" | "retried_failed" | "exhausted"


class ReplayResult(BaseModel):
    status: ReplayStatus
    outcome_code: Optional[OutcomeCode] = None
    outputs: dict[str, Any] = Field(default_factory=dict)
    steps_completed: int = 0
    total_steps: int = 0
    failed_step_id: Optional[str] = None
    failed_step_action: Optional[str] = None
    message: str = ""
    expected: Optional[str] = None
    observed: Optional[str] = None
    recovery_attempts: list[RecoveryAttempt] = Field(default_factory=list)
    screenshot_path: Optional[str] = None
    intervention: Optional[dict] = None  # InterventionRequest.model_dump(), if HUMAN_INTERVENTION

    @property
    def ok(self) -> bool:
        return self.status == ReplayStatus.SUCCESS
