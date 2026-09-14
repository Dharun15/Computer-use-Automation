"""
Human-in-the-loop escalation & handoff -- the real mechanism, not a TODO.

The core idea this models: automation and a human operator share ONE live
session (the same `PlaywrightSurface`/browser page that discovery or
replay was already using). Nothing here ever opens a second browser or a
fresh session for the human -- `HandoffManager` only tracks WHO currently
owns that one session and carries the context a human needs to act on it.
The actual surface object is untouched by any of this; it is simply not
driven by automated tool calls while `owner == HUMAN`.

The sequence this supports:

    automation gets stuck / hits something it can't safely resolve
              |
              v
    request_intervention(...)   -- ownership flips to HUMAN, context recorded
              |
              v
    [a human operator looks at `pending`, physically/manually interacts
     with the SAME live session -- e.g. accepting a dialog Playwright
     already surfaced, or reading a value directly off the page]
              |
              v
    record_human_action(...)    -- what they did is logged, for evidence
              |
              v
    resume()                    -- ownership flips back to AUTOMATION;
                                    caller (ReplayEngine.resume(), or a
                                    discovery re-run) continues from
                                    exactly where it paused

`handoff/routes.py` wraps this in a minimal (intentionally bare/mock, per
the assignment's own scope note) HTTP operator surface. The mechanism and
control-transfer model here are real regardless of what UI sits on top of
them.
"""
from __future__ import annotations

import time
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class SessionOwner(str, Enum):
    AUTOMATION = "automation"
    HUMAN = "human"


class InterventionRequest(BaseModel):
    reason: str  # short machine code, e.g. an OutcomeCode value or "dead_end"
    message: str
    goal: Optional[str] = None
    artifact_id: Optional[str] = None
    step_id: Optional[str] = None
    screenshot_path: Optional[str] = None
    context: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)


class HumanAction(BaseModel):
    description: str
    timestamp: float = Field(default_factory=time.time)


class HandoffManager:
    def __init__(self):
        self._owner = SessionOwner.AUTOMATION
        self._pending: Optional[InterventionRequest] = None
        self._human_actions: list[HumanAction] = []

    @property
    def owner(self) -> SessionOwner:
        return self._owner

    @property
    def pending(self) -> Optional[InterventionRequest]:
        return self._pending

    @property
    def human_actions(self) -> list[HumanAction]:
        return list(self._human_actions)

    def is_paused(self) -> bool:
        return self._owner == SessionOwner.HUMAN

    def request_intervention(
        self,
        reason: str,
        message: str,
        goal: Optional[str] = None,
        artifact_id: Optional[str] = None,
        step_id: Optional[str] = None,
        screenshot_path: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> InterventionRequest:
        if self._pending is not None:
            raise RuntimeError(
                "An intervention is already pending -- resolve it (resume()) "
                "before requesting another."
            )
        request = InterventionRequest(
            reason=reason,
            message=message,
            goal=goal,
            artifact_id=artifact_id,
            step_id=step_id,
            screenshot_path=screenshot_path,
            context=context or {},
        )
        self._pending = request
        self._owner = SessionOwner.HUMAN
        return request

    def record_human_action(self, description: str) -> HumanAction:
        if self._owner != SessionOwner.HUMAN:
            raise RuntimeError(
                "Cannot record a human action while automation holds the session "
                "-- call request_intervention() first."
            )
        action = HumanAction(description=description)
        self._human_actions.append(action)
        return action

    def resume(self) -> InterventionRequest:
        if self._pending is None:
            raise RuntimeError("No intervention is pending to resume from.")
        resolved = self._pending
        self._pending = None
        self._owner = SessionOwner.AUTOMATION
        return resolved
