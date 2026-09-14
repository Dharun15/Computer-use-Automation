"""
Classifies *why* a step failed into one of the taxonomy buckets from
`replay/errors.py`.

Kept as its own small, pure function (not inlined in the engine) because
this is exactly the part of the system that grows over time as new
runtime conditions are recognized in production -- new patterns get added
here, without the engine's control flow ever needing to change.

Classification looks at the fresh page `Observation` taken right after the
failing action, not just the raw ActionStatus -- the same "not_found"
status means something different depending on what's actually on screen
(a business-outcome page vs. a permission-denied page vs. nothing
recognizable at all).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from replay.errors import OutcomeCode, ReplayStatus
from surface.observation import Observation

# Title/text substrings that identify a known page state. Extending this
# dict (rather than hardcoding checks inline) is the whole point: adding
# support for a new legacy screen's "not found" page is a one-line change.
_BUSINESS_OUTCOME_TITLE_PATTERNS: dict[str, OutcomeCode] = {
    "No Records": OutcomeCode.MEMBER_NOT_FOUND,
    "Input Error": OutcomeCode.INVALID_INPUT,
}

_HARD_FAILURE_STATUS_CODES: dict[int, OutcomeCode] = {
    403: OutcomeCode.PERMISSION_DENIED,
    500: OutcomeCode.APPLICATION_ERROR,
}


@dataclass
class Classification:
    status: ReplayStatus
    outcome_code: OutcomeCode
    message: str


def classify_failure(observation: Optional[Observation], raw_message: str) -> Classification:
    """`observation` is the freshest page state available after the
    failure (may be None if even observing failed)."""

    if observation is not None:
        for pattern, code in _BUSINESS_OUTCOME_TITLE_PATTERNS.items():
            if pattern in observation.title:
                return Classification(
                    status=ReplayStatus.BUSINESS_OUTCOME,
                    outcome_code=code,
                    message=f"Reached a known business-outcome page: {observation.title!r}.",
                )

        if observation.status_code in _HARD_FAILURE_STATUS_CODES:
            code = _HARD_FAILURE_STATUS_CODES[observation.status_code]
            return Classification(
                status=ReplayStatus.HARD_FAILURE,
                outcome_code=code,
                message=(
                    f"Hard failure: HTTP {observation.status_code} on "
                    f"{observation.url!r} ({observation.title!r})."
                ),
            )

        if observation.dialog is not None:
            return Classification(
                status=ReplayStatus.HUMAN_INTERVENTION,
                outcome_code=OutcomeCode.UNRESOLVED_DIALOG,
                message=(
                    f"An unrecognized dialog occurred ({observation.dialog.message!r}) "
                    f"and was auto-dismissed, but the target action still cannot "
                    f"proceed. A human should decide whether to accept this dialog."
                ),
            )

    return Classification(
        status=ReplayStatus.HARD_FAILURE,
        outcome_code=OutcomeCode.UNKNOWN,
        message=raw_message or "Step failed for an unrecognized reason.",
    )
