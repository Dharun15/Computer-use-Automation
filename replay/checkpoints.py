"""
Checkpoint verification.

Never assume a step "worked" just because the surface reported OK -- a
click can succeed as a browser action while landing somewhere other than
where the workflow expects (a stale session redirecting to a login page,
for instance). The checkpoint is the one explicit assertion that replay
actually reached the state the artifact was recorded to reach.
"""
from __future__ import annotations

from dataclasses import dataclass

from artifacts.schema import ArtifactCheckpoint
from surface.observation import Observation


@dataclass
class CheckpointResult:
    passed: bool
    expected: str
    observed: str


def verify_checkpoint(checkpoint: ArtifactCheckpoint, observation: Observation) -> CheckpointResult:
    if checkpoint.type == "text_present":
        expected = f"text present: {checkpoint.value!r}"
        haystack = f"{observation.title}\n{observation.text_summary}"
        passed = checkpoint.value in haystack
        observed = f"title={observation.title!r}"
        return CheckpointResult(passed=passed, expected=expected, observed=observed)

    if checkpoint.type == "url_contains":
        expected = f"url contains: {checkpoint.value!r}"
        passed = checkpoint.value in observation.url
        observed = f"url={observation.url!r}"
        return CheckpointResult(passed=passed, expected=expected, observed=observed)

    if checkpoint.type == "status_code":
        expected = f"status_code == {checkpoint.value}"
        passed = str(observation.status_code) == str(checkpoint.value)
        observed = f"status_code={observation.status_code}"
        return CheckpointResult(passed=passed, expected=expected, observed=observed)

    raise ValueError(f"Unknown checkpoint type: {checkpoint.type!r}")
