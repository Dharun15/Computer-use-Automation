"""
Multi-run stability testing (stretch goal).

Replays the same artifact with the same inputs N times against a live
surface and reports a simple stability/flakiness signal: how many runs
landed on each status, and basic timing stats.

Honest limitation: the fake bank app is entirely deterministic (a given
member ID always produces the same page, every time), so running this
against it will always show 100% consistency -- there is no real flakiness
to detect in this target application, and manufacturing artificial
flakiness would misrepresent what the tool actually measures. This tool
is exactly what a reviewer would run against a REAL legacy app (which does
have transient network/load variance) to build confidence before promoting
an artifact to unattended production use -- see REPORT.md's Cuts section
for where a confidence/approval gate built on top of this would plug in.
"""
from __future__ import annotations

import statistics
import time
from typing import Any, Protocol

from pydantic import BaseModel, Field

from artifacts.schema import Artifact
from replay.errors import ReplayResult, ReplayStatus


class Replayer(Protocol):
    """Whatever `run_stability_check` needs from a replay engine -- a
    Protocol (not a concrete import of ReplayEngine) so this stays trivial
    to test with a scripted stand-in, the same pattern used throughout
    this project (ComputerSurface, ModelClient)."""

    def replay(self, artifact: Artifact, inputs: dict[str, Any]) -> ReplayResult:
        ...


class StabilityRunResult(BaseModel):
    run_index: int
    status: str
    duration_seconds: float


class StabilityReport(BaseModel):
    artifact_id: str
    total_runs: int
    successes: int
    failures: int
    success_rate: float
    status_counts: dict[str, int] = Field(default_factory=dict)
    mean_duration_seconds: float
    min_duration_seconds: float
    max_duration_seconds: float
    runs: list[StabilityRunResult] = Field(default_factory=list)


def run_stability_check(
    engine: Replayer, artifact: Artifact, inputs: dict[str, Any], n: int = 5
) -> StabilityReport:
    runs: list[StabilityRunResult] = []
    status_counts: dict[str, int] = {}

    for i in range(n):
        start = time.time()
        result = engine.replay(artifact, inputs)
        duration = time.time() - start
        runs.append(
            StabilityRunResult(run_index=i + 1, status=result.status.value, duration_seconds=duration)
        )
        status_counts[result.status.value] = status_counts.get(result.status.value, 0) + 1

    successes = status_counts.get(ReplayStatus.SUCCESS.value, 0)
    durations = [r.duration_seconds for r in runs]

    return StabilityReport(
        artifact_id=artifact.artifact_id,
        total_runs=n,
        successes=successes,
        failures=n - successes,
        success_rate=(successes / n) if n else 0.0,
        status_counts=status_counts,
        mean_duration_seconds=statistics.mean(durations) if durations else 0.0,
        min_duration_seconds=min(durations) if durations else 0.0,
        max_duration_seconds=max(durations) if durations else 0.0,
        runs=runs,
    )