"""
Evidence writer for replay runs -- mirrors `agent/recorder.py`'s shape for
discovery runs, so both evidence trees look the same to a reviewer.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from artifacts.schema import Artifact
from replay.errors import ReplayResult


def save_replay_evidence(
    artifact: Artifact,
    inputs: dict,
    result: ReplayResult,
    out_dir: str = "evidence/replay",
) -> Path:
    # A timestamp alone can collide when multiple replays happen within the
    # same second (e.g. a scripted demo, or a resume() right after its
    # paused replay) -- a short random suffix guarantees each run gets its
    # own directory rather than silently overwriting a previous one.
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"
    run_dir = Path(out_dir) / f"{run_id}_{artifact.artifact_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "result.json").write_text(result.model_dump_json(indent=2))
    (run_dir / "inputs.json").write_text(json.dumps(inputs, indent=2))

    summary = (
        f"Artifact: {artifact.artifact_id} (v{artifact.artifact_version})\n"
        f"Inputs: {json.dumps(inputs)}\n"
        f"Status: {result.status.value}\n"
        f"Outcome code: {result.outcome_code.value if result.outcome_code else None}\n"
        f"Steps: {result.steps_completed}/{result.total_steps}\n"
        f"Outputs: {json.dumps(result.outputs)}\n"
        f"Message: {result.message}\n"
        f"Expected: {result.expected}\n"
        f"Observed: {result.observed}\n"
        f"Recovery attempts: {len(result.recovery_attempts)}\n"
        f"Screenshot: {result.screenshot_path}\n"
    )
    (run_dir / "summary.txt").write_text(summary)

    return run_dir

def save_stability_evidence(report, out_dir: str = "evidence/replay") -> Path:
    """Same shape as save_replay_evidence, for a StabilityReport instead
    of a single ReplayResult."""
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"
    run_dir = Path(out_dir) / f"{run_id}_{report.artifact_id}_stability"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "stability_report.json").write_text(report.model_dump_json(indent=2))

    summary = (
        f"Artifact: {report.artifact_id}\n"
        f"Total runs: {report.total_runs}\n"
        f"Successes: {report.successes}  Failures: {report.failures}\n"
        f"Success rate: {report.success_rate:.0%}\n"
        f"Status counts: {json.dumps(report.status_counts)}\n"
        f"Duration (s) -- mean: {report.mean_duration_seconds:.2f}, "
        f"min: {report.min_duration_seconds:.2f}, max: {report.max_duration_seconds:.2f}\n"
    )
    (run_dir / "summary.txt").write_text(summary)

    return run_dir
