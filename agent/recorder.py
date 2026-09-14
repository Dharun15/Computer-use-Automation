"""
Minimal evidence writer for discovery runs.

This is intentionally small: Phase 10 will formalize the full evidence
store (screenshots on failure, richer run metadata, replay evidence too).
For now this just gives us something inspectable after a real LLM run --
the structured step log and the raw transcript, saved separately, matching
the same separation `DiscoveryResult` already enforces in memory.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from agent.state import DiscoveryResult


def save_discovery_evidence(
    result: DiscoveryResult, out_dir: str = "evidence/discovery"
) -> Path:
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}"
    run_dir = Path(out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "state.json").write_text(result.state.model_dump_json(indent=2))
    (run_dir / "transcript.json").write_text(json.dumps(result.transcript, indent=2))

    summary = (
        f"Goal: {result.state.goal}\n"
        f"Status: {result.state.status.value}\n"
        f"Steps executed: {result.state.step_count}\n"
        f"Stop reason: {result.state.stop_reason}\n"
        f"Outputs: {json.dumps(result.state.outputs)}\n"
        f"Final message: {result.state.final_message}\n"
    )
    (run_dir / "summary.txt").write_text(summary)

    return run_dir
