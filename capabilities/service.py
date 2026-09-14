"""
Agent-facing capability interface (stretch goal).

This is the concrete shape of "the AI agent invokes it in production" that
the rest of this project has been building toward. Every saved artifact
becomes a discoverable, invokable HTTP capability:

    GET  /capabilities              -- discover what exists, with typed
                                        inputs/outputs (what an agent's
                                        tool-use loop would call first)
    GET  /capabilities/{id}         -- full detail on one capability
                                        (every step, the checkpoint, etc.)
    POST /capabilities/{id}/invoke  -- actually run it via ReplayEngine --
                                        NO LLM anywhere in this path -- with
                                        real input values

Kept deliberately simple: one process, no queueing, one browser session
opened and closed per invocation (`headless=True`, `surface.close()` in a
`finally`). A production version would pool sessions and handle concurrent
invocations of the same capability across many callers, but that is
exactly the "scaling infrastructure" the brief says not to build
prematurely -- this shows the seam (an HTTP boundary in front of
`ReplayEngine`) without pretending to be a production dispatcher.

`base_url` is overridable per-invocation (not just a fixed constant)
specifically because that is the multi-tenant seam described in
REPORT.md Section 4: the same artifact, invoked against a different
tenant's live surface, is what "reuse across tenants running the same
app" looks like mechanically -- this endpoint is where that would plug in.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from artifacts.schema import Artifact
from artifacts.storage import DEFAULT_ARTIFACT_DIR, load_artifact
from handoff.manager import HandoffManager
from replay.engine import ReplayEngine
from replay.evidence import save_replay_evidence
from safety.policy import PolicyEngine, SafetyPolicy
from surface.playwright_surface import PlaywrightSurface
from tools.executor import ToolExecutor

DEFAULT_BASE_URL = "http://localhost:8000"

app = FastAPI(
    title="Capability Catalog",
    description="Agent-facing interface over saved, replayable artifacts.",
)


class CapabilitySummary(BaseModel):
    artifact_id: str
    name: str
    description: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]


class InvokeRequest(BaseModel):
    inputs: dict[str, Any] = {}
    base_url: Optional[str] = None


class InvokeResponse(BaseModel):
    status: str
    outcome_code: Optional[str] = None
    outputs: dict[str, Any] = {}
    message: str = ""
    steps_completed: int
    total_steps: int
    failed_step_id: Optional[str] = None
    expected: Optional[str] = None
    observed: Optional[str] = None


def _list_artifact_paths() -> list[Path]:
    directory = Path(DEFAULT_ARTIFACT_DIR)
    if not directory.exists():
        return []
    return sorted(directory.glob("*.json"))


def _load_all_artifacts() -> list[Artifact]:
    artifacts = []
    for path in _list_artifact_paths():
        try:
            artifacts.append(load_artifact(path))
        except Exception:
            # A corrupted/hand-edited file shouldn't take down the whole
            # catalog -- it just doesn't show up as an invokable capability.
            continue
    return artifacts


def _find_artifact(artifact_id: str) -> Artifact:
    for artifact in _load_all_artifacts():
        if artifact.artifact_id == artifact_id:
            return artifact
    raise HTTPException(status_code=404, detail=f"No capability named '{artifact_id}'.")


@app.get("/capabilities", response_model=list[CapabilitySummary])
def list_capabilities():
    """What an AI agent calls first: what capabilities exist, and what do
    they need/return? This is the discovery half of "discover and invoke
    by name with typed args."""
    return [
        CapabilitySummary(
            artifact_id=a.artifact_id,
            name=a.name,
            description=a.description,
            inputs={k: v.model_dump() for k, v in a.inputs.items()},
            outputs={k: v.model_dump() for k, v in a.outputs.items()},
        )
        for a in _load_all_artifacts()
    ]


@app.get("/capabilities/{artifact_id}")
def get_capability(artifact_id: str):
    return _find_artifact(artifact_id).model_dump()


@app.post("/capabilities/{artifact_id}/invoke", response_model=InvokeResponse)
def invoke_capability(artifact_id: str, request: InvokeRequest):
    """Actually run the capability -- deterministic replay, no LLM. This
    is the exact call an AI agent's tool-use loop would make in production."""
    artifact = _find_artifact(artifact_id)
    base_url = request.base_url or DEFAULT_BASE_URL
    hostname = urlsplit(base_url).hostname or "localhost"

    surface = PlaywrightSurface(base_url=base_url, headless=True)
    try:
        policy = PolicyEngine(SafetyPolicy(allowed_domains=[hostname]))
        executor = ToolExecutor(surface, policy=policy)
        engine = ReplayEngine(executor, handoff=HandoffManager())
        result = engine.replay(artifact, request.inputs)
        save_replay_evidence(artifact, request.inputs, result)
    finally:
        surface.close()

    return InvokeResponse(
        status=result.status.value,
        outcome_code=result.outcome_code.value if result.outcome_code else None,
        outputs=result.outputs,
        message=result.message,
        steps_completed=result.steps_completed,
        total_steps=result.total_steps,
        failed_step_id=result.failed_step_id,
        expected=result.expected,
        observed=result.observed,
    )