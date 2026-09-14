"""
A minimal, intentionally bare operator surface for the handoff mechanism.

The assignment is explicit that a full real-time co-browsing console is
out of scope, and that a mocked operator UI is fine as long as the handoff
mechanism and control-transfer model underneath it are real. This is that
mock: three routes over a shared `HandoffManager`, meant to be looked at
with curl or a browser, not a polished app.

A real operator console would add: authentication, a live view of the
actual browser session (screen-share/co-browse), and a queue of multiple
concurrent interventions across many running artifacts. None of that
changes the mechanism below -- it would sit in front of the same
`HandoffManager` API.
"""
from __future__ import annotations

from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel

from handoff.manager import HandoffManager

router = APIRouter(prefix="/intervention", tags=["handoff"])

# A single shared manager for this mock console. A real deployment would
# key these per session/run rather than using one process-wide instance.
_manager = HandoffManager()


def get_manager() -> HandoffManager:
    return _manager


class ResolveRequest(BaseModel):
    action_description: str


@router.get("/pending")
def get_pending():
    """What an operator sees when they open the console: is anything
    waiting for them, and why?"""
    if _manager.pending is None:
        return {"pending": None}
    return {"pending": _manager.pending.model_dump()}


@router.post("/resolve")
def resolve(body: ResolveRequest):
    """The operator has manually done whatever the intervention needed
    (e.g. accepted a dialog directly on the live session) and is handing
    control back to automation."""
    if _manager.pending is None:
        raise HTTPException(status_code=409, detail="No intervention is currently pending.")
    _manager.record_human_action(body.action_description)
    resolved = _manager.resume()
    return {"resolved": resolved.model_dump(), "owner": _manager.owner.value}


@router.get("/history")
def get_history():
    return {"human_actions": [a.model_dump() for a in _manager.human_actions]}


def build_operator_app() -> FastAPI:
    """Standalone mock console app -- `uvicorn handoff.routes:app`."""
    app = FastAPI(title="Mock Operator Console")
    app.include_router(router)
    return app


app = build_operator_app()
