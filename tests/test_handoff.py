"""
Phase 9 tests:
  1. HandoffManager in isolation -- session ownership, intervention
     lifecycle, human-action logging.
  2. ReplayEngine + HandoffManager integration -- the real pause/resume
     flow, end to end, against the live dialog-gated member.
  3. The mock operator console (FastAPI) -- proves the bare HTTP surface
     actually works against the SAME underlying mechanism.
"""
import pytest
from fastapi.testclient import TestClient

from artifacts.schema import (
    Artifact,
    ArtifactCheckpoint,
    ArtifactClickStep,
    ArtifactExtractStep,
    ArtifactNavigateStep,
    ArtifactOutputSpec,
    ArtifactSurfaceInfo,
    ArtifactTypeStep,
)
from handoff.manager import HandoffManager, SessionOwner
from replay.engine import ReplayEngine
from replay.errors import OutcomeCode, ReplayStatus
from surface.observation import Target
from tools.executor import ToolExecutor


def _lookup_balance_artifact() -> Artifact:
    return Artifact(
        artifact_id="lookup_member_savings_balance",
        name="Lookup Member Savings Balance",
        surface=ArtifactSurfaceInfo(application="fake-bank"),
        inputs={"member_id": {"type": "string", "required": True}},
        outputs={"balance": ArtifactOutputSpec(type="string")},
        steps=[
            ArtifactNavigateStep(id="s1", url="/"),
            ArtifactTypeStep(id="s2", target=Target(label="Member ID"), value="{{member_id}}"),
            ArtifactClickStep(id="s3", target=Target(role="button", name="Search")),
            ArtifactClickStep(id="s4", target=Target(role="link", name="{{member_id}}")),
            ArtifactExtractStep(
                id="s5", target=Target(label="Savings Balance"), output="balance"
            ),
        ],
        checkpoint=ArtifactCheckpoint(
            type="text_present", value="MemberServ 3.2 - Member Details"
        ),
    )


# ------------------------------------------------------- HandoffManager unit


def test_manager_starts_owned_by_automation():
    manager = HandoffManager()
    assert manager.owner == SessionOwner.AUTOMATION
    assert not manager.is_paused()
    assert manager.pending is None


def test_request_intervention_flips_ownership_to_human():
    manager = HandoffManager()
    request = manager.request_intervention(reason="UNRESOLVED_DIALOG", message="stuck")
    assert manager.owner == SessionOwner.HUMAN
    assert manager.is_paused()
    assert manager.pending == request


def test_cannot_request_a_second_intervention_while_one_is_pending():
    manager = HandoffManager()
    manager.request_intervention(reason="x", message="first")
    with pytest.raises(RuntimeError, match="already pending"):
        manager.request_intervention(reason="y", message="second")


def test_cannot_record_human_action_before_intervention_requested():
    manager = HandoffManager()
    with pytest.raises(RuntimeError, match="Cannot record a human action"):
        manager.record_human_action("did something")


def test_record_human_action_and_resume_full_cycle():
    manager = HandoffManager()
    manager.request_intervention(reason="x", message="stuck", step_id="s5")

    action = manager.record_human_action("Accepted the confirmation dialog manually.")
    assert action in manager.human_actions

    resolved = manager.resume()
    assert resolved.step_id == "s5"
    assert manager.owner == SessionOwner.AUTOMATION
    assert manager.pending is None


def test_resume_without_pending_intervention_raises():
    manager = HandoffManager()
    with pytest.raises(RuntimeError, match="No intervention is pending"):
        manager.resume()


# ------------------------------------------------ ReplayEngine integration


def test_replay_creates_a_real_intervention_request_on_unresolved_dialog(surface):
    manager = HandoffManager()
    engine = ReplayEngine(ToolExecutor(surface), handoff=manager)
    artifact = _lookup_balance_artifact()

    result = engine.replay(artifact, {"member_id": "77777"})

    assert result.status == ReplayStatus.HUMAN_INTERVENTION
    assert result.intervention is not None
    assert manager.is_paused()
    assert manager.pending is not None
    assert manager.pending.reason == OutcomeCode.UNRESOLVED_DIALOG.value
    assert manager.pending.artifact_id == "lookup_member_savings_balance"
    assert manager.pending.step_id == "s5"
    assert manager.pending.screenshot_path  # evidence captured for the operator


def test_full_pause_human_acts_resume_cycle_via_manager(surface):
    """The complete Phase 9 story: automation pauses, a human operator
    resolves it on the SAME live session, and automation resumes and
    finishes -- without ever restarting or opening a new session."""
    manager = HandoffManager()
    engine = ReplayEngine(ToolExecutor(surface), handoff=manager)
    artifact = _lookup_balance_artifact()

    paused = engine.replay(artifact, {"member_id": "77777"})
    assert paused.status == ReplayStatus.HUMAN_INTERVENTION
    assert manager.is_paused()

    # The human operator looks at manager.pending (goal/step/screenshot),
    # then acts directly on the SAME live Playwright session -- here
    # simulated the same way a human accepting the dialog would reveal
    # the data.
    surface._page.evaluate(
        "document.getElementById('accounts-block').style.display = 'block'"
    )
    manager.record_human_action("Manually revealed the restricted-view account data.")
    resolved = manager.resume()

    assert manager.owner == SessionOwner.AUTOMATION
    assert resolved.step_id == "s5"

    final = engine.resume(artifact, {"member_id": "77777"}, paused)
    assert final.status == ReplayStatus.SUCCESS
    assert final.outputs == {"balance": "$2200.00"}

    assert len(manager.human_actions) == 1
    assert "revealed" in manager.human_actions[0].description


def test_replay_without_handoff_manager_still_reports_human_intervention(surface):
    """No HandoffManager supplied -- the classification/pause behavior
    (Phase 7) is unaffected; there's simply no InterventionRequest object."""
    engine = ReplayEngine(ToolExecutor(surface))  # no handoff=
    artifact = _lookup_balance_artifact()
    result = engine.replay(artifact, {"member_id": "77777"})
    assert result.status == ReplayStatus.HUMAN_INTERVENTION
    assert result.intervention is None


# --------------------------------------------------------- mock operator console


@pytest.fixture()
def console_client():
    from handoff import routes

    # Reset the module-level shared manager between tests.
    routes._manager = HandoffManager()
    return TestClient(routes.app), routes._manager


def test_console_reports_no_pending_intervention_initially(console_client):
    client, _manager = console_client
    resp = client.get("/intervention/pending")
    assert resp.status_code == 200
    assert resp.json() == {"pending": None}


def test_console_shows_pending_intervention_and_resolves_it(console_client):
    client, manager = console_client
    manager.request_intervention(
        reason="UNRESOLVED_DIALOG", message="Stuck on a dialog.", step_id="s5"
    )

    pending_resp = client.get("/intervention/pending")
    assert pending_resp.json()["pending"]["reason"] == "UNRESOLVED_DIALOG"

    resolve_resp = client.post(
        "/intervention/resolve", json={"action_description": "Accepted the dialog."}
    )
    assert resolve_resp.status_code == 200
    assert resolve_resp.json()["owner"] == "automation"
    assert manager.owner == SessionOwner.AUTOMATION

    history_resp = client.get("/intervention/history")
    assert history_resp.json()["human_actions"][0]["description"] == "Accepted the dialog."


def test_console_resolve_without_pending_intervention_is_a_clean_409(console_client):
    client, _manager = console_client
    resp = client.post("/intervention/resolve", json={"action_description": "nothing to do"})
    assert resp.status_code == 409
