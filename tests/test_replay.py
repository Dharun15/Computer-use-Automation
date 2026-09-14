"""
Phase 6/7 tests: ReplayEngine against the live fake-bank app, plus focused
unit tests for checkpoint verification and template substitution.

No LLM anywhere in this file -- that's the entire point of replay.
"""
import pytest

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
from replay.checkpoints import verify_checkpoint
from replay.engine import ReplayEngine
from replay.errors import OutcomeCode, ReplayStatus
from replay.locator import find_placeholders, substitute_string, substitute_target
from surface.observation import Observation, Target
from tools.executor import ToolExecutor


def _lookup_balance_artifact(output_type: str = "string") -> Artifact:
    """The canonical 'lookup member savings balance' capability, built
    directly (not via discovery) so these tests are fast and don't depend
    on the discovery agent or a scripted LLM."""
    return Artifact(
        artifact_id="lookup_member_savings_balance",
        name="Lookup Member Savings Balance",
        description="Looks up a member and returns their savings balance.",
        surface=ArtifactSurfaceInfo(application="fake-bank"),
        inputs={"member_id": {"type": "string", "required": True}},
        outputs={"balance": ArtifactOutputSpec(type=output_type)},
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


@pytest.fixture()
def engine(surface):
    return ReplayEngine(ToolExecutor(surface), max_retries=1, retry_backoff_seconds=0.1)


# --------------------------------------------------------------- live replay


def test_replay_succeeds_with_a_different_member_than_was_recorded(engine):
    """The actual point of the artifact: it generalizes. This was recorded
    conceptually against member 12345 but never mentions that ID anywhere
    -- replaying with 23456 must return 23456's own data."""
    artifact = _lookup_balance_artifact()
    result = engine.replay(artifact, {"member_id": "23456"})

    assert result.status == ReplayStatus.SUCCESS
    assert result.outputs == {"balance": "$980.50"}
    assert result.steps_completed == result.total_steps == 5


def test_replay_succeeds_with_the_originally_recorded_member_too(engine):
    artifact = _lookup_balance_artifact()
    result = engine.replay(artifact, {"member_id": "12345"})

    assert result.ok
    assert result.outputs == {"balance": "$4250.00"}


def test_replay_coerces_numeric_output_type(engine):
    artifact = _lookup_balance_artifact(output_type="number")
    result = engine.replay(artifact, {"member_id": "34567"})

    assert result.ok
    assert result.outputs == {"balance": 15200.00}
    assert isinstance(result.outputs["balance"], float)


def test_replay_fails_cleanly_on_missing_required_input(engine):
    artifact = _lookup_balance_artifact()
    result = engine.replay(artifact, {})

    assert result.status == ReplayStatus.HARD_FAILURE
    assert "Missing required input 'member_id'" in result.message
    assert result.steps_completed == 0  # never touched the browser


def test_replay_fails_cleanly_on_wrong_input_type(engine):
    artifact = _lookup_balance_artifact()
    artifact.inputs["member_id"].type = "number"
    result = engine.replay(artifact, {"member_id": "not-a-number"})

    assert result.status == ReplayStatus.HARD_FAILURE
    assert "must be a number" in result.message
    assert result.steps_completed == 0


def test_replay_classifies_member_not_found_as_business_outcome(engine):
    """Member 99999 doesn't exist -- after Search, the app shows 'No
    Records', so the following click-the-member-link step has nothing to
    click. This is a legitimate business outcome, not a crash."""
    artifact = _lookup_balance_artifact()
    result = engine.replay(artifact, {"member_id": "99999"})

    assert result.status == ReplayStatus.BUSINESS_OUTCOME
    assert result.outcome_code == OutcomeCode.MEMBER_NOT_FOUND
    assert result.failed_step_id == "s4"
    assert result.failed_step_action == "click"
    assert result.steps_completed == 3  # navigate, type, click(Search) all succeeded
    assert result.total_steps == 5
    assert result.screenshot_path  # evidence captured on the failing path


def test_replay_classifies_permission_denied_as_hard_failure(engine):
    """Member 55555 -- the click to open the member succeeds (it's a valid
    link), but the resulting page is a 403 Access Denied, so there's no
    'Savings Balance' to extract. The classifier correctly identifies this
    as a hard failure from the page reached, not a generic not-found."""
    artifact = _lookup_balance_artifact()
    result = engine.replay(artifact, {"member_id": "55555"})

    assert result.status == ReplayStatus.HARD_FAILURE
    assert result.outcome_code == OutcomeCode.PERMISSION_DENIED
    assert result.failed_step_id == "s5"  # the extract step, on the 403 page
    assert result.steps_completed == 4  # navigate/type/click/click all succeeded


def test_replay_classifies_app_error_as_hard_failure(engine):
    artifact = _lookup_balance_artifact()
    result = engine.replay(artifact, {"member_id": "00000"})

    assert result.status == ReplayStatus.HARD_FAILURE
    assert result.outcome_code == OutcomeCode.APPLICATION_ERROR


def test_replay_escalates_unresolved_dialog_to_human_intervention(engine):
    """Member 77777's balance is hidden behind a confirm() dialog that
    gets auto-dismissed -- extract then finds nothing visible. This isn't
    a page we recognize as a business outcome or a permission/app error,
    so it correctly escalates to a human rather than failing outright."""
    artifact = _lookup_balance_artifact()
    result = engine.replay(artifact, {"member_id": "77777"})

    assert result.status == ReplayStatus.HUMAN_INTERVENTION
    assert result.outcome_code == OutcomeCode.UNRESOLVED_DIALOG
    assert result.failed_step_id == "s5"  # the extract step
    assert result.steps_completed == 4


def test_replay_fails_when_checkpoint_mismatch_is_unrecognized(engine):
    """All steps succeed, and the final page is a perfectly normal one --
    it's just not the page this (deliberately wrong) checkpoint expects.
    No known business/hard-failure pattern applies, so this correctly
    falls back to a generic, but still fully debuggable, hard failure."""
    artifact = _lookup_balance_artifact()
    artifact.checkpoint = ArtifactCheckpoint(
        type="text_present", value="This Text Will Never Appear On The Page"
    )
    result = engine.replay(artifact, {"member_id": "12345"})

    assert result.status == ReplayStatus.HARD_FAILURE
    assert result.outcome_code == OutcomeCode.CHECKPOINT_NOT_SATISFIED
    assert result.steps_completed == 5  # every step itself succeeded
    assert "This Text Will Never Appear" in result.expected


def test_replay_via_save_and_load_round_trip(engine, tmp_path):
    """Confirms replay works against an artifact that went through a real
    JSON save/load cycle, not just the in-memory object."""
    from artifacts.storage import load_artifact, save_artifact

    artifact = _lookup_balance_artifact()
    path = save_artifact(artifact, tmp_path)
    loaded = load_artifact(path)

    result = engine.replay(loaded, {"member_id": "23456"})
    assert result.ok
    assert result.outputs == {"balance": "$980.50"}


def test_resume_continues_from_paused_step_not_from_scratch(engine):
    """Simulates the human-intervention flow end to end: replay pauses on
    the dialog-gated member, a human resolves it out of band, and resume()
    picks up from the SAME point using the outputs already collected --
    it does not restart the whole workflow."""
    artifact = _lookup_balance_artifact()
    paused = engine.replay(artifact, {"member_id": "77777"})
    assert paused.status == ReplayStatus.HUMAN_INTERVENTION
    assert paused.outputs == {}  # extract never completed

    # A human operator would now act on the SAME live session (accept the
    # dialog, or manually read the value) -- here we simulate that by
    # directly telling the underlying surface to reveal the hidden data,
    # exactly as a human clicking "accept" on the confirm() would.
    engine._executor._surface._page.evaluate(
        "document.getElementById('accounts-block').style.display = 'block'"
    )

    resumed = engine.resume(artifact, {"member_id": "77777"}, paused)
    assert resumed.status == ReplayStatus.SUCCESS
    assert resumed.outputs == {"balance": "$2200.00"}
    assert resumed.steps_completed == 5


def test_resume_rejects_a_non_paused_result(engine):
    artifact = _lookup_balance_artifact()
    successful = engine.replay(artifact, {"member_id": "12345"})
    with pytest.raises(ValueError, match="only valid for a result with status=HUMAN_INTERVENTION"):
        engine.resume(artifact, {"member_id": "12345"}, successful)


# --------------------------------------------------------- retry mechanism
# The fake bank's slow_load member sleeps a fixed, deterministic 3s -- it
# can't simulate genuine flaky-then-fast timing (first attempt slow,
# second fast), which is what a retry is actually for. Rather than fight
# real browser timing to approximate that, the retry mechanism itself is
# tested directly and deterministically here with a scripted executor
# double, isolated from Playwright entirely.


class _ScriptedExecutor:
    """Returns the next scripted ToolExecutionResult on each .execute()
    call, regardless of arguments -- enough to drive ReplayEngine's retry
    loop in isolation."""

    def __init__(self, script):
        self._script = list(script)
        self.calls = 0

    def execute(self, action, raw_input=None):
        result = self._script[min(self.calls, len(self._script) - 1)]
        self.calls += 1
        return result


def test_retry_succeeds_on_second_attempt_after_one_timeout():
    from tools.executor import ToolExecutionResult

    script = [
        ToolExecutionResult(action="click", valid=True, status="timeout", message="slow"),
        ToolExecutionResult(action="click", valid=True, status="ok", message="clicked"),
    ]
    fake = _ScriptedExecutor(script)
    engine = ReplayEngine(fake, max_retries=2, retry_backoff_seconds=0.01)

    step = ArtifactClickStep(id="s1", target=Target(role="button", name="Search"))
    attempts = []
    result = engine._execute_with_retries(step, {"target": {}}, attempts)

    assert result.status == "ok"
    assert fake.calls == 2
    assert len(attempts) == 1
    assert attempts[0].outcome == "retried_ok"
    assert attempts[0].reason == "timeout"


def test_retry_budget_exhausts_and_reports_it():
    from tools.executor import ToolExecutionResult

    always_timeout = ToolExecutionResult(action="click", valid=True, status="timeout", message="slow")
    fake = _ScriptedExecutor([always_timeout])  # every call times out
    engine = ReplayEngine(fake, max_retries=2, retry_backoff_seconds=0.01)

    step = ArtifactClickStep(id="s1", target=Target(role="button", name="Search"))
    attempts = []
    result = engine._execute_with_retries(step, {"target": {}}, attempts)

    assert result.status == "timeout"
    assert fake.calls == 3  # 1 initial attempt + 2 retries
    assert len(attempts) == 2
    assert attempts[-1].outcome == "exhausted"


# ------------------------------------------------------------ checkpoints


def _obs(**kwargs) -> Observation:
    defaults = dict(url="http://x/members/1", title="Member Details", status_code=200)
    defaults.update(kwargs)
    return Observation(**defaults)


def test_checkpoint_text_present_matches_title():
    cp = ArtifactCheckpoint(type="text_present", value="Member Details")
    result = verify_checkpoint(cp, _obs(title="Member Details"))
    assert result.passed


def test_checkpoint_text_present_matches_body_text():
    cp = ArtifactCheckpoint(type="text_present", value="Savings Balance")
    result = verify_checkpoint(cp, _obs(text_summary="Savings Balance $100"))
    assert result.passed


def test_checkpoint_text_present_fails_when_absent():
    cp = ArtifactCheckpoint(type="text_present", value="Nope Not Here")
    result = verify_checkpoint(cp, _obs())
    assert not result.passed
    assert "Nope Not Here" in result.expected


def test_checkpoint_url_contains():
    cp = ArtifactCheckpoint(type="url_contains", value="/members/")
    assert verify_checkpoint(cp, _obs(url="http://x/members/12345")).passed
    assert not verify_checkpoint(cp, _obs(url="http://x/search")).passed


def test_checkpoint_status_code():
    cp = ArtifactCheckpoint(type="status_code", value="200")
    assert verify_checkpoint(cp, _obs(status_code=200)).passed
    assert not verify_checkpoint(cp, _obs(status_code=403)).passed


def test_checkpoint_unknown_type_raises():
    cp = ArtifactCheckpoint.model_construct(type="bogus", value="x")
    with pytest.raises(ValueError, match="Unknown checkpoint type"):
        verify_checkpoint(cp, _obs())


# --------------------------------------------------------- template substitution


def test_find_placeholders():
    assert find_placeholders("hello {{member_id}} and {{other}}") == {"member_id", "other"}
    assert find_placeholders("no placeholders here") == set()


def test_substitute_string_replaces_all_occurrences():
    result = substitute_string("id={{member_id}}&again={{member_id}}", {"member_id": "999"})
    assert result == "id=999&again=999"


def test_substitute_string_raises_on_unknown_placeholder():
    with pytest.raises(ValueError, match="member_id"):
        substitute_string("{{member_id}}", {})


def test_substitute_target_only_touches_string_fields():
    target = Target(role="link", name="{{member_id}}", nth=1)
    result = substitute_target(target, {"member_id": "42"})
    assert result.name == "42"
    assert result.role == "link"
    assert result.nth == 1  # untouched, not a template field
