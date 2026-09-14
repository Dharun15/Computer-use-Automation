"""
Tests for the multi-run stability tool (stretch goal 2).

Two layers:
  1. Against the REAL fake bank + ReplayEngine -- proves the tool actually
     drives real replays N times. The fake bank is deterministic, so every
     run of the same input lands on the same status -- that's expected and
     correctly reported as 100% consistent, not a limitation of the tool.
  2. Against a scripted stand-in engine -- proves the counting/reporting
     logic correctly handles a genuine MIX of outcomes, which the
     deterministic fake bank can never produce on its own.
"""
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
from replay.engine import ReplayEngine
from replay.errors import OutcomeCode, ReplayResult, ReplayStatus
from replay.stability import run_stability_check
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


# --------------------------------------------------------------- live engine


def test_stability_check_against_real_deterministic_app_is_fully_consistent(surface):
    """The fake bank never actually flakes -- N runs of a happy-path
    member should all succeed, proving the loop and counting work, even
    though there's no real flakiness for this specific target to surface."""
    engine = ReplayEngine(ToolExecutor(surface))
    artifact = _lookup_balance_artifact()

    report = run_stability_check(engine, artifact, {"member_id": "12345"}, n=3)

    assert report.total_runs == 3
    assert report.successes == 3
    assert report.failures == 0
    assert report.success_rate == 1.0
    assert report.status_counts == {"success": 3}
    assert len(report.runs) == 3
    assert all(r.duration_seconds > 0 for r in report.runs)
    assert report.mean_duration_seconds > 0


def test_stability_check_business_outcome_member_is_also_consistent(surface):
    engine = ReplayEngine(ToolExecutor(surface))
    artifact = _lookup_balance_artifact()

    report = run_stability_check(engine, artifact, {"member_id": "99999"}, n=3)

    assert report.successes == 0
    assert report.failures == 3
    assert report.success_rate == 0.0
    assert report.status_counts == {"business_outcome": 3}


# ------------------------------------------------------- scripted (mixed outcomes)


class _ScriptedEngine:
    def __init__(self, results: list[ReplayResult]):
        self._results = list(results)
        self.calls = 0

    def replay(self, artifact, inputs):
        result = self._results[self.calls]
        self.calls += 1
        return result


def test_stability_report_correctly_tallies_a_genuine_mix_of_outcomes():
    """A real production target CAN be genuinely flaky -- this proves the
    reporting logic handles that correctly, using a scripted engine since
    our deterministic fake bank cannot produce a real mix on its own."""
    script = [
        ReplayResult(status=ReplayStatus.SUCCESS, outputs={"balance": "$1.00"}),
        ReplayResult(status=ReplayStatus.SUCCESS, outputs={"balance": "$1.00"}),
        ReplayResult(
            status=ReplayStatus.HARD_FAILURE,
            outcome_code=OutcomeCode.UNKNOWN,
            message="transient failure",
        ),
        ReplayResult(status=ReplayStatus.SUCCESS, outputs={"balance": "$1.00"}),
        ReplayResult(
            status=ReplayStatus.BUSINESS_OUTCOME,
            outcome_code=OutcomeCode.MEMBER_NOT_FOUND,
        ),
    ]
    engine = _ScriptedEngine(script)
    artifact = _lookup_balance_artifact()

    report = run_stability_check(engine, artifact, {"member_id": "12345"}, n=5)

    assert report.total_runs == 5
    assert report.successes == 3
    assert report.failures == 2
    assert report.success_rate == 0.6
    assert report.status_counts == {
        "success": 3,
        "hard_failure": 1,
        "business_outcome": 1,
    }
    assert [r.status for r in report.runs] == [
        "success",
        "success",
        "hard_failure",
        "success",
        "business_outcome",
    ]
    assert engine.calls == 5


def test_stability_report_with_zero_runs_does_not_divide_by_zero():
    engine = _ScriptedEngine([])
    artifact = _lookup_balance_artifact()

    report = run_stability_check(engine, artifact, {"member_id": "12345"}, n=0)

    assert report.total_runs == 0
    assert report.success_rate == 0.0
    assert report.mean_duration_seconds == 0.0