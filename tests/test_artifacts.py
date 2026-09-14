"""
Phase 5 tests:
  1. Artifact schema validation (discriminated union, output-declared check)
  2. Recording a real (scripted) discovery run into a structured artifact
  3. Save/load round-trip through JSON

Still no replay engine here -- we're only proving the artifact produced
from Phase 4's DiscoveryState is well-formed, correctly parameterized, and
independent of the LLM transcript.
"""
import json

import pytest
from pydantic import ValidationError

from agent.agent import DiscoveryAgent
from artifacts.recorder import record_artifact
from artifacts.schema import Artifact, ArtifactExtractStep, ArtifactTypeStep
from artifacts.storage import load_artifact, save_artifact
from surface.observation import Target
from tests.fakes import ScriptedModelClient, tool_use
from tools.executor import ToolExecutor

# --------------------------------------------------------------- schema only


def test_artifact_rejects_step_action_with_missing_required_field():
    with pytest.raises(ValidationError):
        Artifact(
            artifact_id="x",
            name="x",
            surface={"application": "fake-bank"},
            steps=[{"id": "s1", "action": "type", "target": {"role": "button", "name": "Search"}}],
            checkpoint={"type": "text_present", "value": "x"},
        )  # missing `value` for a type step


def test_artifact_rejects_output_referenced_but_not_declared():
    with pytest.raises(ValidationError, match="not declared"):
        Artifact(
            artifact_id="x",
            name="x",
            surface={"application": "fake-bank"},
            steps=[
                {
                    "id": "s1",
                    "action": "extract",
                    "target": {"label": "Savings Balance"},
                    "output": "balance",
                }
            ],
            outputs={},  # 'balance' referenced but never declared
            checkpoint={"type": "text_present", "value": "x"},
        )


def test_artifact_requires_at_least_one_step():
    with pytest.raises(ValidationError, match="at least one step"):
        Artifact(
            artifact_id="x",
            name="x",
            surface={"application": "fake-bank"},
            steps=[],
            checkpoint={"type": "text_present", "value": "x"},
        )


def test_valid_artifact_round_trips_through_json(tmp_path):
    artifact = Artifact(
        artifact_id="lookup_member_savings_balance",
        name="Lookup Member Savings Balance",
        description="Looks up a member and returns their savings balance.",
        surface={"application": "fake-bank"},
        inputs={"member_id": {"type": "string", "required": True}},
        outputs={"balance": {"type": "string"}},
        steps=[
            {"id": "s1", "action": "navigate", "url": "/"},
            {
                "id": "s2",
                "action": "type",
                "target": {"label": "Member ID"},
                "value": "{{member_id}}",
            },
            {
                "id": "s3",
                "action": "click",
                "target": {"role": "button", "name": "Search"},
            },
            {
                "id": "s4",
                "action": "click",
                "target": {"role": "link", "name": "{{member_id}}"},
            },
            {
                "id": "s5",
                "action": "extract",
                "target": {"label": "Savings Balance"},
                "output": "balance",
            },
        ],
        checkpoint={"type": "text_present", "value": "Member Details"},
    )
    path = save_artifact(artifact, tmp_path)
    assert path.exists()

    loaded = load_artifact(path)
    assert loaded == artifact
    assert isinstance(loaded.steps[1], ArtifactTypeStep)
    assert isinstance(loaded.steps[4], ArtifactExtractStep)


def test_load_artifact_rejects_corrupted_file(tmp_path):
    bad_path = tmp_path / "broken.json"
    bad_path.write_text(json.dumps({"artifact_id": "x", "name": "x"}))  # missing required fields
    with pytest.raises(ValidationError):
        load_artifact(bad_path)


def test_load_artifact_missing_file_raises_clean_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_artifact(tmp_path / "nope.json")


# --------------------------------------------------------- recording (live)


def test_record_artifact_from_successful_discovery_run(surface, tmp_path):
    executor = ToolExecutor(surface)
    script = [
        tool_use("type", {"target": {"label": "Member ID"}, "value": "12345"}),
        tool_use("click", {"target": {"role": "button", "name": "Search"}}),
        tool_use("click", {"target": {"role": "link", "name": "12345"}}),
        tool_use("extract", {"target": {"label": "Savings Balance"}}),
        tool_use(
            "finish_task",
            {"success": True, "outputs": {"balance": "$4250.00"}, "message": "done"},
        ),
    ]
    agent = DiscoveryAgent(executor, ScriptedModelClient(script), max_steps=10)
    result = agent.run(
        goal="Look up member 12345 and return their savings balance.", start_url="/"
    )
    assert result.state.status.value == "success"

    artifact = record_artifact(
        state=result.state,
        artifact_id="lookup_member_savings_balance",
        name="Lookup Member Savings Balance",
        parameters={"member_id": "12345"},
        application="fake-bank",
        description="Looks up a member and returns their savings balance.",
    )

    # navigate + type + click + click + extract = 5 steps (screenshot/observe excluded)
    assert [s.action for s in artifact.steps] == [
        "navigate",
        "type",
        "click",
        "click",
        "extract",
    ]

    # the concrete "12345" used during discovery was correctly parameterized
    type_step = artifact.steps[1]
    assert isinstance(type_step, ArtifactTypeStep)
    assert type_step.value == "{{member_id}}"
    assert type_step.target.label == "Member ID"  # unrelated field untouched

    click_link_step = artifact.steps[3]
    assert click_link_step.target.name == "{{member_id}}"

    # output name recovered automatically from finish_task's declared outputs
    extract_step = artifact.steps[4]
    assert isinstance(extract_step, ArtifactExtractStep)
    assert extract_step.output == "balance"
    assert artifact.outputs["balance"].type == "string"

    assert artifact.inputs["member_id"].type == "string"
    assert artifact.inputs["member_id"].required is True

    # checkpoint derived from the final page reached during discovery
    assert artifact.checkpoint.type == "text_present"
    assert artifact.checkpoint.value == "MemberServ 3.2 - Member Details"

    # round-trips cleanly
    path = save_artifact(artifact, tmp_path)
    reloaded = load_artifact(path)
    assert reloaded == artifact


def test_record_artifact_refuses_a_failed_run(surface):
    executor = ToolExecutor(surface)
    script = [
        tool_use("type", {"target": {"label": "Member ID"}, "value": "99999"}),
        tool_use("click", {"target": {"role": "button", "name": "Search"}}),
        tool_use(
            "finish_task",
            {"success": False, "outputs": {}, "message": "no such member"},
        ),
    ]
    agent = DiscoveryAgent(executor, ScriptedModelClient(script), max_steps=10)
    result = agent.run(goal="Look up member 99999.", start_url="/")
    assert result.state.status.value == "failed"

    with pytest.raises(ValueError, match="did not succeed"):
        record_artifact(
            state=result.state,
            artifact_id="x",
            name="x",
            parameters={"member_id": "99999"},
            application="fake-bank",
        )
