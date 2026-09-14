"""
Tests for the agent-facing capability catalog (stretch goal 1).

Uses FastAPI's TestClient against `capabilities.service.app`, with a real
saved artifact and real invocations against the live fake-bank app --
`/invoke` genuinely runs `ReplayEngine`, no mocking of the replay path.
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
from artifacts.storage import save_artifact
from surface.observation import Target


def _lookup_balance_artifact() -> Artifact:
    return Artifact(
        artifact_id="lookup_member_savings_balance",
        name="Lookup Member Savings Balance",
        description="Looks up a member and returns their savings balance.",
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


@pytest.fixture()
def catalog_client(tmp_path, monkeypatch):
    """A capability catalog pointed at a temp artifact directory containing
    exactly one saved artifact -- isolated from whatever's in the real
    artifacts/saved/ during test runs."""
    from capabilities import service

    save_artifact(_lookup_balance_artifact(), tmp_path)
    monkeypatch.setattr(service, "DEFAULT_ARTIFACT_DIR", tmp_path)
    return TestClient(service.app)


def test_list_capabilities_shows_the_saved_artifact(catalog_client):
    resp = catalog_client.get("/capabilities")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["artifact_id"] == "lookup_member_savings_balance"
    assert "member_id" in body[0]["inputs"]
    assert "balance" in body[0]["outputs"]


def test_list_capabilities_is_empty_when_no_artifacts_saved(tmp_path, monkeypatch):
    from capabilities import service

    monkeypatch.setattr(service, "DEFAULT_ARTIFACT_DIR", tmp_path)
    client = TestClient(service.app)
    resp = client.get("/capabilities")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_capability_detail_includes_steps_and_checkpoint(catalog_client):
    resp = catalog_client.get("/capabilities/lookup_member_savings_balance")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["steps"]) == 5
    assert body["checkpoint"]["value"] == "MemberServ 3.2 - Member Details"


def test_get_unknown_capability_is_404(catalog_client):
    resp = catalog_client.get("/capabilities/does_not_exist")
    assert resp.status_code == 404


def test_invoke_unknown_capability_is_404(catalog_client):
    resp = catalog_client.post("/capabilities/does_not_exist/invoke", json={"inputs": {}})
    assert resp.status_code == 404


def test_invoke_capability_runs_a_real_replay_against_the_live_app(
    catalog_client, fake_bank_url
):
    """The core proof: an HTTP call an AI agent would make actually drives
    a real browser through ReplayEngine and returns real data -- with a
    DIFFERENT member than any artifact was recorded against, same as the
    CLI path already proves."""
    resp = catalog_client.post(
        f"/capabilities/lookup_member_savings_balance/invoke",
        json={"inputs": {"member_id": "23456"}, "base_url": fake_bank_url},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["outputs"] == {"balance": "$980.50"}
    assert body["steps_completed"] == body["total_steps"] == 5


def test_invoke_capability_reports_business_outcome_not_a_500(catalog_client, fake_bank_url):
    resp = catalog_client.post(
        "/capabilities/lookup_member_savings_balance/invoke",
        json={"inputs": {"member_id": "99999"}, "base_url": fake_bank_url},
    )
    assert resp.status_code == 200  # the HTTP call itself succeeded
    body = resp.json()
    assert body["status"] == "business_outcome"
    assert body["outcome_code"] == "MEMBER_NOT_FOUND"


def test_invoke_capability_missing_required_input_is_a_clean_hard_failure(
    catalog_client, fake_bank_url
):
    resp = catalog_client.post(
        "/capabilities/lookup_member_savings_balance/invoke",
        json={"inputs": {}, "base_url": fake_bank_url},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "hard_failure"
    assert "member_id" in body["message"]