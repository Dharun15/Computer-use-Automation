"""
Generates the illustrative evidence bundle under /evidence/.

IMPORTANT -- read this before treating any of this as satisfying the
assignment's live-LLM requirement:

  - The DISCOVERY evidence produced by this script uses
    `tests.fakes.ScriptedModelClient`, NOT a real Anthropic API call. It
    exists to show the exact SHAPE of discovery evidence (state.json,
    transcript.json, summary.txt) and to prove the pipeline (discovery ->
    record artifact -> save) works end to end. It is explicitly labeled
    SIMULATED in its own summary.txt and is NOT a substitute for the real
    discovery run the brief requires. See README.md for the exact command
    to produce a genuine one with your own ANTHROPIC_API_KEY
    (`python -m agent.discover ...`) -- do that before submitting.

  - The REPLAY evidence produced by this script is 100% real: replay never
    uses an LLM at all, so there is nothing to simulate. These are genuine
    runs against the live fake-bank app via real Playwright automation.

Run with the fake bank already running on http://localhost:8000:
    python -m uvicorn app.main:app --port 8000 &
    python scripts/generate_demo_evidence.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

from agent.agent import DiscoveryAgent
from agent.recorder import save_discovery_evidence
from artifacts.recorder import record_artifact
from artifacts.storage import save_artifact
from handoff.manager import HandoffManager
from replay.engine import ReplayEngine
from replay.evidence import save_replay_evidence
from surface.playwright_surface import PlaywrightSurface
from tests.fakes import ScriptedModelClient, tool_use
from tools.executor import ToolExecutor

BASE_URL = "http://localhost:8000"


def generate_discovery_evidence(executor: ToolExecutor) -> None:
    print("--- Discovery (SIMULATED via ScriptedModelClient -- see module docstring) ---")
    script = [
        tool_use("type", {"target": {"label": "Member ID"}, "value": "12345"}),
        tool_use("click", {"target": {"role": "button", "name": "Search"}}),
        tool_use("click", {"target": {"role": "link", "name": "12345"}}),
        tool_use("extract", {"target": {"label": "Savings Balance"}}),
        tool_use(
            "finish_task",
            {
                "success": True,
                "outputs": {"balance": "$4250.00"},
                "message": "Found member 12345's savings balance.",
            },
        ),
    ]
    agent = DiscoveryAgent(executor, ScriptedModelClient(script))
    result = agent.run(
        goal="Look up member 12345 and return their savings balance.", start_url="/"
    )
    evidence_dir = save_discovery_evidence(result)

    # Mark this evidence unambiguously as simulated, right next to the data.
    marker = evidence_dir / "SIMULATED_NOT_A_REAL_LLM_RUN.txt"
    marker.write_text(
        "This discovery run used tests.fakes.ScriptedModelClient, not a real "
        "LLM. It demonstrates the evidence format and the discovery -> "
        "artifact pipeline only. Run `python -m agent.discover` with a real "
        "ANTHROPIC_API_KEY to produce the genuine run this project requires.\n"
    )
    print(f"Discovery status: {result.state.status.value} -> {evidence_dir}")

    artifact = record_artifact(
        state=result.state,
        artifact_id="lookup_member_savings_balance",
        name="Lookup Member Savings Balance",
        parameters={"member_id": "12345"},
        application="fake-bank",
        description="Looks up a member and returns their savings balance.",
    )
    path = save_artifact(artifact)
    print(f"Artifact saved: {path}")


def generate_replay_evidence(executor: ToolExecutor) -> None:
    print("--- Replay (100% real -- no LLM involved at any point) ---")
    from artifacts.storage import load_artifact

    artifact = load_artifact("artifacts/saved/lookup_member_savings_balance.json")
    handoff = HandoffManager()
    engine = ReplayEngine(executor, handoff=handoff)

    # 1. Success, with a DIFFERENT member than was recorded -- the actual
    #    point of parameterization.
    ok_result = engine.replay(artifact, {"member_id": "23456"})
    save_replay_evidence(artifact, {"member_id": "23456"}, ok_result)
    print(f"Replay (success, member 23456): {ok_result.status.value} {ok_result.outputs}")

    # 2. A business outcome -- member does not exist.
    nf_result = engine.replay(artifact, {"member_id": "99999"})
    save_replay_evidence(artifact, {"member_id": "99999"}, nf_result)
    print(f"Replay (business outcome, member 99999): {nf_result.status.value} {nf_result.outcome_code}")

    # 3. Human intervention -- an unresolved dialog blocks progress, then a
    #    human resolves it on the SAME session and replay resumes.
    paused = engine.replay(artifact, {"member_id": "77777"})
    save_replay_evidence(artifact, {"member_id": "77777"}, paused)
    print(f"Replay (paused, member 77777): {paused.status.value} {paused.outcome_code}")

    executor._surface._page.evaluate(
        "document.getElementById('accounts-block').style.display = 'block'"
    )
    handoff.record_human_action("Manually revealed the restricted-view account data.")
    handoff.resume()
    resumed = engine.resume(artifact, {"member_id": "77777"}, paused)
    save_replay_evidence(artifact, {"member_id": "77777"}, resumed)
    print(f"Replay (resumed after human intervention): {resumed.status.value} {resumed.outputs}")


def main() -> None:
    shutil.rmtree("evidence/discovery", ignore_errors=True)
    shutil.rmtree("evidence/replay", ignore_errors=True)
    shutil.rmtree("evidence/_screenshots", ignore_errors=True)

    surface = PlaywrightSurface(base_url=BASE_URL, headless=True)
    executor = ToolExecutor(surface)
    try:
        generate_discovery_evidence(executor)
        generate_replay_evidence(executor)
    finally:
        surface.close()


if __name__ == "__main__":
    main()
