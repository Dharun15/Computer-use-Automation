"""
Demonstrates the full Phase 9 human-intervention cycle against whatever
artifact is currently saved at artifacts/saved/<artifact_id>.json --
using the SAME live browser session throughout, which is the entire point
of the mechanism (see handoff/manager.py and REPORT.md Section 5).

This is a standalone script, not part of the `replay` CLI, specifically
because the CLI opens a browser, does one replay, and closes it -- there
is no way to "pause and come back later" across two separate CLI
invocations without keeping the same process (and therefore the same
browser) alive the whole time. This script keeps everything in one
process precisely so the pause -> human acts -> resume sequence is real,
not simulated across a restart.

Run with the fake bank already running on http://localhost:8000:
    python -m uvicorn app.main:app --port 8000 &
    python scripts/demo_human_intervention.py --artifact artifacts/saved/lookup_member_savings_balance.json --member-id 77777
"""
from __future__ import annotations

import argparse

from artifacts.storage import load_artifact
from handoff.manager import HandoffManager
from replay.engine import ReplayEngine
from replay.errors import ReplayStatus
from replay.evidence import save_replay_evidence
from surface.playwright_surface import PlaywrightSurface
from tools.executor import ToolExecutor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--member-id", default="77777", help="A member whose page has the unexpected dialog.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()

    artifact = load_artifact(args.artifact)
    inputs = {"member_id": args.member_id}

    surface = PlaywrightSurface(base_url=args.base_url, headless=True)
    executor = ToolExecutor(surface)
    handoff = HandoffManager()
    engine = ReplayEngine(executor, handoff=handoff)

    print(f"--- Replaying {artifact.artifact_id} with member_id={args.member_id} ---")
    paused = engine.replay(artifact, inputs)
    save_replay_evidence(artifact, inputs, paused)
    print(f"Status: {paused.status.value}")

    if paused.status != ReplayStatus.HUMAN_INTERVENTION:
        print(
            "This member/artifact combination did not trigger human intervention "
            "-- nothing to resume. (Try --member-id 77777 against an artifact "
            "recorded the same way this project's fake bank expects.)"
        )
        surface.close()
        return

    print("\n*** AUTOMATION PAUSED -- SAME LIVE SESSION, WAITING FOR A HUMAN ***")
    print(f"Reason: {handoff.pending.reason}")
    print(f"Message: {handoff.pending.message}")
    print(f"Failing step: {handoff.pending.step_id}")
    print(f"Screenshot for the operator to look at: {handoff.pending.screenshot_path}")
    print(f"Session owner is now: {handoff.owner.value}")

    input("\nPress Enter to simulate a human operator resolving this on the live session...")

    # What a human would actually do here: look at the live browser (or the
    # screenshot above), and manually accept whatever dialog/condition is
    # blocking progress. We simulate that exact end state directly on the
    # SAME PlaywrightSurface/page instance -- nothing is reopened.
    surface._page.evaluate(
        "document.getElementById('accounts-block').style.display = 'block'"
    )
    handoff.record_human_action(
        "Manually accepted the confirmation dialog and revealed the account data."
    )
    resolved = handoff.resume()
    print(f"\nHuman action recorded. Session owner is now: {handoff.owner.value}")
    print(f"Resolved intervention: {resolved.reason}")

    print("\n--- Resuming automation from exactly where it paused ---")
    final = engine.resume(artifact, inputs, paused)
    save_replay_evidence(artifact, inputs, final)
    print(f"Status: {final.status.value}")
    print(f"Outputs: {final.outputs}")

    surface.close()


if __name__ == "__main__":
    main()