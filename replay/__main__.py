"""
CLI entry point: deterministically replay a saved artifact. No LLM call
happens anywhere in this path -- this is the production execution an AI
agent would trigger.

    python -m replay --artifact artifacts/saved/lookup_member_savings_balance.json \
        --input member_id=12345
"""
from __future__ import annotations

import argparse
from urllib.parse import urlsplit

from artifacts.storage import load_artifact
from handoff.manager import HandoffManager
from replay.engine import ReplayEngine
from replay.evidence import save_replay_evidence, save_stability_evidence
from replay.stability import run_stability_check
from safety.policy import PolicyEngine, SafetyPolicy
from surface.playwright_surface import PlaywrightSurface
from tools.executor import ToolExecutor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministically replay a saved artifact (no LLM)."
    )
    parser.add_argument("--artifact", required=True, help="Path to a saved artifact JSON file.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="An input value the artifact needs, e.g. --input member_id=12345. Repeatable.",
    )
    parser.add_argument("--headed", action="store_true", help="Show the browser window.")
    parser.add_argument(
        "--stability-runs",
        type=int,
        default=1,
        metavar="N",
        help="Replay the same artifact/inputs N times and report a "
        "success-rate/timing stability signal instead of a single result.",
    )
    args = parser.parse_args()

    artifact = load_artifact(args.artifact)
    inputs = dict(p.split("=", 1) for p in args.input)

    hostname = urlsplit(args.base_url).hostname or "localhost"
    policy = PolicyEngine(SafetyPolicy(allowed_domains=[hostname]))

    surface = PlaywrightSurface(base_url=args.base_url, headless=not args.headed)
    executor = ToolExecutor(surface, policy=policy)
    engine = ReplayEngine(executor, handoff=HandoffManager())

    if args.stability_runs > 1:
        report = run_stability_check(engine, artifact, inputs, n=args.stability_runs)
        evidence_dir = save_stability_evidence(report)
        print(f"Artifact: {report.artifact_id}")
        print(f"Runs: {report.total_runs}  Successes: {report.successes}  Failures: {report.failures}")
        print(f"Success rate: {report.success_rate:.0%}")
        print(f"Status counts: {report.status_counts}")
        print(
            f"Duration (s) -- mean: {report.mean_duration_seconds:.2f}, "
            f"min: {report.min_duration_seconds:.2f}, max: {report.max_duration_seconds:.2f}"
        )
        print(f"Evidence saved to: {evidence_dir}")
        surface.close()
        return

    result = engine.replay(artifact, inputs)
    evidence_dir = save_replay_evidence(artifact, inputs, result)

    print(f"Status: {result.status.value}")
    if result.outcome_code:
        print(f"Outcome code: {result.outcome_code.value}")
    print(f"Steps: {result.steps_completed}/{result.total_steps}")
    print(f"Outputs: {result.outputs}")
    print(f"Message: {result.message}")
    if result.expected:
        print(f"Expected: {result.expected}")
        print(f"Observed: {result.observed}")
    print(f"Evidence saved to: {evidence_dir}")

    surface.close()


if __name__ == "__main__":
    main()