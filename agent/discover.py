"""
CLI entry point: run the discovery agent against a live target with a real
LLM. This is the ONE path in this whole project that must be run for real
against Anthropic's API -- see README.md.

    python -m agent.discover --goal "Look up member 12345 and return their savings balance." \\
        --record-artifact --artifact-id lookup_member_savings_balance \\
        --artifact-name "Lookup Member Savings Balance" \\
        --param member_id=12345
"""
from __future__ import annotations

import argparse
import sys
from urllib.parse import urlsplit

from agent.agent import DiscoveryAgent
from agent.llm_client import AnthropicModelClient, GeminiModelClient
from agent.recorder import save_discovery_evidence
from artifacts.recorder import record_artifact
from artifacts.storage import save_artifact
from safety.policy import PolicyEngine, SafetyPolicy
from surface.playwright_surface import PlaywrightSurface
from tools.executor import ToolExecutor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the LLM discovery agent against a live target application."
    )
    parser.add_argument("--goal", required=True, help="Natural-language goal for the agent.")
    parser.add_argument("--start-url", default="/", help="Entry point, relative to --base-url.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--provider",
        choices=["anthropic", "gemini"],
        default="anthropic",
        help="Which LLM provider to use. 'gemini' has a genuinely free, "
        "no-credit-card tier (GEMINI_API_KEY) if you don't have Anthropic "
        "credits -- the provider is fully swappable, this doesn't change "
        "anything else about how the agent runs.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Defaults to claude-sonnet-4-5 for --provider anthropic, "
        "or gemini-2.5-flash for --provider gemini.",
    )
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--headed", action="store_true", help="Show the browser window.")
    parser.add_argument(
        "--record-artifact",
        action="store_true",
        help="If the run succeeds, record it as a reusable, saved artifact.",
    )
    parser.add_argument("--artifact-id")
    parser.add_argument("--artifact-name")
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="A concrete value used this run that should become a named "
        "input, e.g. --param member_id=12345. Repeatable.",
    )
    parser.add_argument("--application", default="fake-bank")
    args = parser.parse_args()

    hostname = urlsplit(args.base_url).hostname or "localhost"
    policy = PolicyEngine(SafetyPolicy(allowed_domains=[hostname]))

    surface = PlaywrightSurface(base_url=args.base_url, headless=not args.headed)
    executor = ToolExecutor(surface, policy=policy)

    if args.provider == "gemini":
        model = GeminiModelClient(model=args.model or "gemini-2.5-flash")
    else:
        model = AnthropicModelClient(model=args.model or "claude-sonnet-4-5")

    agent = DiscoveryAgent(executor, model, max_steps=args.max_steps)

    try:
        result = agent.run(goal=args.goal, start_url=args.start_url)
    finally:
        pass  # surface is closed further down, after any artifact recording

    evidence_dir = save_discovery_evidence(result)

    print(f"Status: {result.state.status.value}")
    print(f"Stop reason: {result.state.stop_reason}")
    print(f"Steps executed: {result.state.step_count}")
    print(f"Outputs: {result.state.outputs}")
    print(f"Final message: {result.state.final_message}")
    print(f"Evidence saved to: {evidence_dir}")

    if args.record_artifact:
        if result.state.status.value != "success":
            print("Refusing to record an artifact: the run did not succeed.", file=sys.stderr)
        else:
            params = dict(p.split("=", 1) for p in args.param)
            artifact = record_artifact(
                state=result.state,
                artifact_id=args.artifact_id or "recorded_capability",
                name=args.artifact_name or args.artifact_id or "Recorded Capability",
                parameters=params,
                application=args.application,
            )
            path = save_artifact(artifact)
            print(f"Artifact saved to: {path}")

    surface.close()


if __name__ == "__main__":
    main()
