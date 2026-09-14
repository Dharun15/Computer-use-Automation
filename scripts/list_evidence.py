"""
Quick one-off helper: prints a compact, one-line-per-folder summary of
everything under evidence/discovery and evidence/replay, so you can decide
what to keep/delete without opening each summary.txt by hand.

Run from the project root:
    python scripts/list_evidence.py
"""
from __future__ import annotations

import json
from pathlib import Path


def list_discovery():
    base = Path("evidence/discovery")
    if not base.exists():
        return
    print("=== evidence/discovery ===")
    for folder in sorted(base.iterdir()):
        if not folder.is_dir():
            continue
        state_path = folder / "state.json"
        marker = "SIMULATED" if (folder / "SIMULATED_NOT_A_REAL_LLM_RUN.txt").exists() else "real"
        if not state_path.exists():
            print(f"{folder.name}: (no state.json)")
            continue
        state = json.loads(state_path.read_text())
        print(
            f"{folder.name} [{marker}]: status={state.get('status')} "
            f"steps={state.get('step_count')} outputs={state.get('outputs')}"
        )


def list_replay():
    base = Path("evidence/replay")
    if not base.exists():
        return
    print("\n=== evidence/replay ===")
    for folder in sorted(base.iterdir()):
        if not folder.is_dir():
            continue
        result_path = folder / "result.json"
        inputs_path = folder / "inputs.json"
        stability_path = folder / "stability_report.json"
        if stability_path.exists():
            report = json.loads(stability_path.read_text())
            print(
                f"{folder.name} [STABILITY]: runs={report.get('total_runs')} "
                f"success_rate={report.get('success_rate')}"
            )
            continue
        if not result_path.exists():
            print(f"{folder.name}: (no result.json)")
            continue
        result = json.loads(result_path.read_text())
        inputs = json.loads(inputs_path.read_text()) if inputs_path.exists() else {}
        print(
            f"{folder.name}: inputs={inputs} status={result.get('status')} "
            f"outcome_code={result.get('outcome_code')} outputs={result.get('outputs')}"
        )


if __name__ == "__main__":
    list_discovery()
    list_replay()