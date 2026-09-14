"""
Turns a successful `DiscoveryState` into a reusable `Artifact`.

This reads ONLY `state.steps` (the structured, typed action log) -- never
`DiscoveryResult.transcript`. That boundary is deliberate and enforced by
this function's signature: it doesn't even accept a transcript argument.

Parameterization strategy: the caller tells us which concrete values used
during discovery should become named inputs (e.g. `{"member_id": "12345"}`).
Every occurrence of that literal value inside a step's target fields, typed
value, or URL is replaced with a `{{member_id}}` placeholder. This is the
same "record with a real example, then parameterize by substitution"
approach used by most record/replay tools -- it's simple, and critically,
it's reviewable: a human can see exactly which literal became which
parameter, rather than the recorder guessing at intent.

Output naming: the model's `finish_task` call already declares named
outputs with their extracted values (e.g. `{"balance": "$4250.00"}`). We
match each `extract` step's actual extracted value against those declared
outputs to recover the intended name automatically -- so the caller doesn't
have to separately specify "which extract step produces which output."
Known limitation: if two extract steps happen to produce identical values,
this matching is ambiguous (documented in REPORT.md's Cuts section).
"""
from __future__ import annotations

import warnings
from datetime import datetime, timezone
from typing import Optional

from agent.state import DiscoveryState, DiscoveryStatus
from artifacts.schema import (
    Artifact,
    ArtifactCheckpoint,
    ArtifactClickStep,
    ArtifactExtractStep,
    ArtifactInputSpec,
    ArtifactNavigateStep,
    ArtifactOutputSpec,
    ArtifactStep,
    ArtifactSurfaceInfo,
    ArtifactTypeStep,
)
from surface.observation import Target

# These discovery actions are bookkeeping, not application steps -- they
# never appear in a recorded artifact.
_NON_ARTIFACT_ACTIONS = {"observe_page", "screenshot"}

_TARGET_STRING_FIELDS = ("role", "name", "label", "text", "css")


def record_artifact(
    state: DiscoveryState,
    artifact_id: str,
    name: str,
    parameters: dict[str, str],
    application: str,
    description: str = "",
    input_types: Optional[dict[str, str]] = None,
) -> Artifact:
    """
    `parameters`: concrete value used during this discovery run -> input
    name it should become, e.g. `{"member_id": "12345"}`.
    """
    if state.status != DiscoveryStatus.SUCCESS:
        raise ValueError(
            f"Refusing to record an artifact from a run that did not "
            f"succeed (status={state.status.value}). Only a successful "
            f"discovery run demonstrates a working capability."
        )

    input_types = input_types or {}
    output_name_by_value = {str(v): k for k, v in state.outputs.items()}

    steps: list[ArtifactStep] = []
    outputs: dict[str, ArtifactOutputSpec] = {}
    fallback_output_counter = 0

    for recorded in state.steps:
        if recorded.action in _NON_ARTIFACT_ACTIONS:
            continue
        if recorded.result_status != "ok":
            # A discovery run can legitimately retry after a failed attempt
            # (e.g. the model tries role+name, it doesn't resolve, then it
            # retries with a label instead). Only the attempt that actually
            # succeeded is part of the real, reusable workflow -- recording
            # the failed attempt too would make replay execute a step that
            # never worked in the first place and fail immediately.
            continue

        step_id = f"step_{len(steps) + 1}_{recorded.action}"

        if recorded.action == "navigate":
            url = _parameterize_string(recorded.tool_input.get("url", ""), parameters)
            steps.append(ArtifactNavigateStep(id=step_id, url=url))

        elif recorded.action == "click":
            target = _parameterize_target(recorded.tool_input.get("target", {}), parameters)
            steps.append(ArtifactClickStep(id=step_id, target=target))

        elif recorded.action == "type":
            target = _parameterize_target(recorded.tool_input.get("target", {}), parameters)
            value = _parameterize_string(recorded.tool_input.get("value", ""), parameters)
            steps.append(ArtifactTypeStep(id=step_id, target=target, value=value))

        elif recorded.action == "extract":
            target = _parameterize_target(recorded.tool_input.get("target", {}), parameters)
            output_key = output_name_by_value.get(str(recorded.result_data))
            if output_key is None:
                fallback_output_counter += 1
                output_key = f"output_{fallback_output_counter}"
            steps.append(ArtifactExtractStep(id=step_id, target=target, output=output_key))
            outputs[output_key] = ArtifactOutputSpec(type="string")

        else:
            raise ValueError(
                f"Discovery step used action '{recorded.action}', which has "
                f"no artifact-step equivalent. This should not happen for a "
                f"successful run produced by DiscoveryAgent."
            )

    inputs = {
        param_name: ArtifactInputSpec(type=input_types.get(param_name, "string"))
        for param_name in parameters
    }

    unmatched_outputs = set(state.outputs) - set(outputs)
    if unmatched_outputs:
        warnings.warn(
            f"finish_task declared output(s) {sorted(unmatched_outputs)} that no "
            f"`extract` step in this discovery run actually produced -- the model "
            f"likely read the value directly from page text instead of calling "
            f"`extract` on it. This artifact will NOT return {sorted(unmatched_outputs)} "
            f"on replay. Re-run discovery with a prompt that requires an explicit "
            f"`extract` call for every value the goal asks for.",
            stacklevel=2,
        )
    
    return Artifact(
        artifact_id=artifact_id,
        name=name,
        description=description,
        surface=ArtifactSurfaceInfo(application=application),
        inputs=inputs,
        outputs=outputs,
        steps=steps,
        checkpoint=_derive_checkpoint(state),
        created_at=datetime.now(timezone.utc).isoformat(),
        source="discovery",
    )


def _parameterize_string(value: str, parameters: dict[str, str]) -> str:
    if not isinstance(value, str):
        return value
    for param_name, concrete_value in parameters.items():
        if concrete_value and concrete_value in value:
            value = value.replace(concrete_value, f"{{{{{param_name}}}}}")
    return value


def _parameterize_target(raw_target: dict, parameters: dict[str, str]) -> Target:
    parameterized = dict(raw_target)
    for field in _TARGET_STRING_FIELDS:
        if isinstance(parameterized.get(field), str):
            parameterized[field] = _parameterize_string(parameterized[field], parameters)
    return Target(**parameterized)


def _derive_checkpoint(state: DiscoveryState) -> ArtifactCheckpoint:
    """Default: assert the page title reached by the last recorded step.
    This is deliberately simple -- see REPORT.md for richer checkpoint
    strategies (multiple conditions, per-step assertions)."""
    for recorded in reversed(state.steps):
        if recorded.observation_after and recorded.observation_after.get("title"):
            return ArtifactCheckpoint(
                type="text_present", value=recorded.observation_after["title"]
            )
    # Fall back to a generic "page loaded" checkpoint if no step captured a title.
    return ArtifactCheckpoint(type="status_code", value="200")
