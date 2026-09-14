"""
The artifact schema -- the central design of this project.

An Artifact is NOT the LLM transcript and NOT a dump of DiscoveryState. It is
a typed, versioned, parameterized "automation recipe": ordered steps, each
identifying its target the same tiered way `surface.observation.Target`
already does, with named/typed inputs, named/typed outputs, and an explicit
checkpoint. Both a human reviewer and the replay engine (Phase 6) read this
same structure -- nothing about it depends on how it was produced (discovery
today; hand-authored or edited tomorrow are both valid).

`ArtifactStep` mirrors the discriminated-union pattern from
`tools/definitions.py` on purpose: the same idea (the `action` field decides
which fields are required) shows up at the tool-call layer and the artifact
layer for the same reason -- reject a malformed shape immediately, with a
clear message, rather than discovering the problem mid-replay.

Deliberately excluded from artifact steps: `observe_page` and `screenshot`.
Both are discovery-time bookkeeping / evidence capture, not actions that
change application state -- replaying them would add nothing but noise and
fragility (an extra screenshot mid-replay is not part of "the workflow").
"""
from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

from surface.observation import Target


class ArtifactInputSpec(BaseModel):
    type: Literal["string", "number", "boolean"] = "string"
    required: bool = True
    description: Optional[str] = None


class ArtifactOutputSpec(BaseModel):
    type: Literal["string", "number", "boolean"] = "string"
    description: Optional[str] = None


class ArtifactSurfaceInfo(BaseModel):
    type: Literal["browser"] = "browser"
    application: str
    version: str = "1.0"


class ArtifactCheckpoint(BaseModel):
    """The condition that must hold at the end of a successful replay.
    Kept intentionally simple (three strategies) -- see REPORT.md for how
    this could grow (multiple checkpoints, per-step checkpoints) without
    changing the shape of a step."""

    type: Literal["text_present", "url_contains", "status_code"]
    value: str


# --------------------------------------------------------------- step shapes


class ArtifactClickStep(BaseModel):
    id: str
    action: Literal["click"] = "click"
    target: Target


class ArtifactTypeStep(BaseModel):
    id: str
    action: Literal["type"] = "type"
    target: Target
    value: str  # may contain a {{parameter}} placeholder


class ArtifactNavigateStep(BaseModel):
    id: str
    action: Literal["navigate"] = "navigate"
    url: str  # may contain a {{parameter}} placeholder


class ArtifactExtractStep(BaseModel):
    id: str
    action: Literal["extract"] = "extract"
    target: Target
    output: str  # key into Artifact.outputs


ArtifactStep = Annotated[
    Union[ArtifactClickStep, ArtifactTypeStep, ArtifactNavigateStep, ArtifactExtractStep],
    Field(discriminator="action"),
]


class Artifact(BaseModel):
    artifact_version: str = "1.0"
    artifact_id: str
    name: str
    description: str = ""

    surface: ArtifactSurfaceInfo
    inputs: dict[str, ArtifactInputSpec] = Field(default_factory=dict)
    outputs: dict[str, ArtifactOutputSpec] = Field(default_factory=dict)
    steps: list[ArtifactStep]
    checkpoint: ArtifactCheckpoint

    created_at: Optional[str] = None
    source: Literal["discovery", "manual"] = "discovery"

    @model_validator(mode="after")
    def _outputs_referenced_by_steps_must_be_declared(self) -> "Artifact":
        referenced = {s.output for s in self.steps if isinstance(s, ArtifactExtractStep)}
        missing = referenced - set(self.outputs)
        if missing:
            raise ValueError(
                f"Step(s) reference output(s) {sorted(missing)} not declared in `outputs`."
            )
        return self

    @model_validator(mode="after")
    def _at_least_one_step(self) -> "Artifact":
        if not self.steps:
            raise ValueError("Artifact must have at least one step.")
        return self
