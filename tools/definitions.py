"""
Structured tool definitions for the discovery agent.

The LLM never emits Python code and never touches Playwright. It only ever
produces a JSON object shaped like one of the `*Call` models below. Every
such object is validated with Pydantic *before* anything is executed
(see tools/executor.py) -- an invalid tool call is a validation error the
agent gets back as feedback, never a crash and never something that reaches
the surface layer unchecked.

`ToolCall` is a Pydantic discriminated union: the `action` field decides
which concrete model (and therefore which required fields) applies. This
means "click with no target" or "type with no value" are rejected before
we ever ask Playwright to do anything.

`ANTHROPIC_TOOL_SPECS` mirrors the same six actions in the JSON-schema
shape Anthropic's Messages API expects for tool-calling, so the discovery
agent (Phase 4) can hand these straight to the model.
"""
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from surface.observation import Target

# ---------------------------------------------------------------------------
# Individual tool call schemas
# ---------------------------------------------------------------------------


class ObserveCall(BaseModel):
    """Look at the current page: URL, title, controls, visible text."""

    action: Literal["observe_page"] = "observe_page"


class ClickCall(BaseModel):
    """Click a control identified by `target`."""

    action: Literal["click"] = "click"
    target: Target


class TypeCall(BaseModel):
    """Type `value` into the control identified by `target`."""

    action: Literal["type"] = "type"
    target: Target
    value: str


class NavigateCall(BaseModel):
    """Navigate directly to `url` (absolute, or relative to the app's base URL)."""

    action: Literal["navigate"] = "navigate"
    url: str


class ExtractCall(BaseModel):
    """Read text/value from the element identified by `target`."""

    action: Literal["extract"] = "extract"
    target: Target


class ScreenshotCall(BaseModel):
    """Capture a screenshot of the current page."""

    action: Literal["screenshot"] = "screenshot"


# Discriminated union: pydantic picks the right model based on `action`.
ToolCall = Annotated[
    Union[ObserveCall, ClickCall, TypeCall, NavigateCall, ExtractCall, ScreenshotCall],
    Field(discriminator="action"),
]


class ToolCallEnvelope(BaseModel):
    """Wraps a ToolCall so we can validate a raw dict against the whole
    union in one call: `ToolCallEnvelope(call=raw_dict)`."""

    call: ToolCall


# Registry used by the executor to know which model validates which action name.
TOOL_MODELS: dict[str, type[BaseModel]] = {
    "observe_page": ObserveCall,
    "click": ClickCall,
    "type": TypeCall,
    "navigate": NavigateCall,
    "extract": ExtractCall,
    "screenshot": ScreenshotCall,
}

# ---------------------------------------------------------------------------
# Anthropic Messages API tool specs (for the Phase 4 discovery agent)
# ---------------------------------------------------------------------------

_TARGET_SCHEMA = {
    "type": "object",
    "description": (
        "How to identify the control. Provide exactly one strategy, in "
        "order of preference: (1) role+name, (2) label, (3) text, (4) css. "
        "Add `nth` only if you already know multiple elements will match "
        "and you want a specific one (0-indexed)."
    ),
    "properties": {
        "role": {
            "type": "string",
            "description": "Accessibility role, e.g. 'button', 'textbox', 'link'. Must be paired with `name`.",
        },
        "name": {
            "type": "string",
            "description": "Accessible name to pair with `role`.",
        },
        "label": {
            "type": "string",
            "description": "Associated label text (works even without a real <label> element).",
        },
        "text": {
            "type": "string",
            "description": "Exact visible text to match.",
        },
        "css": {
            "type": "string",
            "description": "CSS selector, last resort only.",
        },
        "exact": {
            "type": "boolean",
            "description": "Whether label/name/text matches must be exact. Defaults to true.",
        },
        "nth": {
            "type": "integer",
            "description": "0-indexed disambiguator, only when multiple matches are expected.",
        },
    },
}

ANTHROPIC_TOOL_SPECS: list[dict] = [
    {
        "name": "observe_page",
        "description": (
            "Look at the current page: URL, title, the list of controls "
            "(role/name/value/enabled), and a summary of other visible "
            "text. Call this whenever you need to see the current state "
            "before deciding what to do next."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "click",
        "description": "Click a control on the current page.",
        "input_schema": {
            "type": "object",
            "properties": {"target": _TARGET_SCHEMA},
            "required": ["target"],
        },
    },
    {
        "name": "type",
        "description": "Type text into a control on the current page (replaces existing content).",
        "input_schema": {
            "type": "object",
            "properties": {
                "target": _TARGET_SCHEMA,
                "value": {"type": "string", "description": "Text to type."},
            },
            "required": ["target", "value"],
        },
    },
    {
        "name": "navigate",
        "description": "Navigate directly to a URL (absolute, or a path relative to the app's base URL).",
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
    {
        "name": "extract",
        "description": (
            "Read the text/value of an element on the current page (e.g. a "
            "balance figure). CRITICAL: never target by the exact value you "
            "expect to read (e.g. `text: \"$4250.00\"`) -- that value is "
            "specific to this one record and this recording will later be "
            "replayed against DIFFERENT records where that exact text will "
            "not exist, so the extraction would fail every time. Instead, "
            "target by the value's stable LABEL or heading (e.g. `label: "
            "\"Savings Balance\"`), or its role, so the same extraction "
            "works regardless of what the actual value turns out to be."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"target": _TARGET_SCHEMA},
            "required": ["target"],
        },
    },
    {
        "name": "screenshot",
        "description": "Capture a screenshot of the current page state.",
        "input_schema": {"type": "object", "properties": {}},
    },
]
