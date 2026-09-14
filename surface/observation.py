"""
Surface-independent data model.

Nothing in this file knows about Playwright, browsers, or HTML. This is the
vocabulary the agent (and later, the artifact/replay engine) speaks. A
different surface adapter (legacy web via a different automation lib, a
desktop accessibility adapter, etc.) would produce/consume the exact same
shapes.

Locator strategy (see Target): we deliberately support a *tiered* hierarchy
rather than a single generic "selector" string, because the tier used is
itself meaningful -- it tells a human reviewer (and the replay engine's
error messages) *why* a target should still resolve next month:

    1. role + accessible name   -- most robust, matches how a screen reader
                                    or a human operator identifies a control
    2. label                    -- resolved via an explicit <label>, or (for
                                    legacy table-based forms with no <label>)
                                    a "nearest preceding table-cell text"
                                    heuristic implemented in the surface
    3. text                     -- exact visible text match (links, buttons,
                                    or read-only table cells whose only
                                    identity is their text)
    4. css                      -- a stable-attribute fallback (e.g.
                                    `input[name=member_id]`), used only when
                                    tiers 1-3 cannot identify the control
    5. nth                      -- optional disambiguator when a resolved
                                    locator legitimately matches >1 element
                                    (e.g. "the 2nd row") -- controlled, never
                                    a silent guess
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


class Target(BaseModel):
    """Identifies a single UI control or piece of text, surface-independently."""

    role: Optional[str] = Field(
        default=None,
        description="Accessibility role, e.g. 'textbox', 'button', 'link'.",
    )
    name: Optional[str] = Field(
        default=None,
        description="Accessible name to pair with `role` (tier 1).",
    )
    label: Optional[str] = Field(
        default=None,
        description="Associated label text, resolved via <label> or a "
        "legacy row/table-cell heuristic (tier 2).",
    )
    text: Optional[str] = Field(
        default=None,
        description="Exact visible text to match (tier 3).",
    )
    css: Optional[str] = Field(
        default=None,
        description="Stable-attribute CSS selector fallback (tier 4).",
    )
    exact: bool = Field(
        default=True,
        description="Whether name/text/label matches must be exact.",
    )
    nth: Optional[int] = Field(
        default=None,
        description="0-indexed disambiguator when multiple matches are "
        "legitimately expected (tier 5, controlled fallback only).",
    )

    @model_validator(mode="after")
    def _at_least_one_strategy(self) -> "Target":
        if not any([self.role, self.label, self.text, self.css]):
            raise ValueError(
                "Target must specify at least one of: role(+name), label, text, css"
            )
        if self.role and not self.name:
            raise ValueError("Target with `role` must also specify `name`")
        return self

    def strategy_tier(self) -> int:
        """Which locator tier this target primarily relies on (1=best)."""
        if self.role and self.name:
            return 1
        if self.label:
            return 2
        if self.text:
            return 3
        if self.css:
            return 4
        return 99

    def describe(self) -> str:
        parts = []
        if self.role:
            parts.append(f"role={self.role!r} name={self.name!r}")
        if self.label:
            parts.append(f"label={self.label!r}")
        if self.text:
            parts.append(f"text={self.text!r}")
        if self.css:
            parts.append(f"css={self.css!r}")
        if self.nth is not None:
            parts.append(f"nth={self.nth}")
        return " ".join(parts) or "<empty target>"


class Control(BaseModel):
    """A single interactive (or informational) element found on the page."""

    role: str
    name: str
    value: Optional[str] = None
    enabled: bool = True


class DialogInfo(BaseModel):
    dialog_type: str
    message: str
    auto_dismissed: bool = True


class Observation(BaseModel):
    """
    What the agent 'sees'. Deliberately NOT raw HTML -- this is a compact,
    semantic summary an LLM (or a human) can reason over directly.
    """

    url: str
    title: str
    status_code: Optional[int] = None
    controls: list[Control] = Field(default_factory=list)
    text_summary: str = Field(
        default="",
        description="Other visible text on the page not already captured "
        "as a control name (headings, messages, table data).",
    )
    dialog: Optional[DialogInfo] = None
    screenshot_path: Optional[str] = None

    def to_prompt_text(self) -> str:
        """Render as compact text for an LLM prompt."""
        lines = [f"URL: {self.url}", f"Title: {self.title}"]
        if self.status_code and self.status_code >= 400:
            lines.append(f"HTTP Status: {self.status_code} (error page)")
        if self.dialog:
            lines.append(
                f"[Dialog occurred: {self.dialog.dialog_type} - "
                f"\"{self.dialog.message}\" "
                f"({'auto-dismissed' if self.dialog.auto_dismissed else 'pending'})]"
            )
        lines.append("")
        lines.append("Controls:")
        if self.controls:
            for c in self.controls:
                state = "" if c.enabled else " [disabled]"
                val = f" = {c.value!r}" if c.value else ""
                lines.append(f"  - {c.role}: {c.name!r}{val}{state}")
        else:
            lines.append("  (none detected)")
        if self.text_summary:
            lines.append("")
            lines.append("Other visible text:")
            lines.append(self.text_summary)
        return "\n".join(lines)


class ActionStatus(str, Enum):
    OK = "ok"
    NOT_FOUND = "not_found"
    AMBIGUOUS = "ambiguous"
    TIMEOUT = "timeout"
    ERROR = "error"


class ActionResult(BaseModel):
    """Result of a single surface action (click/type/navigate/extract)."""

    status: ActionStatus
    message: str = ""
    data: Optional[Any] = None  # e.g. extracted text for extract()
    target_tier_used: Optional[int] = None

    @property
    def ok(self) -> bool:
        return self.status == ActionStatus.OK
