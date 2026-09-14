"""
The `ComputerSurface` protocol is the entire boundary between "the agent"
and "however we actually drive this particular application."

The agent (and, later, the replay engine) only ever calls these six
methods. Nothing above this line knows that Playwright, or a browser, or
even a screen exists. This is what lets us say, credibly, that the same
artifact/replay engine could target a legacy web app (different adapter,
e.g. Selenium against framesets) or a desktop app (an OS accessibility-tree
adapter) without changing the agent or the artifact schema -- only a new
class implementing this Protocol.
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from surface.observation import ActionResult, Observation, Target


@runtime_checkable
class ComputerSurface(Protocol):
    def observe(self) -> Observation:
        """Return a semantic snapshot of the current state."""
        ...

    def click(self, target: Target) -> ActionResult:
        """Click the control identified by `target`."""
        ...

    def type(self, target: Target, value: str) -> ActionResult:
        """Type `value` into the control identified by `target`."""
        ...

    def navigate(self, url: str) -> ActionResult:
        """Navigate directly to `url` (e.g. to open the target application)."""
        ...

    def extract(self, target: Target) -> ActionResult:
        """Read a value/text from the element identified by `target`.
        Result is returned in `ActionResult.data`."""
        ...

    def screenshot(self, path: Optional[str] = None) -> str:
        """Capture a screenshot, return the path it was saved to."""
        ...

    def close(self) -> None:
        """Release any underlying resources (browser, session, etc.)."""
        ...
