"""
ToolExecutor -- sits between "the LLM emitted a tool call" and "the surface
actually did something."

Every call goes through two hard gates before it touches a browser:
    1. Is `action` one we know about at all?
    2. Does the input validate against that action's Pydantic model?

Both failures are returned as structured, non-throwing results (`valid=False`
with a `validation_error` message) -- the discovery agent loop (Phase 4) is
expected to feed that message straight back to the LLM as tool feedback,
the same way a real tool-use API would. Nothing about a malformed tool call
should ever crash the process.

This layer deliberately does NOT know about recording steps into a reusable
artifact (that's Phase 5). It DOES enforce safety policy (Phase 8), if a
`PolicyEngine` was supplied -- that check happens after input validation
(a malformed call is rejected on its own merits first) but strictly before
anything touches the surface.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ValidationError

from safety.policy import PolicyDecision, PolicyEngine
from surface.base import ComputerSurface
from tools.definitions import TOOL_MODELS, ToolCallEnvelope


class ToolExecutionResult(BaseModel):
    """JSON-serializable outcome of a single tool call, valid or not."""

    action: str
    valid: bool
    validation_error: Optional[str] = None
    status: Optional[str] = None  # "ok"/"not_found"/"ambiguous"/"timeout"/"error"/"blocked"/"requires_confirmation"
    message: str = ""
    data: Optional[Any] = None
    observation: Optional[dict] = None
    screenshot_path: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.valid and self.status == "ok"


class ToolExecutor:
    def __init__(self, surface: ComputerSurface, policy: Optional[PolicyEngine] = None):
        self._surface = surface
        self._policy = policy

    def execute(
        self, action: str, raw_input: Optional[dict] = None, confirmed: bool = False
    ) -> ToolExecutionResult:
        raw_input = dict(raw_input or {})

        model_cls = TOOL_MODELS.get(action)
        if model_cls is None:
            return ToolExecutionResult(
                action=action,
                valid=False,
                validation_error=(
                    f"Unknown action '{action}'. Valid actions: "
                    f"{', '.join(sorted(TOOL_MODELS))}."
                ),
            )

        try:
            # Validate through the full discriminated union so a bad/missing
            # `action` field inside raw_input is also caught cleanly.
            envelope = ToolCallEnvelope(call={"action": action, **raw_input})
            call = envelope.call
        except ValidationError as e:
            return ToolExecutionResult(
                action=action,
                valid=False,
                validation_error=_flatten_validation_error(e),
            )

        if self._policy is not None:
            policy_result = self._policy.check(action, call)
            if policy_result.decision == PolicyDecision.BLOCK:
                return ToolExecutionResult(
                    action=action, valid=True, status="blocked", message=policy_result.reason
                )
            if policy_result.decision == PolicyDecision.REQUIRE_CONFIRMATION and not confirmed:
                return ToolExecutionResult(
                    action=action,
                    valid=True,
                    status="requires_confirmation",
                    message=policy_result.reason,
                )

        try:
            return self._dispatch(action, call)
        except Exception as e:  # noqa: BLE001 - last line of defense before the agent loop
            return ToolExecutionResult(
                action=action,
                valid=True,
                status="error",
                message=f"Unexpected error executing '{action}': {e}",
            )

    # ------------------------------------------------------------- dispatch

    def _dispatch(self, action: str, call) -> ToolExecutionResult:
        if action == "observe_page":
            obs = self._surface.observe()
            return ToolExecutionResult(
                action=action,
                valid=True,
                status="ok",
                message=obs.to_prompt_text(),
                observation=obs.model_dump(),
            )

        if action == "click":
            result = self._surface.click(call.target)
            return ToolExecutionResult(
                action=action, valid=True, status=result.status.value, message=result.message
            )

        if action == "type":
            result = self._surface.type(call.target, call.value)
            return ToolExecutionResult(
                action=action, valid=True, status=result.status.value, message=result.message
            )

        if action == "navigate":
            result = self._surface.navigate(call.url)
            return ToolExecutionResult(
                action=action, valid=True, status=result.status.value, message=result.message
            )

        if action == "extract":
            result = self._surface.extract(call.target)
            return ToolExecutionResult(
                action=action,
                valid=True,
                status=result.status.value,
                message=result.message,
                data=result.data,
            )

        if action == "screenshot":
            path = self._surface.screenshot()
            return ToolExecutionResult(
                action=action,
                valid=True,
                status="ok",
                message=f"Screenshot saved to {path}",
                screenshot_path=path,
            )

        # Unreachable: `action` was already validated against TOOL_MODELS.
        raise AssertionError(f"no dispatch handler for validated action '{action}'")


def _flatten_validation_error(e: ValidationError) -> str:
    """Render a Pydantic ValidationError as a short, LLM-readable message
    rather than the full multi-line default repr."""
    parts = []
    for err in e.errors():
        loc = ".".join(str(p) for p in err["loc"] if p != "call")
        parts.append(f"{loc or '(root)'}: {err['msg']}")
    return "; ".join(parts)
