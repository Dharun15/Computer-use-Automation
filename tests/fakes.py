"""
A scripted, deterministic stand-in for `ModelClient`, used only in tests.

It ignores the actual message history and just returns the next
pre-programmed response each time `complete()` is called -- enough to
exercise the discovery loop's mechanics (step counting, dead-end detection,
auto-observation, finish_task handling) without a network call or an API key.

The real discovery run against a live LLM (required by the assignment) uses
`agent.llm_client.AnthropicModelClient` instead -- this fake is never used
outside `tests/`.
"""
from __future__ import annotations

from agent.state import LLMContentBlock, LLMTextBlock, LLMToolUseBlock


class ScriptedModelClient:
    def __init__(self, script: list[list[LLMContentBlock]]):
        self._script = list(script)
        self._calls = 0

    def complete(self, messages, tools, system) -> list[LLMContentBlock]:
        if self._calls >= len(self._script):
            raise AssertionError(
                f"ScriptedModelClient exhausted after {self._calls} calls "
                f"-- script only had {len(self._script)} turns."
            )
        response = self._script[self._calls]
        self._calls += 1
        return response

    @property
    def calls_made(self) -> int:
        return self._calls


def tool_use(name: str, input: dict, call_id: str | None = None) -> list[LLMContentBlock]:
    """Convenience: a single-tool-use model turn."""
    return [LLMToolUseBlock(id=call_id or f"call_{name}", name=name, input=input)]


def text_only(text: str) -> list[LLMContentBlock]:
    return [LLMTextBlock(text=text)]
