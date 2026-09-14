"""
DiscoveryAgent -- the genuine LLM-driven observe/decide/act loop.

    observe -> LLM decision -> tool call -> execute -> observe -> ...

The loop terminates one of five ways, all recorded in `DiscoveryState.status`:
  - SUCCESS / FAILED      -- the model called finish_task
  - MAX_STEPS_EXCEEDED    -- hit the step budget before finishing
  - TIMEOUT               -- hit the wall-clock budget
  - DEAD_END              -- too many consecutive action failures in a row

This file has never heard of Playwright. It depends on `ToolExecutor`
(Phase 3) and, through it, whatever `ComputerSurface` was constructed with.
It also depends on `ModelClient` (a Protocol) rather than the Anthropic SDK
directly -- production wiring passes `AnthropicModelClient`; tests pass a
scripted fake. Swapping either dependency requires no change here.
"""
from __future__ import annotations

import time
from typing import Optional

from agent.llm_client import FINISH_TOOL_SPEC, ModelClient
from agent.prompts import SYSTEM_PROMPT
from agent.state import (
    DiscoveryResult,
    DiscoveryState,
    DiscoveryStatus,
    LLMToolUseBlock,
    RecordedStep,
)
from tools.definitions import ANTHROPIC_TOOL_SPECS
from tools.executor import ToolExecutor

# Statuses from ToolExecutor that count as "the action failed" for the
# purposes of dead-end detection. Ambiguous/not_found/timeout/error are all
# real problems; a blocked or unconfirmed-risky action also counts -- the
# model should not be able to spin forever retrying something the safety
# policy will never let it do unattended.
_FAILURE_STATUSES = {"not_found", "ambiguous", "timeout", "error", "blocked", "requires_confirmation"}

_AGENT_TOOLS = ANTHROPIC_TOOL_SPECS + [FINISH_TOOL_SPEC]


class DiscoveryAgent:
    def __init__(
        self,
        executor: ToolExecutor,
        model_client: ModelClient,
        max_steps: int = 15,
        max_runtime_seconds: float = 180.0,
        max_consecutive_failures: int = 3,
    ):
        self._executor = executor
        self._model = model_client
        self.max_steps = max_steps
        self.max_runtime_seconds = max_runtime_seconds
        self.max_consecutive_failures = max_consecutive_failures

    def run(self, goal: str, start_url: str) -> DiscoveryResult:
        state = DiscoveryState(goal=goal, start_url=start_url)
        transcript: list[dict] = []

        # Seed the run with a real navigation + observation rather than
        # spending the model's first turn just asking it to "go to the app".
        # This IS recorded as a real step (not just loop bookkeeping) --
        # Phase 5's artifact recorder needs to know where a replay of this
        # capability should start, and that only exists here.
        nav_result = self._executor.execute("navigate", {"url": start_url})
        state.step_count += 1
        obs_result = self._executor.execute("observe_page", {})
        seed_observation_after = None
        if obs_result.observation is not None:
            seed_observation_after = {
                "url": obs_result.observation["url"],
                "title": obs_result.observation["title"],
                "status_code": obs_result.observation["status_code"],
            }
        state.steps.append(
            RecordedStep(
                step_index=state.step_count,
                action="navigate",
                tool_input={"url": start_url},
                result_status=nav_result.status,
                result_message=nav_result.message,
                observation_after=seed_observation_after,
            )
        )

        initial_text = (
            f"Goal: {goal}\n\n"
            f"You have been navigated to the starting page. Current state:\n\n"
            f"{obs_result.message}"
        )
        messages: list[dict] = [{"role": "user", "content": initial_text}]

        while True:
            if state.step_count >= self.max_steps:
                state.status = DiscoveryStatus.MAX_STEPS_EXCEEDED
                state.stop_reason = f"Reached max_steps={self.max_steps} without finishing."
                break
            if time.time() - state.started_at > self.max_runtime_seconds:
                state.status = DiscoveryStatus.TIMEOUT
                state.stop_reason = (
                    f"Exceeded max_runtime_seconds={self.max_runtime_seconds}."
                )
                break
            if state.consecutive_failures >= self.max_consecutive_failures:
                state.status = DiscoveryStatus.DEAD_END
                state.stop_reason = (
                    f"{state.consecutive_failures} consecutive action failures "
                    f"-- stopping rather than wandering indefinitely."
                )
                break

            blocks = self._model.complete(
                messages=messages, tools=_AGENT_TOOLS, system=SYSTEM_PROMPT
            )
            assistant_content = [b.model_dump() for b in blocks]
            messages.append({"role": "assistant", "content": assistant_content})
            transcript.append({"role": "assistant", "content": assistant_content})

            tool_use_blocks = [b for b in blocks if isinstance(b, LLMToolUseBlock)]

            if not tool_use_blocks:
                # The model produced only text -- nudge it rather than spin
                # silently. This still counts as a step and a failure toward
                # dead-end detection, since no progress was made.
                state.step_count += 1
                state.consecutive_failures += 1
                nudge = (
                    "You did not call a tool. Please call one of: observe_page, "
                    "click, type, navigate, extract, screenshot, or finish_task."
                )
                messages.append({"role": "user", "content": nudge})
                transcript.append({"role": "user", "content": nudge})
                continue

            finished = self._handle_tool_calls(tool_use_blocks, state, messages, transcript)
            if finished:
                break

        state.finished_at = time.time()
        return DiscoveryResult(state=state, transcript=transcript)

    # ------------------------------------------------------------- helpers

    def _handle_tool_calls(
        self,
        tool_use_blocks: list[LLMToolUseBlock],
        state: DiscoveryState,
        messages: list[dict],
        transcript: list[dict],
    ) -> bool:
        """Executes each tool call in this turn. Returns True if the run
        should stop (finish_task was called)."""
        tool_result_content: list[dict] = []

        for block in tool_use_blocks:
            state.step_count += 1

            if block.name == "finish_task":
                success = bool(block.input.get("success"))
                state.status = DiscoveryStatus.SUCCESS if success else DiscoveryStatus.FAILED
                state.outputs = dict(block.input.get("outputs") or {})
                state.final_message = str(block.input.get("message", ""))
                state.stop_reason = "model_declared_finish"
                return True

            result = self._executor.execute(block.name, block.input)

            failed = (not result.valid) or (result.status in _FAILURE_STATUSES)
            state.consecutive_failures = 0 if not failed else state.consecutive_failures + 1

            observation_after = None
            content_text = result.validation_error or result.message

            if result.valid and block.name not in ("observe_page",):
                # Auto-attach fresh state after real actions so the model
                # doesn't have to spend an extra step re-observing.
                fresh = self._executor.execute("observe_page", {})
                if fresh.observation is not None:
                    observation_after = {
                        "url": fresh.observation["url"],
                        "title": fresh.observation["title"],
                        "status_code": fresh.observation["status_code"],
                    }
                    content_text = f"{content_text}\n\n[Updated page state]\n{fresh.message}"
            elif result.valid and block.name == "observe_page" and result.observation:
                observation_after = {
                    "url": result.observation["url"],
                    "title": result.observation["title"],
                    "status_code": result.observation["status_code"],
                }

            state.steps.append(
                RecordedStep(
                    step_index=state.step_count,
                    action=block.name,
                    tool_input=block.input,
                    result_status=result.status if result.valid else "invalid",
                    result_message=result.validation_error or result.message,
                    result_data=result.data,
                    observation_after=observation_after,
                )
            )

            tool_result_content.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": content_text,
                    **({"is_error": True} if failed else {}),
                }
            )

        messages.append({"role": "user", "content": tool_result_content})
        transcript.append({"role": "user", "content": tool_result_content})
        return False
