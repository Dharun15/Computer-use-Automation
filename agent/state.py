"""
Discovery state.

Deliberately split into two things that never get conflated:

  - `DiscoveryState.steps` -- a structured, typed log of *actions actually
    executed against the surface* (what tool, what input, what result).
    This is the only thing Phase 5's artifact recorder is allowed to read.

  - The raw LLM message transcript (system prompt, assistant turns, tool
    results) -- useful for debugging/evidence, but it is NOT the artifact
    and Phase 5 must not derive the artifact from it. `DiscoveryResult`
    carries both, but as clearly separate fields, specifically so nothing
    downstream can quietly treat the transcript as the source of truth.
"""
from __future__ import annotations

import time
from enum import Enum
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field


# --------------------------------------------------------------------- LLM IO
# Normalized content blocks so the discovery loop never depends directly on
# the Anthropic SDK's response types -- a fake/scripted client used in tests
# produces the exact same shapes a real model client does.


class LLMTextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class LLMToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict


LLMContentBlock = Union[LLMTextBlock, LLMToolUseBlock]


# ------------------------------------------------------------ discovery state


class DiscoveryStatus(str, Enum):
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"  # model explicitly declared the goal unreachable
    MAX_STEPS_EXCEEDED = "max_steps_exceeded"
    TIMEOUT = "timeout"
    DEAD_END = "dead_end"  # too many consecutive action failures


class RecordedStep(BaseModel):
    """One executed tool call and its outcome. This -- not the transcript --
    is what Phase 5 turns into artifact steps."""

    step_index: int
    action: str
    tool_input: dict
    result_status: Optional[str] = None
    result_message: str = ""
    result_data: Optional[Any] = None
    observation_after: Optional[dict] = None  # compact snapshot: url/title/status_code


class DiscoveryState(BaseModel):
    goal: str
    start_url: str
    status: DiscoveryStatus = DiscoveryStatus.RUNNING
    step_count: int = 0
    consecutive_failures: int = 0
    steps: list[RecordedStep] = Field(default_factory=list)
    outputs: dict = Field(default_factory=dict)
    final_message: str = ""
    stop_reason: Optional[str] = None
    started_at: float = Field(default_factory=time.time)
    finished_at: Optional[float] = None

    @property
    def succeeded(self) -> bool:
        return self.status == DiscoveryStatus.SUCCESS


class DiscoveryResult(BaseModel):
    """What `DiscoveryAgent.run()` returns. `state` is the structured
    record; `transcript` is the full raw LLM conversation, kept for
    debugging/evidence only."""

    state: DiscoveryState
    transcript: list[dict] = Field(default_factory=list)
    

class LLMToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict
    provider_metadata: Optional[dict] = None 
