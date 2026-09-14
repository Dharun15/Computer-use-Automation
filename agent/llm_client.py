"""
Model client abstraction.

`ModelClient` is a Protocol with a single method: given the running message
history, the available tools, and a system prompt, return the model's next
turn as a list of normalized content blocks. The discovery agent only ever
depends on this Protocol -- tests use a scripted fake (see tests/fakes.py),
production uses `AnthropicModelClient` below. Neither the agent loop nor
its tests need to change if we ever swap providers.

`FINISH_TOOL_SPEC` is intentionally defined here, not in tools/definitions.py:
it is NOT a surface action (it never touches the browser), it's how the
model signals "the goal is done, here's the outcome" to the discovery loop
itself. The ToolExecutor from Phase 3 has no idea this tool exists.
"""
from __future__ import annotations

import os
from typing import Optional, Protocol

from agent.state import LLMContentBlock, LLMTextBlock, LLMToolUseBlock

FINISH_TOOL_SPEC = {
    "name": "finish_task",
    "description": (
        "Call this when the goal has been completed OR when you have "
        "determined it cannot be completed (e.g. the record does not "
        "exist, or access was denied) -- both are valid, expected outcomes. "
        "This ends the discovery run."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "success": {
                "type": "boolean",
                "description": "True if the goal was accomplished, false otherwise.",
            },
            "outputs": {
                "type": "object",
                "description": (
                    "Key-value data extracted to satisfy the goal, e.g. "
                    '{"balance": "$4250.00"}. Omit or leave empty if success=false.'
                ),
            },
            "message": {
                "type": "string",
                "description": "Short explanation of the outcome.",
            },
        },
        "required": ["success", "message"],
    },
}


class ModelClient(Protocol):
    def complete(
        self, messages: list[dict], tools: list[dict], system: str
    ) -> list[LLMContentBlock]:
        ...


class AnthropicModelClient:
    """Real LLM client. Requires `ANTHROPIC_API_KEY` in the environment
    (or passed explicitly) and the `anthropic` package installed."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-5",
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
    ):
        import anthropic  # local import: only required when actually used

        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self.model = model
        self.max_tokens = max_tokens

    def complete(
        self, messages: list[dict], tools: list[dict], system: str
    ) -> list[LLMContentBlock]:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            tools=tools,
            messages=messages,
        )
        blocks: list[LLMContentBlock] = []
        for block in response.content:
            if block.type == "text":
                blocks.append(LLMTextBlock(text=block.text))
            elif block.type == "tool_use":
                blocks.append(
                    LLMToolUseBlock(id=block.id, name=block.name, input=block.input)
                )
        return blocks


# ---------------------------------------------------------------------------
# Gemini client
#
# Included specifically because Google's Gemini API has a genuinely free,
# permanent, no-credit-card tier (the Flash models via Google AI Studio),
# unlike Anthropic's one-time trial credit. The assignment leaves "LLM
# provider / model" explicitly up to the implementer -- this is exactly the
# substitution `ModelClient` was designed to make free: nothing in
# `agent/agent.py` needed to change to add this.
#
# Implemented against the `google-genai` SDK's documented function-calling
# pattern (automatic function calling disabled -- this project runs its own
# tool-execution loop via `ToolExecutor`, so the SDK must not try to run
# tools itself). Google has changed this SDK's surface more than once
# (see their own migration guide); if a method name below has moved, only
# this class needs to change -- the `ModelClient` Protocol boundary is the
# whole point.
# ---------------------------------------------------------------------------

_JSON_TO_GEMINI_TYPE = {
    "string": "STRING",
    "number": "NUMBER",
    "integer": "INTEGER",
    "boolean": "BOOLEAN",
    "object": "OBJECT",
    "array": "ARRAY",
}


def _json_schema_to_gemini_schema(schema: dict):
    from google.genai import types

    kwargs: dict = {"type": _JSON_TO_GEMINI_TYPE.get(schema.get("type", "object"), "STRING")}
    if schema.get("description"):
        kwargs["description"] = schema["description"]
    if schema.get("type") == "object" and "properties" in schema:
        kwargs["properties"] = {
            name: _json_schema_to_gemini_schema(sub) for name, sub in schema["properties"].items()
        }
        if schema.get("required"):
            kwargs["required"] = schema["required"]
    if schema.get("type") == "array" and "items" in schema:
        kwargs["items"] = _json_schema_to_gemini_schema(schema["items"])
    return types.Schema(**kwargs)


def _build_gemini_tool(tool_specs: list[dict]):
    from google.genai import types

    declarations = [
        types.FunctionDeclaration(
            name=spec["name"],
            description=spec["description"],
            parameters=_json_schema_to_gemini_schema(spec["input_schema"]),
        )
        for spec in tool_specs
    ]
    return types.Tool(function_declarations=declarations)


def _messages_to_gemini_contents(messages: list[dict]):
    """Converts this project's Anthropic-shaped message history into
    Gemini `types.Content` objects.

    Two real structural mismatches between the two APIs, both handled here
    so nothing else in the project needs to know about them:

    1. Anthropic addresses a `tool_result` by an opaque `tool_use_id`;
       Gemini addresses a `function_response` by the function's NAME.
       `GeminiModelClient` resolves this by generating its own tool-call
       ids as `"{function_name}::{random_suffix}"` (see `complete()`
       below) and parsing the name back out here.

    2. Gemini 3 models require a "thought signature" to be attached to
       every function-call `Part` when history is round-tripped back to
       the model (mandatory for tool use -- see
       https://ai.google.dev/gemini-api/docs/thought-signatures). Since
       this project manages conversation history itself (rather than
       letting the SDK's automatic-function-calling own it), that
       signature has to be captured when the model first produces a tool
       call and re-attached here on every subsequent turn. It's carried
       through `LLMToolUseBlock.provider_metadata["thought_signature"]`
       (base64-encoded, since Pydantic models need JSON-safe values) --
       the Anthropic path simply never populates this field.
    """
    import base64

    from google.genai import types

    contents = []
    for message in messages:
        role = "model" if message["role"] == "assistant" else "user"
        content = message["content"]

        if isinstance(content, str):
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=content)]))
            continue

        parts = []
        for item in content:
            if item.get("type") == "text":
                parts.append(types.Part.from_text(text=item["text"]))
            elif item.get("type") == "tool_use":
                part = types.Part.from_function_call(name=item["name"], args=item["input"])
                signature = (item.get("provider_metadata") or {}).get("thought_signature")
                if signature:
                    part.thought_signature = base64.b64decode(signature)
                parts.append(part)
            elif item.get("type") == "tool_result":
                function_name = str(item["tool_use_id"]).split("::")[0]
                parts.append(
                    types.Part.from_function_response(
                        name=function_name, response={"result": item["content"]}
                    )
                )
        contents.append(types.Content(role=role, parts=parts))
    return contents


class GeminiModelClient:
    """Real LLM client using Google's Gemini API. Requires `GEMINI_API_KEY`
    in the environment (or passed explicitly) and the `google-genai`
    package installed. Uses a free-tier-eligible model by default."""

    def __init__(
        self,
        model: str = "gemini-3.6-flash",
        api_key: Optional[str] = None,
    ):
        from google import genai  # local import: only required when actually used

        self._client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))
        self.model = model

    def complete(
        self, messages: list[dict], tools: list[dict], system: str
    ) -> list[LLMContentBlock]:
        import base64
        import uuid

        from google.genai import types

        response = self._client.models.generate_content(
            model=self.model,
            contents=_messages_to_gemini_contents(messages),
            config=types.GenerateContentConfig(
                system_instruction=system,
                tools=[_build_gemini_tool(tools)],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )

        blocks: list[LLMContentBlock] = []
        candidate = response.candidates[0]
        for part in candidate.content.parts:
            if getattr(part, "text", None):
                blocks.append(LLMTextBlock(text=part.text))
            elif getattr(part, "function_call", None):
                fc = part.function_call
                call_id = f"{fc.name}::{uuid.uuid4().hex[:8]}"
                provider_metadata = None
                signature = getattr(part, "thought_signature", None)
                if signature:
                    encoded = (
                        base64.b64encode(signature).decode("ascii")
                        if isinstance(signature, (bytes, bytearray))
                        else signature
                    )
                    provider_metadata = {"thought_signature": encoded}
                blocks.append(
                    LLMToolUseBlock(
                        id=call_id,
                        name=fc.name,
                        input=dict(fc.args or {}),
                        provider_metadata=provider_metadata,
                    )
                )
        return blocks
