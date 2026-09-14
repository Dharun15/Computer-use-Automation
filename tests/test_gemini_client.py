"""
Tests for the Gemini-provider conversion logic in `agent/llm_client.py`.

These test only the pure, non-network parts: JSON-schema -> Gemini Schema
conversion, and message-history -> Gemini Content conversion (including
the tool_use_id -> function_name resolution this provider needs). No
GEMINI_API_KEY or network call is required or made anywhere in this file.
`GeminiModelClient.complete()` itself (the part that actually calls the
API) is intentionally not exercised here -- there is nothing to verify
without a real key and a real network call.
"""
import os

os.environ.setdefault("GEMINI_API_KEY", "fake-key-for-tests-only")

from agent.llm_client import (
    FINISH_TOOL_SPEC,
    GeminiModelClient,
    _build_gemini_tool,
    _json_schema_to_gemini_schema,
    _messages_to_gemini_contents,
)
from tools.definitions import ANTHROPIC_TOOL_SPECS


def test_gemini_client_constructs_without_a_network_call():
    client = GeminiModelClient(model="gemini-2.5-flash")
    assert client.model == "gemini-2.5-flash"


def test_json_schema_object_with_nested_properties_converts():
    schema = _json_schema_to_gemini_schema(
        {
            "type": "object",
            "properties": {
                "target": {
                    "type": "object",
                    "properties": {"role": {"type": "string"}, "name": {"type": "string"}},
                },
                "value": {"type": "string"},
            },
            "required": ["target", "value"],
        }
    )
    assert schema.type.name == "OBJECT"
    assert set(schema.properties.keys()) == {"target", "value"}
    assert schema.required == ["target", "value"]
    assert schema.properties["target"].type.name == "OBJECT"
    assert set(schema.properties["target"].properties.keys()) == {"role", "name"}


def test_json_schema_empty_object_converts_cleanly():
    schema = _json_schema_to_gemini_schema({"type": "object", "properties": {}})
    assert schema.type.name == "OBJECT"


def test_all_six_surface_tools_plus_finish_task_build_into_one_gemini_tool():
    tool = _build_gemini_tool(ANTHROPIC_TOOL_SPECS + [FINISH_TOOL_SPEC])
    names = {d.name for d in tool.function_declarations}
    assert names == {
        "observe_page",
        "click",
        "type",
        "navigate",
        "extract",
        "screenshot",
        "finish_task",
    }


def test_plain_string_user_message_converts_to_a_single_text_part():
    contents = _messages_to_gemini_contents([{"role": "user", "content": "Goal: do the thing"}])
    assert len(contents) == 1
    assert contents[0].role == "user"
    assert contents[0].parts[0].text == "Goal: do the thing"


def test_assistant_role_maps_to_gemini_model_role():
    contents = _messages_to_gemini_contents(
        [{"role": "assistant", "content": [{"type": "text", "text": "thinking..."}]}]
    )
    assert contents[0].role == "model"


def test_tool_use_block_converts_to_function_call_part():
    contents = _messages_to_gemini_contents(
        [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "click::deadbeef",
                        "name": "click",
                        "input": {"target": {"role": "button", "name": "Search"}},
                    }
                ],
            }
        ]
    )
    part = contents[0].parts[0]
    assert part.function_call.name == "click"
    assert part.function_call.args["target"]["name"] == "Search"


def test_tool_result_resolves_function_name_from_embedded_id():
    """The core structural adaptation this provider needs: Anthropic
    addresses a tool_result by an opaque id; Gemini needs the function's
    actual NAME. GeminiModelClient embeds the name in the id it generates
    (`"click::<random>"`) specifically so this conversion can recover it."""
    contents = _messages_to_gemini_contents(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "extract::deadbeef",
                        "content": "$4250.00",
                    }
                ],
            }
        ]
    )
    part = contents[0].parts[0]
    assert part.function_response.name == "extract"
    assert part.function_response.response == {"result": "$4250.00"}


def test_full_conversation_round_trip_shape():
    """The exact shape agent.py actually produces across a few turns."""
    messages = [
        {"role": "user", "content": "Goal: look up member 12345."},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "type::aaa",
                    "name": "type",
                    "input": {"target": {"label": "Member ID"}, "value": "12345"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "type::aaa", "content": "Typed."}
            ],
        },
    ]
    contents = _messages_to_gemini_contents(messages)
    assert [c.role for c in contents] == ["user", "model", "user"]
