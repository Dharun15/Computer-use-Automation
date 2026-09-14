"""
Phase 4 tests: observe -> LLM decide -> act -> observe loop, against the
real fake-bank app + real Playwright surface. The LLM itself is replaced
with a deterministic script (see tests/fakes.py) so these tests are fast,
free, and reproducible -- the loop mechanics being tested here (step
counting, auto-observation, dead-end/timeout/max-step handling, finish_task)
are identical regardless of which model produces the tool calls.

The actual live-LLM discovery run required by the assignment is a separate,
one-time demo (see README.md) using agent.llm_client.AnthropicModelClient.
"""
from agent.agent import DiscoveryAgent
from agent.state import DiscoveryStatus
from tests.fakes import ScriptedModelClient, text_only, tool_use
from tools.executor import ToolExecutor


def test_full_discovery_success_for_member_12345(surface):
    executor = ToolExecutor(surface)
    script = [
        tool_use("type", {"target": {"label": "Member ID"}, "value": "12345"}),
        tool_use("click", {"target": {"role": "button", "name": "Search"}}),
        tool_use("click", {"target": {"role": "link", "name": "12345"}}),
        tool_use("extract", {"target": {"label": "Savings Balance"}}),
        tool_use(
            "finish_task",
            {
                "success": True,
                "outputs": {"balance": "$4250.00"},
                "message": "Found member 12345's savings balance.",
            },
        ),
    ]
    model = ScriptedModelClient(script)
    agent = DiscoveryAgent(executor, model, max_steps=10)

    result = agent.run(
        goal="Look up member 12345 and return their savings balance.",
        start_url="/",
    )

    assert result.state.status == DiscoveryStatus.SUCCESS
    assert result.state.outputs == {"balance": "$4250.00"}
    assert result.state.stop_reason == "model_declared_finish"
    # seed navigate + 4 real actions + 1 finish_task call = 6
    assert result.state.step_count == 6
    # the seed navigate + the 4 real actions are recorded as reusable steps
    assert len(result.state.steps) == 5
    assert [s.action for s in result.state.steps] == [
        "navigate",
        "type",
        "click",
        "click",
        "extract",
    ]
    assert result.state.steps[-1].result_data == "$4250.00"
    assert model.calls_made == 5
    # transcript exists and is clearly separate from the structured steps
    assert len(result.transcript) > 0


def test_business_outcome_member_not_found_is_a_clean_failed_finish(surface):
    executor = ToolExecutor(surface)
    script = [
        tool_use("type", {"target": {"label": "Member ID"}, "value": "99999"}),
        tool_use("click", {"target": {"role": "button", "name": "Search"}}),
        tool_use(
            "finish_task",
            {
                "success": False,
                "outputs": {},
                "message": "No record found for member 99999 -- this is an expected outcome.",
            },
        ),
    ]
    model = ScriptedModelClient(script)
    agent = DiscoveryAgent(executor, model, max_steps=10)

    result = agent.run(goal="Look up member 99999's savings balance.", start_url="/")

    assert result.state.status == DiscoveryStatus.FAILED
    assert "expected outcome" in result.state.final_message
    assert result.state.outputs == {}


def test_dead_end_stops_before_exhausting_step_budget(surface):
    executor = ToolExecutor(surface)
    failing_call = tool_use("click", {"target": {"role": "button", "name": "Nonexistent"}})
    # Provide more scripted turns than should ever be consumed.
    script = [failing_call for _ in range(10)]
    model = ScriptedModelClient(script)
    agent = DiscoveryAgent(
        executor, model, max_steps=10, max_consecutive_failures=3
    )

    result = agent.run(goal="Click a button that does not exist.", start_url="/")

    assert result.state.status == DiscoveryStatus.DEAD_END
    assert model.calls_made == 3  # stopped before a 4th model call
    click_steps = [s for s in result.state.steps if s.action == "click"]
    assert len(click_steps) == 3
    assert all(s.result_status == "not_found" for s in click_steps)


def test_max_steps_exceeded_when_model_never_finishes(surface):
    executor = ToolExecutor(surface)
    # observe_page always succeeds, so this never trips dead-end detection --
    # it should instead run out of step budget.
    script = [tool_use("observe_page", {}) for _ in range(10)]
    model = ScriptedModelClient(script)
    agent = DiscoveryAgent(
        executor, model, max_steps=4, max_consecutive_failures=100
    )

    result = agent.run(goal="Just look around forever.", start_url="/")

    assert result.state.status == DiscoveryStatus.MAX_STEPS_EXCEEDED
    assert result.state.step_count == 4
    # the seed navigate consumes 1 step before the loop's own budget check,
    # so only 3 model turns fit inside a max_steps=4 budget
    assert model.calls_made == 3


def test_text_only_response_is_nudged_not_silently_accepted(surface):
    executor = ToolExecutor(surface)
    script = [
        text_only("Let me think about this for a moment..."),
        tool_use(
            "finish_task",
            {"success": True, "outputs": {}, "message": "done thinking"},
        ),
    ]
    model = ScriptedModelClient(script)
    agent = DiscoveryAgent(executor, model, max_steps=10, max_consecutive_failures=5)

    result = agent.run(goal="Do nothing in particular.", start_url="/")

    assert result.state.status == DiscoveryStatus.SUCCESS
    assert result.state.step_count == 3  # seed navigate + 1 nudged non-action + 1 finish_task
    # the nudge message should appear in the transcript
    assert any(
        isinstance(m.get("content"), str) and "did not call a tool" in m["content"]
        for m in result.transcript
    )


def test_timeout_status_when_runtime_budget_is_effectively_zero(surface):
    executor = ToolExecutor(surface)
    script = [tool_use("observe_page", {}) for _ in range(5)]
    model = ScriptedModelClient(script)
    agent = DiscoveryAgent(
        executor, model, max_steps=100, max_runtime_seconds=0.0
    )

    result = agent.run(goal="This should time out immediately.", start_url="/")

    assert result.state.status == DiscoveryStatus.TIMEOUT
    assert model.calls_made == 0
