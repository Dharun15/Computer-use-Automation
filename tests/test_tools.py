"""
Phase 3 acceptance tests: LLM-shaped tool call (raw dict) -> Pydantic
validation -> ToolExecutor -> PlaywrightSurface -> real browser.

Still no LLM here -- we hand-write the raw dicts an LLM would eventually
produce, to prove the validation + dispatch path works before wiring an
actual model into the loop (Phase 4).
"""
import pytest

from tools.executor import ToolExecutor


@pytest.fixture()
def executor(surface):
    return ToolExecutor(surface)


# --------------------------------------------------------------------- happy


def test_full_lookup_flow_via_tool_calls(executor):
    """Same acceptance scenario as Phase 2, but driven entirely through
    the tool-call interface an LLM would actually use."""

    nav = executor.execute("navigate", {"url": "/"})
    assert nav.valid and nav.status == "ok"

    obs = executor.execute("observe_page", {})
    assert obs.valid and obs.status == "ok"
    assert obs.observation is not None
    assert any(c["name"] == "Search" for c in obs.observation["controls"])

    typed = executor.execute(
        "type", {"target": {"label": "Member ID"}, "value": "12345"}
    )
    assert typed.valid and typed.status == "ok", typed.message

    clicked = executor.execute(
        "click", {"target": {"role": "button", "name": "Search"}}
    )
    assert clicked.valid and clicked.status == "ok", clicked.message

    opened = executor.execute(
        "click", {"target": {"role": "link", "name": "12345"}}
    )
    assert opened.valid and opened.status == "ok", opened.message

    balance = executor.execute(
        "extract", {"target": {"label": "Savings Balance"}}
    )
    assert balance.valid and balance.status == "ok", balance.message
    assert balance.data == "$4250.00"

    shot = executor.execute("screenshot", {})
    assert shot.valid and shot.status == "ok"
    assert shot.screenshot_path


# ---------------------------------------------------------- validation gate


def test_unknown_action_is_rejected_before_touching_surface(executor):
    result = executor.execute("delete_everything", {})
    assert result.valid is False
    assert "Unknown action" in result.validation_error
    assert "click" in result.validation_error  # lists valid actions


def test_type_missing_value_is_rejected(executor):
    result = executor.execute("type", {"target": {"role": "button", "name": "Search"}})
    assert result.valid is False
    assert "value" in result.validation_error


def test_click_missing_target_is_rejected(executor):
    result = executor.execute("click", {})
    assert result.valid is False
    assert "target" in result.validation_error


def test_target_with_no_strategy_is_rejected(executor):
    result = executor.execute("click", {"target": {}})
    assert result.valid is False
    # message should be informative, not a raw stack trace
    assert "target" in result.validation_error.lower()


def test_navigate_missing_url_is_rejected(executor):
    result = executor.execute("navigate", {})
    assert result.valid is False
    assert "url" in result.validation_error


# ------------------------------------------------------- surface-level errors passthrough


def test_not_found_target_surfaces_as_status(executor):
    executor.execute("navigate", {"url": "/"})
    result = executor.execute(
        "click", {"target": {"role": "button", "name": "Does Not Exist"}}
    )
    assert result.valid is True  # input itself was well-formed
    assert result.status == "not_found"


def test_ambiguous_target_surfaces_as_status(executor):
    executor.execute("navigate", {"url": "/"})
    result = executor.execute("click", {"target": {"css": "input[type=text]"}})
    assert result.valid is True
    assert result.status == "ambiguous"


def test_business_outcome_page_reachable_via_tool_calls(executor):
    executor.execute("navigate", {"url": "/"})
    executor.execute("type", {"target": {"label": "Member ID"}, "value": "99999"})
    executor.execute("click", {"target": {"role": "button", "name": "Search"}})
    obs = executor.execute("observe_page", {})
    assert "No Records" in obs.observation["title"]
