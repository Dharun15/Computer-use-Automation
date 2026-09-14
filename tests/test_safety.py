"""
Phase 8 tests: SafetyPolicy / PolicyEngine in isolation, plus its
integration into ToolExecutor.

The fake bank has no destructive actions to demonstrate risk classification
against for real, so the risky-action tests use synthetic Target objects
(a hypothetical "Delete Account" button) -- the policy engine's logic is
identical either way, since it only ever looks at target name/label/text/
value strings, never at what application it's running against.
"""
import pytest

from safety.policy import PolicyDecision, PolicyEngine, SafetyPolicy
from tools.definitions import ClickCall, NavigateCall, TypeCall
from tools.executor import ToolExecutor
from surface.observation import Target


@pytest.fixture()
def engine():
    return PolicyEngine(
        SafetyPolicy(allowed_domains=["localhost", "127.0.0.1"], blocked_path_prefixes=["/admin"])
    )


# ------------------------------------------------------------- domain/path


def test_relative_navigation_is_always_allowed(engine):
    call = NavigateCall(url="/members/12345")
    result = engine.check("navigate", call)
    assert result.decision == PolicyDecision.ALLOW


def test_relative_navigation_to_blocked_prefix_is_blocked(engine):
    call = NavigateCall(url="/admin/danger")
    result = engine.check("navigate", call)
    assert result.decision == PolicyDecision.BLOCK
    assert "blocked prefix" in result.reason


def test_absolute_navigation_to_allowed_domain_is_allowed(engine):
    call = NavigateCall(url="http://localhost:8000/members/12345")
    result = engine.check("navigate", call)
    assert result.decision == PolicyDecision.ALLOW


def test_absolute_navigation_to_disallowed_domain_is_blocked(engine):
    call = NavigateCall(url="https://evil-example.com/steal-data")
    result = engine.check("navigate", call)
    assert result.decision == PolicyDecision.BLOCK
    assert "evil-example.com" in result.reason


def test_subdomain_of_allowed_domain_is_allowed():
    engine = PolicyEngine(SafetyPolicy(allowed_domains=["example.com"]))
    call = NavigateCall(url="https://app.example.com/page")
    assert engine.check("navigate", call).decision == PolicyDecision.ALLOW


def test_absolute_navigation_to_allowed_domain_but_blocked_path_is_blocked(engine):
    call = NavigateCall(url="http://localhost:8000/admin/danger")
    result = engine.check("navigate", call)
    assert result.decision == PolicyDecision.BLOCK


def test_empty_domain_allowlist_fails_closed():
    engine = PolicyEngine(SafetyPolicy(allowed_domains=[]))
    call = NavigateCall(url="http://localhost:8000/")
    assert engine.check("navigate", call).decision == PolicyDecision.BLOCK


# ------------------------------------------------------------- action gate


def test_action_not_in_allowlist_is_blocked():
    policy = SafetyPolicy(allowed_actions=["observe_page", "click"])  # navigate excluded
    engine = PolicyEngine(policy)
    call = NavigateCall(url="/")
    result = engine.check("navigate", call)
    assert result.decision == PolicyDecision.BLOCK
    assert "not in the allowed action list" in result.reason


# --------------------------------------------------------------- risky targets


def test_click_on_ordinary_button_is_allowed(engine):
    call = ClickCall(target=Target(role="button", name="Search"))
    result = engine.check("click", call)
    assert result.decision == PolicyDecision.ALLOW


def test_click_on_risky_named_button_requires_confirmation(engine):
    call = ClickCall(target=Target(role="button", name="Delete Account"))
    result = engine.check("click", call)
    assert result.decision == PolicyDecision.REQUIRE_CONFIRMATION
    assert "delete" in result.reason.lower()


def test_click_risk_check_is_case_insensitive(engine):
    call = ClickCall(target=Target(role="button", name="APPROVE TRANSFER"))
    result = engine.check("click", call)
    assert result.decision == PolicyDecision.REQUIRE_CONFIRMATION


def test_type_with_risky_value_requires_confirmation(engine):
    # Even an innocuous-looking field can carry a risky *value* -- the
    # policy checks the typed value too, not just the target's own name.
    call = TypeCall(target=Target(label="Action"), value="please transfer funds now")
    result = engine.check("type", call)
    assert result.decision == PolicyDecision.REQUIRE_CONFIRMATION


def test_custom_risky_patterns_are_used_instead_of_defaults():
    policy = SafetyPolicy(
        allowed_domains=["localhost"], risky_name_patterns=["launch missiles"]
    )
    engine = PolicyEngine(policy)
    # A pattern that IS in the default list but NOT in this custom list:
    ordinary_now = engine.check("click", ClickCall(target=Target(role="button", name="Delete")))
    assert ordinary_now.decision == PolicyDecision.ALLOW

    risky_now = engine.check(
        "click", ClickCall(target=Target(role="button", name="Launch Missiles"))
    )
    assert risky_now.decision == PolicyDecision.REQUIRE_CONFIRMATION


# --------------------------------------------------- ToolExecutor integration


@pytest.fixture()
def guarded_executor(surface):
    policy = PolicyEngine(SafetyPolicy(allowed_domains=["127.0.0.1"]))
    return ToolExecutor(surface, policy=policy)


def test_executor_blocks_navigation_outside_allowlist(guarded_executor):
    result = guarded_executor.execute("navigate", {"url": "https://not-allowed.example.com/"})
    assert result.valid is True
    assert result.status == "blocked"
    assert not result.ok


def test_executor_allows_relative_navigation_within_app(guarded_executor):
    result = guarded_executor.execute("navigate", {"url": "/"})
    assert result.ok


def test_executor_requires_confirmation_for_risky_click_then_allows_when_confirmed(
    guarded_executor,
):
    guarded_executor.execute("navigate", {"url": "/"})
    risky_call = {"target": {"role": "button", "name": "Delete Account"}}

    unconfirmed = guarded_executor.execute("click", risky_call)
    assert unconfirmed.status == "requires_confirmation"
    assert not unconfirmed.ok

    # Without a matching button on the real page this will now fail at the
    # surface level (not_found) -- but the important thing is it was
    # actually ATTEMPTED once confirmed=True was passed, proving the gate
    # opens rather than staying permanently shut.
    confirmed = guarded_executor.execute("click", risky_call, confirmed=True)
    assert confirmed.status != "requires_confirmation"


def test_executor_with_no_policy_behaves_exactly_as_before(surface):
    """No PolicyEngine supplied at all -- Phases 1-7 behavior, unchanged."""
    executor = ToolExecutor(surface)
    result = executor.execute("navigate", {"url": "https://anything.example.com/"})
    # No policy configured means no domain check happens at this layer;
    # whether the surface can actually reach it is a separate question.
    assert result.status != "blocked"
