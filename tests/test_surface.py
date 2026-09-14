"""
Phase 2 acceptance tests: prove the ComputerSurface/PlaywrightSurface
abstraction works end-to-end against the live fake bank app.

No LLM, no artifact, no replay engine here -- purely: can we drive a real
browser through observe/click/type/navigate/extract/screenshot using only
surface-independent Target objects?
"""
from surface.base import ComputerSurface
from surface.observation import ActionStatus, Target


def test_playwright_surface_satisfies_protocol(surface):
    assert isinstance(surface, ComputerSurface)


def test_full_happy_path_lookup_flow(surface, tmp_path):
    """The canonical acceptance scenario from the Phase 2 brief:
    open -> observe -> type -> click -> observe -> extract -> screenshot."""

    nav = surface.navigate("/")
    assert nav.ok

    obs = surface.observe()
    assert obs.title == "MemberServ 3.2 - Member Search"
    control_names = [(c.role, c.name) for c in obs.controls]
    assert ("button", "Search") in control_names
    # The real textbox has no accessible name (no <label>, no placeholder) --
    # that's the intentional legacy-hostility from Phase 1. Confirm we see
    # *a* textbox control (possibly nameless) rather than asserting a name.
    assert any(c.role == "textbox" for c in obs.controls)

    typed = surface.type(Target(label="Member ID"), "12345")
    assert typed.ok, typed.message
    assert typed.target_tier_used == 2  # resolved via legacy row-label heuristic

    clicked = surface.click(Target(role="button", name="Search"))
    assert clicked.ok, clicked.message
    assert clicked.target_tier_used == 1  # accessible name works for the submit button

    obs2 = surface.observe()
    assert "Search Results" in obs2.title
    assert "12345" in obs2.text_summary

    opened = surface.click(Target(role="link", name="12345"))
    assert opened.ok, opened.message

    obs3 = surface.observe()
    assert obs3.title == "MemberServ 3.2 - Member Details"
    assert obs3.status_code == 200

    balance = surface.extract(Target(label="Savings Balance"))
    assert balance.ok, balance.message
    assert balance.data == "$4250.00"
    assert balance.target_tier_used == 2

    shot_path = surface.screenshot(str(tmp_path / "member_details.png"))
    assert (tmp_path / "member_details.png").exists()
    assert shot_path == str(tmp_path / "member_details.png")


def test_business_outcome_member_not_found(surface):
    surface.navigate("/")
    surface.type(Target(label="Member ID"), "99999")
    surface.click(Target(role="button", name="Search"))
    obs = surface.observe()
    assert "No Records" in obs.title
    assert "No records found" in obs.text_summary


def test_validation_error_bad_format(surface):
    surface.navigate("/")
    surface.type(Target(label="Member ID"), "abc")
    surface.click(Target(role="button", name="Search"))
    obs = surface.observe()
    assert "Input Error" in obs.title
    assert "Invalid Member ID" in obs.text_summary


def test_permission_denied_status_code(surface):
    result = surface.navigate("/members/55555")
    assert result.ok
    obs = surface.observe()
    assert obs.status_code == 403
    assert "Access Denied" in obs.text_summary


def test_app_error_status_code(surface):
    surface.navigate("/members/00000")
    obs = surface.observe()
    assert obs.status_code == 500
    assert "Application Error" in obs.text_summary


def test_confirm_dialog_is_captured_and_auto_dismissed(surface):
    surface.navigate("/members/77777")
    obs = surface.observe()
    assert obs.dialog is not None
    assert obs.dialog.dialog_type == "confirm"
    assert obs.dialog.auto_dismissed is True
    # Data stays hidden because the dialog was dismissed (safe default).
    balance = surface.extract(Target(label="Savings Balance"))
    assert balance.status == ActionStatus.NOT_FOUND

    # dialog is reported once, not repeated on the next observation
    obs2 = surface.observe()
    assert obs2.dialog is None


def test_slow_load_member_completes_within_default_timeout(surface):
    result = surface.navigate("/members/66666")
    assert result.ok
    obs = surface.observe()
    assert obs.status_code == 200
    assert "Slow Loader" in obs.text_summary


def test_ambiguous_css_fallback_without_nth_is_reported_not_guessed(surface):
    surface.navigate("/")
    # Both the disabled decoy field and the real field match this coarse
    # CSS fallback -- the surface must report ambiguity rather than
    # silently acting on whichever one Playwright happens to find first.
    result = surface.click(Target(css="input[type=text]"))
    assert result.status == ActionStatus.AMBIGUOUS

    # Adding `nth` makes it a controlled, explicit choice instead of a guess.
    disambiguated = surface.type(Target(css="input[type=text]", nth=1), "12345")
    assert disambiguated.ok, disambiguated.message


def test_target_requires_at_least_one_strategy():
    import pytest as _pytest

    with _pytest.raises(Exception):
        Target()


def test_not_found_target_reports_clean_error(surface):
    surface.navigate("/")
    result = surface.click(Target(role="button", name="This Button Does Not Exist"))
    assert result.status == ActionStatus.NOT_FOUND
    assert "No element matched" in result.message
