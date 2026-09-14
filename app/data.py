"""
Synthetic in-memory data for the fake "legacy" credit-union back-office app.

No real PII. All member records below are fabricated for demo/testing purposes.

Special member IDs are wired to specific failure modes so the automation
system (discovery + replay) has real, reproducible exceptional states to
handle -- not just the happy path. See README.md / REPORT.md for how each
of these is used in the demo.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Member:
    member_id: str
    name: str
    savings_balance: float
    checking_balance: float
    # Behavioral flags used to simulate runtime conditions on this member's detail page.
    slow_load: bool = False          # artificial delay before the page renders
    permission_denied: bool = False  # simulates an authz failure for this record
    confirm_dialog: bool = False     # page fires a native JS confirm() on load
    app_error: bool = False          # simulates an unhandled server-side error


MEMBERS: dict[str, Member] = {
    "12345": Member(
        member_id="12345",
        name="John Smith",
        savings_balance=4250.00,
        checking_balance=1200.00,
    ),
    "23456": Member(
        member_id="23456",
        name="Maria Garcia",
        savings_balance=980.50,
        checking_balance=3400.10,
    ),
    "34567": Member(
        member_id="34567",
        name="Wei Chen",
        savings_balance=15200.00,
        checking_balance=560.25,
    ),
    # --- Failure-mode members ---
    "55555": Member(
        member_id="55555",
        name="Restricted Account",
        savings_balance=0.0,
        checking_balance=0.0,
        permission_denied=True,
    ),
    "66666": Member(
        member_id="66666",
        name="Slow Loader",
        savings_balance=750.00,
        checking_balance=100.00,
        slow_load=True,
    ),
    "77777": Member(
        member_id="77777",
        name="Dialog Dan",
        savings_balance=2200.00,
        checking_balance=300.00,
        confirm_dialog=True,
    ),
    "00000": Member(
        member_id="00000",
        name="Broken Record",
        savings_balance=0.0,
        checking_balance=0.0,
        app_error=True,
    ),
    # "99999" is intentionally absent -> MEMBER_NOT_FOUND business outcome.
}


def find_member(member_id: str) -> Member | None:
    return MEMBERS.get(member_id.strip())


def is_valid_member_id_format(member_id: str) -> bool:
    """Legacy validation rule: member IDs are exactly 5 digits."""
    return member_id.strip().isdigit() and len(member_id.strip()) == 5
