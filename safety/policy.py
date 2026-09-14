"""
Safety & policy guardrails.

Every action -- from either the LLM during discovery or a saved artifact
during replay -- passes through this before it ever reaches the surface
(see the `policy` parameter on `tools.executor.ToolExecutor`).

Two independent gates:

  1. Allowlist: which domains/paths the agent is permitted to navigate to,
     and which of the six action types are permitted at all. Fail-closed --
     an unconfigured/empty domain allowlist means nothing is reachable by
     absolute URL (relative paths, which by construction stay within
     whatever app the surface was already pointed at, are always allowed).

  2. Risk classification: "safe/reversible" vs "risky/irreversible" is
     decided by matching the target's visible name/label/text/value (and,
     for `type`, the value being typed) against a configurable list of
     keyword patterns (e.g. "delete", "transfer", "approve"). This fake
     bank's real UI has no destructive actions to demonstrate against, so
     this is exercised directly with synthetic targets in tests -- see
     REPORT.md Section 6 for why keyword matching was chosen over, say, a
     hardcoded allowlist of exact button names (it generalizes across
     tenants running visually different but similarly-worded UIs, at the
     cost of both false positives and false negatives, which is why it
     defaults to REQUIRE_CONFIRMATION rather than a silent ALLOW/BLOCK).

Chosen default for risky actions: REQUIRE_CONFIRMATION, not an outright
BLOCK. An unattended discovery/replay run cannot itself supply that
confirmation (see `ToolExecutor.execute`'s `confirmed` parameter) -- so in
practice an un-confirmed risky action behaves exactly like a block unless
something upstream (a human, via the handoff mechanism in Phase 9) explicitly
approves it. This was preferred over a hardcoded BLOCK because a real
deployment will have some risky actions that legitimate, supervised
workflows must still be able to perform -- outright blocking would make the
system unable to ever support them, even with a human in the loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

_DEFAULT_ALLOWED_ACTIONS = [
    "observe_page",
    "click",
    "type",
    "navigate",
    "extract",
    "screenshot",
]

_DEFAULT_RISKY_NAME_PATTERNS = [
    "delete",
    "remove",
    "close account",
    "terminate",
    "submit transfer",
    "transfer funds",
    "approve",
    "withdraw",
    "confirm transaction",
]


class PolicyDecision(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    REQUIRE_CONFIRMATION = "require_confirmation"


@dataclass
class PolicyResult:
    decision: PolicyDecision
    reason: str

    @property
    def allowed(self) -> bool:
        return self.decision == PolicyDecision.ALLOW


class SafetyPolicy(BaseModel):
    allowed_domains: list[str] = Field(default_factory=list)
    blocked_path_prefixes: list[str] = Field(default_factory=list)
    allowed_actions: list[str] = Field(default_factory=lambda: list(_DEFAULT_ALLOWED_ACTIONS))
    risky_name_patterns: list[str] = Field(
        default_factory=lambda: list(_DEFAULT_RISKY_NAME_PATTERNS)
    )


class PolicyEngine:
    def __init__(self, policy: SafetyPolicy):
        self.policy = policy

    def check(self, action: str, call) -> PolicyResult:
        if action not in self.policy.allowed_actions:
            return PolicyResult(
                PolicyDecision.BLOCK, f"Action '{action}' is not in the allowed action list."
            )

        if action == "navigate":
            return self._check_url(call.url)

        if action in ("click", "type"):
            target = call.target
            texts = [target.role, target.name, target.label, target.text]
            if action == "type":
                texts.append(getattr(call, "value", None))
            risk = self._matches_risky_pattern(texts)
            if risk:
                return PolicyResult(
                    PolicyDecision.REQUIRE_CONFIRMATION,
                    f"Target/value matches risky pattern {risk!r} -- requires human confirmation.",
                )

        return PolicyResult(PolicyDecision.ALLOW, "OK")

    def _check_url(self, url: str) -> PolicyResult:
        if url.startswith("/"):
            # Relative to whatever app the surface is already scoped to --
            # allowed by construction (the surface itself, not this policy,
            # is what decides which application is being automated).
            for prefix in self.policy.blocked_path_prefixes:
                if url.startswith(prefix):
                    return PolicyResult(
                        PolicyDecision.BLOCK, f"Path {url!r} matches blocked prefix {prefix!r}."
                    )
            return PolicyResult(PolicyDecision.ALLOW, "OK (relative path)")

        hostname = urlparse(url).hostname or ""
        for allowed in self.policy.allowed_domains:
            if hostname == allowed or hostname.endswith("." + allowed):
                for prefix in self.policy.blocked_path_prefixes:
                    if urlparse(url).path.startswith(prefix):
                        return PolicyResult(
                            PolicyDecision.BLOCK,
                            f"Path {urlparse(url).path!r} matches blocked prefix {prefix!r}.",
                        )
                return PolicyResult(PolicyDecision.ALLOW, "OK")

        return PolicyResult(
            PolicyDecision.BLOCK,
            f"Domain {hostname!r} is not in the allowed domain list {self.policy.allowed_domains}.",
        )

    def _matches_risky_pattern(self, texts: list[Optional[str]]) -> Optional[str]:
        haystack = " ".join(t for t in texts if t).lower()
        for pattern in self.policy.risky_name_patterns:
            if pattern.lower() in haystack:
                return pattern
        return None
