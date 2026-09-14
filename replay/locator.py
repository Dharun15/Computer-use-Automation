"""
Template substitution for replay -- the inverse of what
`artifacts/recorder.py` does at recording time.

Recording:  "12345"        -> "{{member_id}}"   (literal -> placeholder)
Replay:     "{{member_id}}" -> "67890"           (placeholder -> real input)

Kept in its own module (rather than inlined in the engine) because this is
also where a future surface adapter's target-resolution quirks would live
if they ever needed replay-time awareness of *which* input produced a given
target -- for now it's pure string substitution, deliberately simple.
"""
from __future__ import annotations

import re
from typing import Any

from surface.observation import Target

_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")

_TARGET_STRING_FIELDS = ("role", "name", "label", "text", "css")


def find_placeholders(text: str) -> set[str]:
    return set(_PLACEHOLDER_RE.findall(text)) if isinstance(text, str) else set()


def substitute_string(text: str, inputs: dict[str, Any]) -> str:
    if not isinstance(text, str):
        return text

    def _replace(match: re.Match) -> str:
        name = match.group(1)
        if name not in inputs:
            raise ValueError(
                f"Artifact step references input '{{{{{name}}}}}' which was "
                f"not supplied (or not declared in the artifact's `inputs`)."
            )
        return str(inputs[name])

    return _PLACEHOLDER_RE.sub(_replace, text)


def substitute_target(target: Target, inputs: dict[str, Any]) -> Target:
    data = target.model_dump()
    for field in _TARGET_STRING_FIELDS:
        if isinstance(data.get(field), str):
            data[field] = substitute_string(data[field], inputs)
    return Target(**data)
