"""Response construction and contract validation.

Owned by role 4 because role 4 also owns the contract test, and the two need to
agree. Role 3 builds turn content; this module guarantees the envelope is legal.

Every rule below comes from `docs/agent_api_contract.json` turn_response, which
sets `additionalProperties: false` on the response, on each recommendation, and
on `usage`. An extra key is a contract violation even though the local evaluator
happens to tolerate it, so we validate against the contract, not the evaluator.
"""

from __future__ import annotations

import math
from typing import Any

from src.interfaces import ASK_ATTRIBUTES


def build(
    message: str,
    ask_attribute: str | None,
    recommendations: list[str] | list[tuple[str, float]],
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> dict[str, Any]:
    """Assemble a contract-legal turn response.

    Accepts either bare parent_asin strings or (parent_asin, score) pairs.
    Anything illegal is coerced rather than raised, because a malformed response
    costs a whole session and a crash costs the same. Fail soft, then let the
    contract test catch the bug offline.
    """
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in recommendations:
        # A (asin, score) pair survives a JSON round trip as a two element
        # LIST, not a tuple, so checking only for tuple would silently take
        # entry[0] of the list, str() the whole thing, and emit garbage.
        if isinstance(entry, (tuple, list)) and len(entry) == 2:
            parent_asin = entry[0]
            try:
                score = float(entry[1])
            except (TypeError, ValueError):
                score = None
            # NaN and Infinity serialise as bare NaN/Infinity, which is not
            # valid JSON. Drop the score rather than emit an unparseable body.
            if score is not None and not math.isfinite(score):
                score = None
        else:
            parent_asin, score = entry, None
        parent_asin = str(parent_asin).strip()
        if not parent_asin or parent_asin in seen:
            continue
        seen.add(parent_asin)
        item: dict[str, Any] = {"parent_asin": parent_asin}
        if score is not None:
            item["score"] = score
        items.append(item)
        # The contract caps recommendations at 100. Only the first 10 valid
        # unique values are scored, so anything past that is dead weight.
        if len(items) >= 100:
            break

    return {
        "message": str(message),
        "ask_attribute": ask_attribute if ask_attribute in ASK_ATTRIBUTES else None,
        "recommendations": items,
        "usage": {
            "prompt_tokens": max(0, int(prompt_tokens)),
            "completion_tokens": max(0, int(completion_tokens)),
        },
    }


def violations(response: Any) -> list[str]:
    """Return every way `response` breaks the contract. Empty means legal."""
    problems: list[str] = []
    if not isinstance(response, dict):
        return [f"response must be a dict, got {type(response).__name__}"]

    allowed = {"message", "ask_attribute", "recommendations", "usage"}
    extra = set(response) - allowed
    if extra:
        problems.append(f"additionalProperties: {sorted(extra)}")
    for required in ("message", "ask_attribute", "recommendations"):
        if required not in response:
            problems.append(f"missing required field: {required}")

    if not isinstance(response.get("message"), str):
        problems.append("message must be a string")

    attribute = response.get("ask_attribute")
    if attribute is not None:
        try:
            legal = attribute in ASK_ATTRIBUTES
        except TypeError:
            # An unhashable value (a list or dict) would raise on the set
            # lookup. That is a violation to report, not an exception to throw
            # out of a validator.
            legal = False
        if not legal:
            problems.append(f"ask_attribute not in enum: {attribute!r}")

    items = response.get("recommendations")
    if not isinstance(items, list):
        problems.append("recommendations must be a list")
    else:
        if len(items) > 100:
            problems.append(f"recommendations exceeds maxItems 100: {len(items)}")
        for position, item in enumerate(items):
            if not isinstance(item, dict):
                problems.append(f"recommendations[{position}] must be an object")
                continue
            item_extra = set(item) - {"parent_asin", "score"}
            if item_extra:
                problems.append(f"recommendations[{position}] extra keys: {sorted(item_extra)}")
            asin = item.get("parent_asin")
            if not isinstance(asin, str) or not asin:
                problems.append(f"recommendations[{position}].parent_asin must be a non-empty string")
            if "score" in item:
                score = item["score"]
                # bool subclasses int in Python, but JSON Schema treats true and
                # false as a distinct type from number.
                if isinstance(score, bool) or not isinstance(score, (int, float)):
                    problems.append(f"recommendations[{position}].score must be a number")
                elif not math.isfinite(score):
                    problems.append(
                        f"recommendations[{position}].score is {score}, which is not valid JSON"
                    )

    # `usage` is optional, so an absent key is legal. But a key that is present
    # and null is not: the contract types it as an object. Testing membership
    # rather than `is not None` is what separates the two cases.
    if "usage" in response:
        usage = response["usage"]
        if not isinstance(usage, dict):
            problems.append("usage must be an object")
        else:
            usage_extra = set(usage) - {"prompt_tokens", "completion_tokens"}
            if usage_extra:
                problems.append(f"usage extra keys: {sorted(usage_extra)}")
            for key in ("prompt_tokens", "completion_tokens"):
                if key not in usage:
                    problems.append(f"usage missing {key}")
                elif not isinstance(usage[key], int) or isinstance(usage[key], bool):
                    problems.append(f"usage.{key} must be an integer")
                elif usage[key] < 0:
                    problems.append(f"usage.{key} must be non-negative")
    return problems
