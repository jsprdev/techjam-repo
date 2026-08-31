"""Live LLM semantic reranking. Spec 6.5, Pillar I.

This is the stage the brief actually describes: a model that READS THE
CONVERSATION and reorders the shortlist for this customer, rather than a fixed
per-product opinion looked up from a file.

    "Rerank the top candidates against the current slot state and the user's
     phrasing. The reranker sees a shortlist, not the catalog."   spec 6.5

Off by default, and that is not a hedge. `docs/submission_rules.md` warns that
official scoring may run with network access disabled, so a model on the turn
path is a way to score zero. The spec anticipated exactly this:

    "If the reranker is unavailable, the deterministic ordering from Layer 1
     ships instead."   spec 6.5

So the deterministic ordering is the product and this is an enhancement that
must never be load-bearing. Every failure path, no key, timeout, malformed
reply, refusal, returns the input order unchanged.

THE LAYER BOUNDARY HOLDS. This module proposes an ordering. It does not hold
the belief and it does not decide whether to commit. `src/rank/baseline.py`
blends the proposal into its own score rather than replacing it, so a confident
model cannot override a confident belief.

Enable with `Config.use_llm = True` and an ANTHROPIC_API_KEY in the environment.
Never commit a key.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Sequence

# The reranker only ever sees this many candidates. Latency and cost scale with
# it, and Feasibility is 15% of judging with token usage a required disclosure.
MAX_CANDIDATES = 20
# Product text sent per candidate. Enough to judge relevance, not enough to
# blow up the prompt.
CHARS_PER_CANDIDATE = 180

SYSTEM = """You rank clothing products for a shopper, given what they have said.

You receive the shopper's stated requirements and a numbered candidate list.
Return the candidate numbers reordered, best match first.

Judge only how well each product satisfies what the shopper actually said.
Ignore brand prestige and popularity; those are handled elsewhere. Every
candidate must appear exactly once in your answer."""

ORDER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["order"],
    "properties": {
        "order": {
            "type": "array",
            "items": {"type": "integer", "minimum": 1},
            "description": "Candidate numbers, best first, each exactly once",
        }
    },
}


@dataclass(frozen=True)
class RerankResult:
    """What the reranker produced, and whether it actually ran."""

    order: list[str]
    used_llm: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reason: str = ""


def available() -> bool:
    """True when a live call could plausibly succeed. Never raises."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def _prompt(constraints: Sequence[str], candidates: Sequence[tuple[str, str]]) -> str:
    said = "\n".join(f"- {c}" for c in constraints) or "- (nothing specific yet)"
    listing = "\n".join(
        f"{n}. {text[:CHARS_PER_CANDIDATE]}" for n, (_, text) in enumerate(candidates, 1)
    )
    return f"The shopper has said:\n{said}\n\nCandidates:\n{listing}"


def rerank(
    constraints: Sequence[str],
    candidates: Sequence[tuple[str, str]],
    config: Any,
    client: Any = None,
) -> RerankResult:
    """Reorder `candidates` (parent_asin, text) by fit to what the shopper said.

    Returns the input order unchanged on ANY failure. `client` is injectable so
    the failure paths can be tested without a network or a key.
    """
    original = [asin for asin, _ in candidates]

    if not getattr(config, "use_llm", False):
        return RerankResult(original, used_llm=False, reason="disabled")
    if not candidates:
        return RerankResult(original, used_llm=False, reason="no candidates")

    shortlist = list(candidates)[:MAX_CANDIDATES]
    tail = original[len(shortlist) :]

    if client is None:
        if not available():
            return RerankResult(original, used_llm=False, reason="no api key")
        import anthropic

        client = anthropic.Anthropic(timeout=config.llm_timeout_seconds, max_retries=1)

    try:
        response = client.messages.create(
            model=getattr(config, "llm_model", "claude-opus-5"),
            max_tokens=1024,
            system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
            output_config={"format": {"type": "json_schema", "schema": ORDER_SCHEMA}},
            messages=[{"role": "user", "content": _prompt(constraints, shortlist)}],
        )
    except Exception as error:  # noqa: BLE001 - a graded turn must never die here
        return RerankResult(original, used_llm=False, reason=f"call failed: {type(error).__name__}")

    # A safety refusal returns HTTP 200 with stop_reason "refusal", so check it
    # before reading content.
    if getattr(response, "stop_reason", None) == "refusal":
        return RerankResult(original, used_llm=False, reason="refused")

    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        order = json.loads(text)["order"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return RerankResult(original, used_llm=False, reason="unparseable reply")
    if not isinstance(order, list):
        # A non-list would fall through the loop below and silently rebuild the
        # input order while still reporting used_llm, which would overstate the
        # stage in the usage disclosure.
        return RerankResult(original, used_llm=False, reason="order was not a list")

    # Trust nothing: the model may drop, duplicate or invent indices. Take the
    # valid ones in the order given, then append anything it forgot.
    seen: set[int] = set()
    reordered: list[str] = []
    for value in order:
        if not isinstance(value, int) or not 1 <= value <= len(shortlist) or value in seen:
            continue
        seen.add(value)
        reordered.append(shortlist[value - 1][0])
    for position, (asin, _) in enumerate(shortlist, 1):
        if position not in seen:
            reordered.append(asin)

    usage = getattr(response, "usage", None)
    return RerankResult(
        reordered + tail,
        used_llm=True,
        prompt_tokens=getattr(usage, "input_tokens", 0) or 0,
        completion_tokens=getattr(usage, "output_tokens", 0) or 0,
    )
