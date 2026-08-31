"""The offline LLM artefact, loaded once and read deterministically at run time.

This is the runtime half of the LLM semantic ranking stage the brief names in
Pillar I. The model half runs in `offline/build_semantic_prior.py`, once, before
submission. What ships is a JSON lookup.

The split exists because the two requirements conflict: the brief asks for LLM
semantic ranking, and `docs/submission_rules.md` warns that official scoring may
run with network access disabled. Moving the model call off the turn path
satisfies both. Nothing here opens a socket, and a missing or malformed artefact
degrades to no signal rather than an error, so the agent still runs if the file
was never generated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "artifacts/semantic_prior.json"


@dataclass(frozen=True)
class SemanticPrior:
    """Per-product judgments an LLM made offline. Empty is a valid state."""

    appeal: dict[str, float]
    use_case: dict[str, frozenset[str]]
    formality: dict[str, str]

    def __len__(self) -> int:
        return len(self.appeal)

    def appeal_of(self, parent_asin: str, default: float = 0.0) -> float:
        """Judged mainstream desirability, or `default` when unenriched.

        The default matters: a partial artefact must not penalise the products
        it does not cover, or coverage becomes a ranking signal in itself.
        Callers pass the mean appeal of what IS covered so an unjudged product
        sits neutrally among judged ones.
        """
        return self.appeal.get(parent_asin, default)


EMPTY = SemanticPrior(appeal={}, use_case={}, formality={})


def load(path: str | Path = DEFAULT_PATH) -> SemanticPrior:
    """Load the artefact. Never raises: no artefact means no signal, not a crash."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return EMPTY
    if not isinstance(raw, dict):
        return EMPTY

    appeal: dict[str, float] = {}
    use_case: dict[str, frozenset[str]] = {}
    formality: dict[str, str] = {}
    for parent_asin, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        value = entry.get("appeal")
        if isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0:
            appeal[str(parent_asin)] = float(value)
        tags = entry.get("use_case")
        if isinstance(tags, list):
            use_case[str(parent_asin)] = frozenset(str(t).lower() for t in tags)
        level = entry.get("formality")
        if isinstance(level, str):
            formality[str(parent_asin)] = level
    return SemanticPrior(appeal=appeal, use_case=use_case, formality=formality)
