"""Every tunable in one place.

Owned by role 4 so that the sweep harness has a single object to permute. If you
find yourself typing a bare number inside a module, it belongs here instead.
Each field carries a note on what moving it actually does, because a constant
whose effect nobody remembers is a constant nobody dares tune.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any


@dataclass(frozen=True)
class Config:
    # ---- retrieval, role 1 -------------------------------------------------
    # Candidate pool width per turn. Narrow sharpens precision on the Buying
    # track, wide preserves recall for open ended Browsing. Phase 0 measured
    # 100% recall at depth 1000, so going wider than that buys nothing.
    truncate_buying: int = 200
    truncate_browsing: int = 800

    # Field weights for the lexical index. Title and features carry the phrases
    # the simulated customer actually repeats, so they dominate. Raising
    # description weight pulls in long boilerplate and tends to hurt.
    weight_title: float = 3.0
    weight_features: float = 2.5
    weight_categories: float = 2.0
    weight_description: float = 1.0
    weight_store: float = 1.5

    # Multiplier applied when a whole disclosed phrase appears verbatim in a
    # product's text. The customer quotes the target's own record, so exact
    # phrase agreement is the single strongest signal available. Set to 1.0 to
    # disable and measure the difference.
    exact_phrase_boost: float = 4.0

    # ---- dialogue, role 2 --------------------------------------------------
    # Weight decay applied to a constraint per turn since it was last
    # reinforced. 1.0 disables decay. Lower values let a turn eight statement
    # outrank a turn one inference, which matters for Intent Override.
    slot_decay: float = 0.9

    # Entropy above which the belief counts as too flat to commit, triggering
    # the over-generality cutoff and a clarifying question instead of a weak
    # list. Raise it to ask less often.
    flat_belief_entropy: float = 0.85

    # Whether to fall back to the catch-all `other` attribute once specific
    # asks stop yielding new constraints. Effective, but reads as a worse
    # product experience, so measure both and disclose the choice.
    allow_other_fallback: bool = True

    # ---- ranking, role 3 ---------------------------------------------------
    # Priors blended into the final ordering. Popularity is a genuinely strong
    # baseline in leave-last-out Amazon benchmarks, so it earns real weight.
    #
    # PROVISIONAL, set by role 4 from a sweep, owned by role 3. Swept 0.0 to 5.0
    # over the 160 train sessions (artifacts/sweep_popularity.json and
    # sweep_pop_extended.json): the curve is unimodal, rising from 0.4608 at 0.0
    # to a plateau of about 0.743 across 1.5 to 3.0, then falling to 0.7176 at
    # 5.0. 2.0 is the middle of that plateau rather than its exact argmax,
    # because 1.5, 2.0 and 3.0 differ by less than 0.005 and picking the argmax
    # would be fitting noise on 160 sessions.
    #
    # It is not degenerate: evaluation/check_degeneracy.py shows unrelated
    # queries still return completely disjoint top tens even at weight 20,
    # because the prior only reorders a shortlist retrieval already filtered.
    weight_popularity: float = 2.0
    weight_rating: float = 0.05
    # Shortlist depth handed to the reranker. Deeper costs latency for little
    # gain once recall@100 is already 88%.
    rerank_depth: int = 100

    # ---- language, role 3, off by default ----------------------------------
    # The organiser may score us with networking disabled, so every LLM path
    # must be opt in and must degrade to the deterministic ordering.
    use_llm: bool = False
    llm_timeout_seconds: float = 8.0

    # Re-raise instead of falling back to the popularity list when a turn
    # fails. Production keeps this False, because a crash costs a whole session
    # and a degraded answer still might hit. Tests and the offline probe set it
    # True, because the blanket fallback otherwise makes them unfalsifiable: a
    # completely dead pipeline still returns a contract-legal popular list.
    strict_errors: bool = False

    # ---- platform, role 4 --------------------------------------------------
    # Sessions reserved from tuning. Never fit against these; the held out
    # curve is worth more to a judge than the score itself.
    holdout_size: int = 40
    # Seed for every shuffle and split, so a rerun reproduces exactly.
    seed: int = 20260101

    def with_overrides(self, **overrides: Any) -> "Config":
        """Return a copy with fields replaced, rejecting unknown names.

        The sweep harness builds variants through this rather than mutating, so
        a typo in a sweep definition fails loudly instead of silently tuning
        nothing.
        """
        known = {f.name for f in fields(self)}
        unknown = set(overrides) - known
        if unknown:
            raise ValueError(f"unknown config fields: {sorted(unknown)}")
        return replace(self, **overrides)

    def to_dict(self) -> dict[str, Any]:
        """Flat dict for logging alongside a result row."""
        return asdict(self)


DEFAULT = Config()
