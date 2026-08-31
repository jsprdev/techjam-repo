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
    weight_title: float = 0.0
    weight_features: float = 0.0
    weight_categories: float = 0.0
    weight_description: float = 0.0
    weight_store: float = 0.0

    # Per-field weights, all DEFAULTED OFF after measurement. The field-aware
    # route is kept and works, but every mixing ratio tested scored at or below
    # the pooled index alone. See docs/retrieval-merge-finding.md. A field whose
    # weight is zero is not indexed at all, so this costs nothing at runtime.
    weight_details: float = 0.0

    # Weight of the pooled all-fields index inside the field-weighted sum.
    # 0.0 gives pure per-field behaviour, high values approach a single pooled
    # index. Swept, because the two schemes trade MRR against MTTC.
    weight_pooled: float = 1.0

    # Multi-route retrieval. The field-aware route finds targets the pooled
    # route misses (HitRate 0.988 against 0.975) but orders them badly, so the
    # union is taken and the pooled route supplies the ordering.
    fuse_field_route: bool = False
    # Fraction of the k slots reserved for items only the field route found.
    # 0.0 disables fusion entirely; too high starves the pooled route.
    fuse_reserve: float = 0.25

    # Phrase boost applied inside RETRIEVAL, distinct from exact_phrase_boost
    # which the ranker applies to the shortlist. Retrieval boosts a whole
    # disclosed phrase before truncation; the ranker boosts it after.
    retrieval_exact_phrase_boost: float = 0.0

    # Multiplier applied when a whole disclosed phrase appears verbatim in a
    # product's text. The customer quotes the target's own record, so exact
    # phrase agreement is the single strongest signal available. Set to 1.0 to
    # disable and measure the difference.
    exact_phrase_boost: float = 4.0

    # ---- dialogue, role 2 --------------------------------------------------
    # Weight decay applied to a constraint per turn since it was last
    # reinforced. 1.0 disables decay and reproduces the pre-decay ranking
    # exactly. Lower values let a turn eight statement outrank a turn one
    # inference, which is what Intent Override needs.
    #
    # Read by `Slots.constraint_weights`, consumed by the ranker's phrase
    # evidence term. It does NOT reweight the retrieval query: retrieval takes a
    # plain string and the only way to weight a bag of words is repetition,
    # which collides with the measured finding that repetition is already
    # signal. See the note on `constraint_weights`.
    slot_decay: float = 1.0

    # How far a constraint the customer pivoted away from is demoted. 1.0 keeps
    # it at full strength, 0.0 is literal slot erasure.
    #
    # Not 0.0, and that is a measured decision rather than a soft reading of the
    # brief. In 28 of the 30 public intent_override sessions the preference the
    # customer says to ignore is itself lifted from the target product's own
    # record and appears verbatim in that product's text, so erasing it deletes
    # true evidence. Reproduce with evaluation/override_audit.py.
    override_demote: float = 0.5

    # Discard "I don't have an additional preference for X" style replies
    # instead of feeding them to retrieval. MEASURED FALSE: dropping them
    # scores 0.7369 against 0.7422 for keeping them, on the 160 train sessions.
    # The intuition that they are pure noise is wrong, or at least too small to
    # detect: their tokens are near-universal and carry almost no idf, so
    # removing them mostly just shortens the query.
    drop_no_information: bool = False

    # Collapse a repeated constraint into one occurrence. Off, because the
    # customer repeating something is signal: the repetition raises its term
    # frequency in the query, which is the behaviour we want.
    dedupe_phrases: bool = False

    # Append the profile's preference_tags to the query. They are vague ("fit",
    # "comfort") and Phase 0 found the profile carries no brand, category or
    # price history, so this is worth measuring rather than assuming.
    use_profile_tags: bool = True

    # Normalised belief entropy above which the candidate pool counts as
    # overloaded, triggering the over-generality cutoff in src/policy/commit.py.
    # Raise it to cut off less often.
    #
    # Set from the observed distribution rather than guessed, because a
    # threshold below what the system actually produces is not a cutoff, it is
    # the default branch. See evaluation/self_evolution.py for the per turn
    # distribution this number was read off.
    flat_belief_entropy: float = 0.92

    # ---- routing and commit policy, role 2 ---------------------------------
    # How much a Buying or Browsing cue in the wording moves the routing score
    # away from an undecided 0.5. Only consulted when no opener matched.
    intent_cue_weight: float = 0.15
    # How much each accumulated constraint pushes the turn towards Buying,
    # counted up to three. Constraints are the clearest sign a session has
    # converged, so this is the heaviest single signal.
    intent_constraint_weight: float = 0.2
    # How much belief flatness pushes the turn back towards Browsing, measured
    # as the gap between current entropy and `flat_belief_entropy`. Entropy
    # spans a narrow band in practice, so this needs a large multiplier to
    # matter at all.
    intent_entropy_weight: float = 2.0

    # Shortlist depth and belief sharpening per track. Both default to the
    # untracked behaviour so that turning routing on changes nothing until these
    # are moved, which is what makes the routing ablation honest.
    track_depth_buying: int = 200
    track_depth_browsing: int = 200
    track_sharpen_buying: float = 1.0
    track_sharpen_browsing: float = 1.0

    # Share of the belief mass the best candidate must hold before the agent
    # presents its list as a recommendation rather than a clarification. Read
    # off the training distribution: sessions that hit at rank one carry a
    # median peak share of 0.089, those that hit lower down 0.065.
    commit_peak_share: float = 0.10
    # Shortlist depth used on a turn where the over-generality cutoff fired.
    # This is the "immediate retrieval cutoff" of Pillar II: the wide low
    # confidence pool is cut at source rather than suppressed in the answer.
    overload_depth: int = 200

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
    # Shortlist depth handed to the reranker. This interacts with the
    # truncation width above: effective depth is min(rerank_depth, width), so
    # anything past truncate_buying (200) only affects Browsing turns.
    #
    # Swept twice, and the answer REVERSED once exact phrase overlap existed.
    # Before it, depth 200 bought composite at the cost of MRR. After it:
    #
    #     100  0.8540  hit 0.919  mrr 0.804  mttc 3.33
    #     200  0.8951  hit 0.975  mrr 0.790  mttc 2.48
    #     400  0.8737  hit 0.975  mrr 0.704  mttc 2.24
    #     800  0.8670  hit 0.981  mrr 0.664  mttc 2.14
    #
    # Every remaining miss at depth 100 was a target sitting in the candidate
    # pool beyond rank 100, never reranked and so never given its phrase
    # evidence. Widening to 200 rescues them. Past 200 the extra candidates
    # dilute rank one faster than they add hits.
    rerank_depth: int = 200

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
