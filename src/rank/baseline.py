"""Reranking the shortlist. ROLE 3, owns MRR.

Two signals beyond the retrieval score.

Popularity, because in leave-last-out Amazon benchmarks popularity baselines are
competitive with far more sophisticated models: real purchases concentrate on
popular items. Swept, and worth more than anything else in the system.

Exact phrase overlap, because the simulated customer's constraints are lifted
verbatim from the target product's own `features` and `details`. A whole-phrase
hit is therefore near-oracle evidence, and much stronger than the bag-of-words
similarity that put the candidate on the shortlist in the first place.

The output of the blend is a `Belief`, not a bare list. Same arithmetic, same
ordering, but the distribution it came from survives the call, so the commit
policy can read how decisive the evidence actually was instead of guessing from
the ranking alone. `rank()` is still the frozen `Ranker` interface and still
returns identifiers: it is now one line over `believe()`.

Deterministic throughout. Official scoring may run with networking disabled, so
an LLM rerank can only ever sit behind `Config.use_llm`, which defaults to off.
"""

from __future__ import annotations

import math
from typing import Any

from src import semantic
from src.catalog import Catalog, product_text
from src.config import Config
from src.interfaces import Candidate, SlotState
from src.state.belief import Belief
from src.trace import SINK

# A phrase shorter than this matches too much to be evidence of anything.
MIN_PHRASE_CHARS = 4


class PriorRanker:
    """Retrieval score blended with a popularity prior and phrase evidence."""

    def __init__(self, catalog: Catalog, config: Config | None = None) -> None:
        self.catalog = catalog
        self.config = config or Config()
        # The offline LLM artefact, read once. Absent means no signal.
        self.prior = semantic.load()
        # Unenriched products sit at the mean of the enriched ones, so a partial
        # artefact does not turn coverage itself into a ranking signal.
        self._appeal_default = (
            sum(self.prior.appeal.values()) / len(self.prior.appeal)
            if len(self.prior)
            else 0.0
        )
        # Lowercased product text, built lazily and only for products that
        # actually reach a shortlist. Precomputing all 50,000 would add seconds
        # to construction for text most sessions never look at.
        self._lowered: dict[str, str] = {}

    def _text(self, parent_asin: str) -> str:
        cached = self._lowered.get(parent_asin)
        if cached is None:
            product = self.catalog.get(parent_asin)
            cached = product_text(product).lower() if product else ""
            self._lowered[parent_asin] = cached
        return cached

    def believe(
        self,
        candidates: list[Candidate],
        slots: SlotState,
        profile: dict,
        depth: int | None = None,
        sharpen: float = 1.0,
    ) -> Belief:
        """Blend the evidence into a belief over the candidate pool.

        Weights are recency weighted when `Config.slot_decay` is below 1.0, so a
        constraint the customer stated this turn counts for more than one they
        stated six turns ago. With decay at 1.0 every phrase weighs 1.0 and the
        term collapses to the plain hit fraction.

        `depth` and `sharpen` come from the track chosen for this turn
        (`src/policy/intent.py`) and default to the untracked behaviour, so a
        caller that does not route, such as a diagnostic, gets exactly what this
        ranker produced before routing existed. `sharpen` above 1.0 pulls the
        retrieval score's top away from its tail, which is how spec 5.1's
        "sharpen the belief aggressively around constraint matches" on the Buying
        track is actually implemented.

        Nothing is filtered. A candidate that matches no phrase keeps its
        retrieval score and its priors and simply sinks, per spec 5.4, and the
        pool past the rerank depth is carried in `Belief.tail` rather than
        discarded.
        """
        if not candidates:
            return Belief(asins=[], scores=[])

        cut = self.config.rerank_depth if depth is None else depth
        shortlist = candidates[:cut]
        best = max((score for _, score in shortlist), default=0.0) or 1.0

        phrases, weights = self._weighted_phrases(slots)
        # Guards a pathological decay setting where every weight has underflowed
        # to zero. The blend then reduces to retrieval plus priors, which is a
        # degraded answer rather than a crash, and agent.py must never raise.
        weight_total = sum(weights) or 0.0
        boost = self.config.exact_phrase_boost if weight_total > 0.0 else 0.0

        # Per-candidate score components, kept only while the trace sink is on.
        # `evaluation/rank_diagnostics.py` needs them to answer the one question
        # a bare score cannot: when the target lands at rank 7, which component
        # did the item above it win on? Building this list unconditionally would
        # allocate a dict per candidate per turn on the scored path, so it is
        # gated on the sink being enabled and is empty during a real run.
        explain = SINK.enabled
        components: list[dict[str, Any]] = []

        scored: list[tuple[float, str]] = []
        for parent_asin, score in shortlist:
            product = self.catalog.get(parent_asin)
            if product is None:
                continue
            retrieval_term = score / best
            if sharpen != 1.0:
                retrieval_term **= sharpen
            total = retrieval_term

            ratings = product.get("rating_number") or 0
            popularity_term = self.config.weight_popularity * math.log1p(float(ratings)) / 12.0
            total += popularity_term

            appeal_term = 0.0
            if self.config.weight_appeal and len(self.prior):
                appeal_term = self.config.weight_appeal * self.prior.appeal_of(
                    parent_asin, self._appeal_default
                )
                total += appeal_term

            rating_term = 0.0
            average = product.get("average_rating")
            if isinstance(average, (int, float)):
                rating_term = self.config.weight_rating * (float(average) / 5.0)
                total += rating_term

            phrase_term = 0.0
            if phrases and boost:
                text = self._text(parent_asin)
                hits = sum(w for phrase, w in zip(phrases, weights) if phrase in text)
                phrase_term = boost * (hits / weight_total)
                total += phrase_term

            if explain:
                components.append(
                    {
                        "parent_asin": parent_asin,
                        "retrieval": round(retrieval_term, 6),
                        "popularity": round(popularity_term, 6),
                        "appeal": round(appeal_term, 6),
                        "rating": round(rating_term, 6),
                        "phrase": round(phrase_term, 6),
                        "total": round(total, 6),
                    }
                )

            scored.append((total, parent_asin))

        scored.sort(key=lambda pair: -pair[0])
        tail = [asin for asin, _ in candidates[cut:]]
        if explain:
            components.sort(key=lambda row: -row["total"])
        return Belief(
            asins=[asin for _, asin in scored],
            scores=[score for score, _ in scored],
            tail=tail,
            components=components,
        )

    def rank(
        self,
        candidates: list[Candidate],
        slots: SlotState,
        profile: dict,
    ) -> list[str]:
        """Return `parent_asin` values, best first. The frozen `Ranker` seam."""
        return self.believe(candidates, slots, profile).ranking()

    def _weighted_phrases(self, slots: SlotState) -> tuple[list[str], list[float]]:
        """Constraint phrases paired with their recency weight.

        `SlotState` is a protocol, so a state object that predates decay still
        works: without `constraint_weights` every phrase weighs 1.0, which is
        exactly what `Config.slot_decay = 1.0` produces.
        """
        raw = slots.constraints()
        getter = getattr(slots, "constraint_weights", None)
        raw_weights = list(getter()) if callable(getter) else [1.0] * len(raw)

        phrases: list[str] = []
        weights: list[float] = []
        for phrase, weight in zip(raw, raw_weights):
            if len(phrase) >= MIN_PHRASE_CHARS:
                phrases.append(phrase.lower())
                weights.append(float(weight))
        return phrases, weights
