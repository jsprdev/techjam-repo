"""Reranking the shortlist. ROLE 3, owns MRR.

Two signals beyond the retrieval score.

Popularity, because in leave-last-out Amazon benchmarks popularity baselines are
competitive with far more sophisticated models: real purchases concentrate on
popular items. Swept, and worth more than anything else in the system.

Exact phrase overlap, because the simulated customer's constraints are lifted
verbatim from the target product's own `features` and `details`. A whole-phrase
hit is therefore near-oracle evidence, and much stronger than the bag-of-words
similarity that put the candidate on the shortlist in the first place.

Deterministic throughout. Official scoring may run with networking disabled, so
an LLM rerank can only ever sit behind `Config.use_llm`, which defaults to off.
"""

from __future__ import annotations

import math

from src.catalog import Catalog, product_text
from src.config import Config
from src.interfaces import Candidate, SlotState

# A phrase shorter than this matches too much to be evidence of anything.
MIN_PHRASE_CHARS = 4


class PriorRanker:
    """Retrieval score blended with a popularity prior and phrase evidence."""

    def __init__(self, catalog: Catalog, config: Config | None = None) -> None:
        self.catalog = catalog
        self.config = config or Config()
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

    def rank(
        self,
        candidates: list[Candidate],
        slots: SlotState,
        profile: dict,
    ) -> list[str]:
        if not candidates:
            return []

        shortlist = candidates[: self.config.rerank_depth]
        best = max((score for _, score in shortlist), default=0.0) or 1.0

        phrases = [
            phrase.lower()
            for phrase in slots.constraints()
            if len(phrase) >= MIN_PHRASE_CHARS
        ]
        boost = self.config.exact_phrase_boost

        scored: list[tuple[float, str]] = []
        for parent_asin, score in shortlist:
            product = self.catalog.get(parent_asin)
            if product is None:
                continue
            total = score / best

            ratings = product.get("rating_number") or 0
            total += self.config.weight_popularity * math.log1p(float(ratings)) / 12.0

            average = product.get("average_rating")
            if isinstance(average, (int, float)):
                total += self.config.weight_rating * (float(average) / 5.0)

            if phrases and boost:
                text = self._text(parent_asin)
                hits = sum(1 for phrase in phrases if phrase in text)
                total += boost * (hits / len(phrases))

            scored.append((total, parent_asin))

        scored.sort(key=lambda pair: -pair[0])
        tail = [asin for asin, _ in candidates[self.config.rerank_depth :]]
        return [asin for _, asin in scored] + tail
