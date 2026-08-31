"""Reranking the shortlist. ROLE 3, owns MRR.

Popularity is a useful prior in leave-last-out Amazon benchmarks. Exact phrase
overlap is equally important here because the simulated customer repeats text
from the target catalog record. The implementation remains deterministic and
fully offline, with score explanations available only to local diagnostics.
"""

from __future__ import annotations

import math

from src.catalog import Catalog, product_text
from src.config import Config
from src.interfaces import Candidate, SlotState

MIN_PHRASE_CHARS = 4


class PriorRanker:
    """Retrieval score blended with popularity and phrase evidence."""

    def __init__(self, catalog: Catalog, config: Config | None = None) -> None:
        self.catalog = catalog
        self.config = config or Config()
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
        scored = self._score(candidates, slots)
        tail = [asin for asin, _ in candidates[self.config.rerank_depth :]]
        return [str(item["parent_asin"]) for item in scored] + tail

    def explain(
        self,
        candidates: list[Candidate],
        slots: SlotState,
        profile: dict,
    ) -> list[dict[str, float | str]]:
        """Return trace-only score components for the current shortlist.

        The evaluator never receives these explanations. They let offline
        diagnostics attribute rank losses to retrieval, phrase evidence, or the
        popularity prior instead of tuning constants blindly.
        """
        return self._score(candidates, slots)

    def _score(
        self,
        candidates: list[Candidate],
        slots: SlotState,
    ) -> list[dict[str, float | str]]:
        if not candidates:
            return []

        shortlist = candidates[: self.config.rerank_depth]
        best = max((score for _, score in shortlist), default=0.0) or 1.0
        phrases = [
            phrase.lower()
            for phrase in slots.constraints()
            if len(phrase) >= MIN_PHRASE_CHARS
        ]
        scored: list[dict[str, float | str]] = []
        for parent_asin, score in shortlist:
            product = self.catalog.get(parent_asin)
            if product is None:
                continue
            lexical = score / best
            popularity = self.config.weight_popularity * math.log1p(
                float(product.get("rating_number") or 0)
            ) / 12.0
            average = product.get("average_rating")
            rating = (
                self.config.weight_rating * (float(average) / 5.0)
                if isinstance(average, (int, float))
                else 0.0
            )
            phrase = 0.0
            if phrases and self.config.exact_phrase_boost:
                hits = sum(1 for phrase_value in phrases if phrase_value in self._text(parent_asin))
                phrase = self.config.exact_phrase_boost * (hits / len(phrases))
            scored.append(
                {
                    "parent_asin": parent_asin,
                    "retrieval": lexical,
                    "popularity": popularity,
                    "rating": rating,
                    "phrase": phrase,
                    "total": lexical + popularity + rating + phrase,
                }
            )
        return sorted(scored, key=lambda item: float(item["total"]), reverse=True)
