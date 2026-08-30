"""v0 ranker. OWNED BY ROLE 3, this is your starting point.

Passes retrieval order through, nudged by a popularity prior. That prior is not
arbitrary: in leave-last-out Amazon benchmarks, popularity baselines are
competitive with far more sophisticated models, because real purchases
concentrate on popular items.

Role 3's job is the reranker that turns 67% at rank ten into a high MRR, which
is 30% of the score. Phase 0 says the strongest available feature is exact
phrase overlap between the customer's disclosed constraints and a product's own
text, because the customer quotes that text verbatim. Also available: category
path agreement, rating, and price band.

Whatever you add stays deterministic. An LLM rerank may sit behind
`Config.use_llm`, which defaults to off, because the organiser may score us with
networking disabled.
"""

from __future__ import annotations

import math

from src.catalog import Catalog
from src.config import Config
from src.interfaces import Candidate, SlotState


class PriorRanker:
    """Retrieval score blended with a log popularity prior."""

    def __init__(self, catalog: Catalog, config: Config | None = None) -> None:
        self.catalog = catalog
        self.config = config or Config()

    def rank(
        self,
        candidates: list[Candidate],
        slots: SlotState,
        profile: dict,
    ) -> list[str]:
        if not candidates:
            return []
        shortlist = candidates[: self.config.rerank_depth]
        best = max(score for _, score in shortlist) or 1.0
        scored: list[tuple[float, str]] = []
        for parent_asin, score in shortlist:
            product = self.catalog.get(parent_asin)
            if product is None:
                continue
            # Normalise retrieval score so the priors stay comparable across
            # queries of different lengths.
            total = score / best
            ratings = product.get("rating_number") or 0
            total += self.config.weight_popularity * math.log1p(float(ratings)) / 12.0
            average = product.get("average_rating")
            if isinstance(average, (int, float)):
                total += self.config.weight_rating * (float(average) / 5.0)
            scored.append((total, parent_asin))
        scored.sort(key=lambda pair: -pair[0])
        tail = [asin for asin, _ in candidates[self.config.rerank_depth :]]
        return [asin for _, asin in scored] + tail
