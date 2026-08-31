"""Field-aware lexical retrieval. OWNED BY ROLE 1.

Each catalog field receives its own TF-IDF representation, then their cosine
scores are blended at query time. This keeps title, feature, category, store,
description, and details evidence separately tunable without a brittle
text-duplication approximation. Whole customer phrases add soft evidence before
truncation, so they can promote a target that pooled lexical scoring placed too
low for the reranker to see.

Runs entirely in memory with no network access, which is required: the organiser
may score us with networking disabled.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.catalog import Catalog, product_text
from src.config import Config
from src.interfaces import Candidate

INDEX_FIELDS = (
    # The pooled index over all fields concatenated, kept ALONGSIDE the
    # per-field ones. Per-field cosine normalises by that field's own length,
    # so a product whose short `store` field contains a query term scores near
    # 1.0 on it, while the same term is diluted across a long document. That
    # systematically over-rewards short-field matches and crowds the top of the
    # list with plausible generics, which costs MRR. The pooled index restores
    # the "this document matches many query terms" signal that per-field
    # averaging throws away. Its weight is swept, not assumed.
    "pooled",
    "title",
    "features",
    "categories",
    "description",
    "store",
    "details",
)
FIELD_WEIGHT_NAMES = {
    "pooled": "weight_pooled",
    "title": "weight_title",
    "features": "weight_features",
    "categories": "weight_categories",
    "description": "weight_description",
    "store": "weight_store",
    "details": "weight_details",
}
MIN_PHRASE_CHARS = 4
WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class FieldIndex:
    """One independently vectorized catalog field."""

    vectorizer: TfidfVectorizer
    matrix: Any


def field_text(product: dict[str, Any], field: str) -> str:
    """Render one catalog field without mixing evidence from another field.

    The exception is "pooled", which deliberately mixes every field: that is
    the whole point of it, and it is what the per-field indices cannot see.
    """
    if field == "pooled":
        return product_text(product)
    value = product.get(field)
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return "" if value is None else str(value)


def normalize_phrases(phrases: Sequence[str]) -> list[str]:
    """Keep meaningful disclosed phrases in the form used for text matching."""
    normalized: list[str] = []
    for phrase in phrases:
        text = WHITESPACE.sub(" ", str(phrase).strip().lower())
        if len(text) >= MIN_PHRASE_CHARS:
            normalized.append(text)
    return normalized


class TfidfRetriever:
    """Weighted cosine similarity with soft whole-phrase evidence."""

    def __init__(self, catalog: Catalog, config: Config | None = None) -> None:
        self.catalog = catalog
        self.config = config or Config()
        self._asins = catalog.asins
        self._indices: dict[str, FieldIndex] = {}
        for field in INDEX_FIELDS:
            # Building an index nothing reads costs real time and memory: six
            # unused TF-IDF matrices add about two minutes to construction.
            if float(getattr(self.config, FIELD_WEIGHT_NAMES[field])) <= 0.0:
                continue
            index = self._build_field_index(field)
            if index is not None:
                self._indices[field] = index
        # Exact phrase matching needs to score every product before truncation.
        # Cache normalised text once, rather than reconstructing it every turn.
        self._searchable_texts = [product_text(p).lower() for p in catalog.products]

    def _build_field_index(self, field: str) -> FieldIndex | None:
        """Build one field matrix, tolerating an absent field in a small catalog."""
        vectorizer = TfidfVectorizer(
            lowercase=True,
            sublinear_tf=True,
            min_df=2,
            max_features=300_000,
            ngram_range=(1, 2),
            strip_accents="unicode",
        )
        try:
            matrix = vectorizer.fit_transform(
                field_text(product, field) for product in self.catalog.products
            )
        except ValueError as error:
            if "empty vocabulary" in str(error) or "no terms remain" in str(error):
                return None
            raise
        return FieldIndex(vectorizer=vectorizer, matrix=matrix)

    def retrieve(
        self,
        query: str,
        k: int,
        phrases: Sequence[str] = (),
    ) -> list[Candidate]:
        if not query.strip():
            return []

        scores = np.zeros(len(self._asins), dtype=np.float64)
        pooled_scores = np.zeros(len(self._asins), dtype=np.float64)
        total_weight = 0.0
        for field, index in self._indices.items():
            weight = float(getattr(self.config, FIELD_WEIGHT_NAMES[field]))
            if weight <= 0.0:
                continue
            vector = index.vectorizer.transform([query])
            field_score = (vector @ index.matrix.T).toarray()[0]
            if field == "pooled":
                # Kept separate as its own route rather than averaged in.
                pooled_scores = field_score
            scores += weight * field_score
            total_weight += weight

        if total_weight <= 0.0:
            return []
        scores /= total_weight
        if not pooled_scores.any():
            pooled_scores = scores

        normalized_phrases = normalize_phrases(phrases)
        phrase_boost = self.config.retrieval_exact_phrase_boost
        if normalized_phrases and phrase_boost:
            for position, text in enumerate(self._searchable_texts):
                hits = sum(phrase in text for phrase in normalized_phrases)
                bonus = phrase_boost * (hits / len(normalized_phrases))
                scores[position] += bonus
                pooled_scores[position] += bonus

        return self._fuse(scores, pooled_scores, k)

    def _top_k(self, scores: np.ndarray, k: int) -> np.ndarray:
        if k >= len(scores):
            return np.argsort(-scores)
        # argpartition is O(n) and we only need the top k ordered.
        top = np.argpartition(-scores, k)[:k]
        return top[np.argsort(-scores[top])]

    def _fuse(self, field_scores: np.ndarray, pooled_scores: np.ndarray, k: int) -> list[Candidate]:
        """Union two retrieval routes, ordered by the pooled route.

        The two routes fail differently. Field-aware scoring finds targets the
        pooled index misses, measured at HitRate 0.988 against 0.975, because
        per-field cosine surfaces a product whose one relevant field matches
        strongly. But it ORDERS badly: normalising by each field's own length
        over-rewards short-field matches, so plausible generics crowd the top
        and MRR falls from 0.790 to 0.615.

        Taking the union captures the recall without inheriting the ordering.
        Items are ordered by the pooled score, and anything only the field-aware
        route found is appended rather than interleaved, so it reaches the
        reranker (which rescores the whole shortlist) without displacing a
        confident pooled match ahead of the ranker ever seeing it.

        This is the multi-route retrieval the brief names, with the routes
        fused rather than one of them chosen.
        """
        # The seam promises at most k candidates, so the routes share the k
        # slots rather than widening the pool. The pooled route takes the
        # majority and the field route fills a reserved tail with items the
        # pooled route did not find at all.
        pooled_order = self._top_k(pooled_scores, min(k, len(pooled_scores)))
        pooled_hits = [
            (self._asins[i], float(pooled_scores[i]))
            for i in pooled_order
            if pooled_scores[i] > 0.0
        ]
        if not self.config.fuse_field_route:
            return pooled_hits[:k]

        reserved = int(k * self.config.fuse_reserve)
        results = pooled_hits[: k - reserved]
        seen = {asin for asin, _ in results}

        field_order = self._top_k(field_scores, min(k, len(field_scores)))
        for i in field_order:
            if len(results) >= k:
                break
            asin = self._asins[i]
            if asin in seen or field_scores[i] <= 0.0:
                continue
            seen.add(asin)
            # Carry the pooled score so the reranker sees one comparable scale.
            results.append((asin, float(pooled_scores[i])))

        # Any reserved slot the field route could not fill goes back to pooled.
        for candidate in pooled_hits[k - reserved :]:
            if len(results) >= k:
                break
            if candidate[0] not in seen:
                seen.add(candidate[0])
                results.append(candidate)
        return results[:k]
