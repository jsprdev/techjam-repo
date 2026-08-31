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
        total_weight = 0.0
        for field, index in self._indices.items():
            weight = float(getattr(self.config, FIELD_WEIGHT_NAMES[field]))
            if weight <= 0.0:
                continue
            vector = index.vectorizer.transform([query])
            scores += weight * (vector @ index.matrix.T).toarray()[0]
            total_weight += weight

        if total_weight <= 0.0:
            return []
        scores /= total_weight

        normalized_phrases = normalize_phrases(phrases)
        phrase_boost = self.config.retrieval_exact_phrase_boost
        if normalized_phrases and phrase_boost:
            for position, text in enumerate(self._searchable_texts):
                hits = sum(phrase in text for phrase in normalized_phrases)
                scores[position] += phrase_boost * (hits / len(normalized_phrases))

        if k >= len(scores):
            order = np.argsort(-scores)
        else:
            # argpartition is O(n) and we only need the top k ordered.
            top = np.argpartition(-scores, k)[:k]
            order = top[np.argsort(-scores[top])]
        return [(self._asins[i], float(scores[i])) for i in order[:k] if scores[i] > 0.0]
