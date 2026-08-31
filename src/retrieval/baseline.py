"""v0 lexical retriever. OWNED BY ROLE 1, this is your starting point.

Shipped on day 1 so roles 2 and 3 are never blocked waiting on retrieval. It is
deliberately plain: TF-IDF over the concatenated product text, cosine scored.
Phase 0 measured this exact approach at 67% recall@10 and 100% recall@1000 once
every constraint is disclosed, against a 12.5% baseline.

Role 1's job is to beat it. The two levers Phase 0 points at:

1. Per field weighting. Right now every field is pooled into one bag of words,
   so a brand name in `store` counts the same as a word in a care instruction.
   `Config` already carries weight_title, weight_features and friends, unused.
2. Exact phrase boosting. The simulated customer quotes the target product's own
   `features` and `details` verbatim, so a whole-phrase hit is far stronger
   evidence than the sum of its words. `Config.exact_phrase_boost` is unused.

Runs entirely in memory with no network access, which is required: the organiser
may score us with networking disabled.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.catalog import Catalog, product_text
from src.config import Config
from src.interfaces import Candidate


class TfidfRetriever:
    """Cosine similarity over a TF-IDF index of the whole catalog."""

    def __init__(self, catalog: Catalog, config: Config | None = None) -> None:
        self.catalog = catalog
        self.config = config or Config()
        self._asins = catalog.asins
        # Bigrams matter here: the customer speaks in phrases, so "stainless
        # steel" should not decompose into two common unigrams. min_df=2 drops
        # the long tail of one-off tokens that only bloat the vocabulary.
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            sublinear_tf=True,
            min_df=2,
            max_features=300_000,
            ngram_range=(1, 2),
            strip_accents="unicode",
        )
        self._matrix = self._vectorizer.fit_transform(
            product_text(p) for p in catalog.products
        )

    def retrieve(self, query: str, k: int) -> list[Candidate]:
        if not query.strip():
            return []
        vector = self._vectorizer.transform([query])
        scores = (vector @ self._matrix.T).toarray()[0]
        if k >= len(scores):
            order = np.argsort(-scores)
        else:
            # argpartition is O(n) and we only need the top k ordered.
            top = np.argpartition(-scores, k)[:k]
            order = top[np.argsort(-scores[top])]
        return [(self._asins[i], float(scores[i])) for i in order[:k] if scores[i] > 0.0]
