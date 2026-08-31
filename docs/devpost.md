# ContextCart

## Inspiration

Shopping rarely begins with a perfect query. A customer may start with a vague
idea, reveal constraints gradually, or change direction halfway through. We
built ContextCart to move beyond static keyword matching and adapt its search
strategy as that intent evolves.

## What it does

ContextCart is a multi-turn shopping agent for a 50,000-product catalog. It
classifies each turn as open-ended Browsing or high-intent Buying, updates its
belief from the conversation, retrieves and ranks candidates, and either asks a
useful clarification question or returns a ten-product shortlist. It reaches a
recorded TechnicalScore of 0.893583 against the organiser's 0.1067 baseline,
with 0.975 Hit Rate@10.

## How we built it

The pipeline runs entirely in memory. A per-turn intent router uses the current
message, accumulated constraints, and belief uncertainty to choose a broad
Browsing route or a narrower Buying route. Retrieval uses sparse TF-IDF vectors
and cosine similarity across product titles, features, categories,
descriptions, stores, and details. We also implemented field-aware retrieval and
route fusion, but measurements showed that the pooled route produced better
MRR, so that is the shipped default.

The ranker blends retrieval relevance with popularity, rating, recency-weighted
exact phrase evidence, and a small offline LLM-derived appeal prior. A live LLM
reranker can read the conversation and reorder the top 20 candidates, but it is
optional and fails safely to deterministic ranking. This keeps the official
path reliable when network access is unavailable.

The offline semantic sample was produced with Claude Haiku 4.5 through the
Batches API, and its output is committed as `artifacts/semantic_prior.json` so
the ranker reads an LLM judgment at runtime without making a call. The optional
live conversational reranker is configured for `claude-opus-5` and is off by
default. The shipped path makes no runtime model calls, reports zero tokens,
and costs $0.00 per evaluation.

Latency depends on the hardware: a full 200-session run takes about 25 seconds
on an Apple silicon laptop and 1 minute 40 seconds on a mid-range cloud
instance. The score itself does not depend on the machine, which we verified
across three Python and numpy combinations after fixing a tie-break in the
retriever that had made it drift.

## Challenges we ran into

Catalog metadata was sparse and inconsistent, so hard filters could remove the
correct product permanently. We used soft ranking signals instead. Some ideas
that sounded stronger, including dense embeddings, field-aware weighting, and
route fusion, did not improve the measured score. We also had to reconcile the
brief's LLM ranking goal with the possibility that official evaluation would
disable networking.

## Accomplishments that we're proud of

We improved the starter baseline by more than eight times while keeping the
system reproducible, offline-capable, and within the ten-turn limit. Every major
design choice was supported by an ablation or diagnostic, including popularity
weight, rerank depth, clarification order, intent override handling, and the
decision to keep optional components off by default.

## What we learned

The best-looking architecture is not always the best-performing one.
Measurement repeatedly changed our decisions. Retrieval already found nearly
every target, so ranking and dialogue efficiency mattered more than adding more
retrieval complexity. We also learned that a safe deterministic fallback can
make an LLM-enhanced system more practical, not less ambitious.

## What's next for ContextCart

We would improve ranking for open-ended browsing, test the live LLM reranker
with a real key, and expand the semantic prior only if a larger sample shows a
clear gain. For real shoppers, we would add stronger semantic retrieval for
natural paraphrases, evaluate on less simulator-specific language, and support
long-term preferences across sessions.

## Built With

- Python 3.11
- NumPy
- scikit-learn, TF-IDF vectorization and cosine similarity
- Anthropic Claude Haiku 4.5 for the offline semantic sample
- Optional Anthropic API integration for live reranking
- pytest
- Git and GitHub
- Amazon Reviews 2023, Clothing, Shoes and Jewelry catalog
- TechJam local evaluator and Python agent interface
