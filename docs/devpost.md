# ContextCart

## Inspiration

Shopping rarely begins with a perfect query. A customer starts with a vague idea,
reveals constraints gradually, and sometimes changes direction halfway through.
Most product search assumes you can already describe what you want, and real
shopping works the other way around: you recognise the right thing when you see
it, and the words come later. We built ContextCart to hold a conversation
through that process and adapt as the intent takes shape.

## What it does

ContextCart is a multi-turn shopping agent over a frozen 50,000 product Amazon
catalog. Each turn it reads what the shopper has said so far, updates a belief
over which product they mean, and either returns a ten product shortlist or asks
the one question that narrows the field fastest. It has ten turns at most.

It reaches a TechnicalScore of **0.893583** against the organiser's 0.1067
baseline, with Hit Rate@10 of 0.975, MRR of 0.7796 and mean turns to conversion
of 2.39 against the baseline's 9.81.

## How we built it

The idea that shaped the build is information gain per turn. Before writing any
dialogue policy we measured which questions this shopper can actually answer.
Asking about a product feature gets a useful reply in 96 percent of
conversations, asking about size gets one in twenty two, and three of the ten
attributes the interface allows can never be answered at all. The agent asks in
that measured order, and retires an attribute for the rest of a session once the
shopper shows they have nothing to say about it.

The pipeline runs entirely in memory. A per-turn router reads the current
message, the accumulated constraints and the belief's uncertainty to classify the
turn as open-ended Browsing or high-intent Buying, and widens or narrows the
candidate pool accordingly. Retrieval uses sparse TF-IDF vectors and cosine
similarity over a pooled index across product titles, features, categories,
descriptions, stores and details.

The ranker blends retrieval relevance with popularity, average rating, exact
phrase evidence from what the shopper has quoted, and an offline language model
appeal prior. Nothing is ever hard filtered. A violated constraint demotes a
candidate and always leaves it in the pool, because Amazon metadata has holes and
a filter that drops the target costs the whole session with no way back. That
decision is what lets the agent recover when a shopper reverses a preference
three turns in.

We resolved the brief's call for LLM semantic ranking against its warning that
official scoring may run with no network by moving the model off the turn path.
Claude Haiku 4.5 scores catalog products once, ahead of time, through the Batches
API, and the result is committed as a JSON artefact the ranker reads as a lookup.
An LLM judgment therefore sits inside the shipped ranking while the graded run
makes no network call and reports zero tokens. A live conversational reranker
also exists and is off by default, and every one of its failure paths returns the
deterministic ordering.

## Challenges we ran into

Catalog metadata is sparse and inconsistent, which is why we rank softly instead
of filtering.

The larger challenge was learning which of our own ideas were wrong. Field-aware
per-field retrieval reached a better Hit Rate in isolation and cost MRR
monotonically once measured against the pooled index, so it ships with its
weights at zero and the sweep recorded. Route fusion added no recall the pooled
route had not already found. A profile preference-tags ranking term produced a
gain inside the noise band and was removed. Three parser improvements that each
read as obvious each lost score, including dropping the shopper's "no additional
preference" replies, which turn out to carry more signal than they appear to.

Dense embeddings were ruled out rather than measured, and the distinction
matters. Our phase 0 diagnostics showed plain TF-IDF already reaches the target
inside the top 1000 in 100 percent of sessions once constraints are revealed, so
there was no recall problem for embeddings to solve. Every remaining miss is a
ranking failure. Downloading embedding weights would also have been a liability
under offline scoring.

## Accomplishments that we're proud of

We improved on the starter baseline by more than eight times while keeping the
system reproducible, offline capable and inside the ten turn limit. Every major
design choice is backed by an ablation or a diagnostic rather than an argument,
including the popularity weight, the rerank depth, the clarification order and
the intent override handling.

The one we are most pleased with is the measurement discipline. Forty of the 200
public sessions were reserved on day one, stratified and seeded, and no tuning
decision ever saw them. We spent them once, at the end, on the frozen
configuration: **0.8801 held out against 0.8969 on the sessions we tuned on**,
with Hit Rate@10 identical at 0.975 on both, so nothing about retrieval was
fitted to the training set.

We also found and fixed a bug that made the score depend on the installed numpy
version. The candidate pool was selected with an unstable partition, and TF-IDF
leaves most of the catalog tied at exactly zero, so which tied products entered a
wide shortlist varied between numpy builds. Ties now break by catalog index, and
we verified the same score across three Python and numpy combinations. A score
that moves with the grader's machine is not a score.

## What we learned

Measurement repeatedly overturned our design instincts, and the honest record of
what failed is now the most useful document in the repository.

Two results stand out. Retrieval was never the bottleneck, so effort spent adding
retrieval complexity would have been wasted, and ranking plus dialogue efficiency
mattered far more. And our intent router, which classifies the turn correctly in
100 percent of first turns, currently makes no difference to the shortlist the
shopper sees, because the measured optimum gives both tracks the same rerank
depth. We tried three ways to make the two tracks diverge and each one cost
score. We report that plainly rather than describing the router as though it
carries weight it does not.

## What's next for ContextCart

Our diagnostics point at one specific gap. When the target product fails to reach
first place, it is equal or better than the item above it on phrase evidence and
on retrieval similarity, and it loses on popularity alone. The phrase signal has
run out of resolution by that point. Breaking that tie calls for a model reading
the conversation, which is exactly what the live reranker is built for and what
we would measure next.

Beyond that: expand the offline semantic prior across the full catalog if a
larger sample shows a clear direction, strengthen ranking for open-ended
browsing, where MRR trails high-intent buying by 0.15, and move toward semantic
retrieval for shoppers who paraphrase rather than quote.

## Built With

- Python 3.11
- NumPy
- scikit-learn, for TF-IDF vectorisation and cosine similarity
- Anthropic Claude Haiku 4.5, through the Batches API, for the offline semantic prior
- Anthropic Messages API, for the optional live conversational reranker
- pytest
- Git and GitHub
- Amazon Reviews 2023, Clothing Shoes and Jewelry catalog, 50,000 products
- The organiser's TechJam local evaluator and Python agent interface

## Model choice, cost, token usage and latency

| | |
| --- | --- |
| Model | Claude Haiku 4.5, run offline through the Batches API to build the semantic prior. The optional live reranker is configured for `claude-opus-5` and is off by default. |
| Cost per evaluation | $0.00. The offline artefact cost about $0.14 to build once. |
| Token usage | 0 prompt, 0 completion, reported as such in the response `usage` field. |
| Latency | 938 ms p95 per turn and a 30 s one-off index build on a shared cloud instance. The same commit runs all 200 sessions in about 25 seconds on an Apple silicon laptop, so latency depends on the hardware rather than on the code. |
| Memory | 964 MB peak across a 200 session run. |
| Network | Not required. `evaluation/verify_offline.py` runs a full ten turn session with every socket entry point poisoned, and the probe is negative controlled, so injecting a real socket call makes it fail. |

The offline semantic prior currently covers 60 of the 50,000 products, a cost
controlled sample rather than a full pass, and it moves the score by 0.0005. The
full catalog run is costed at about $13.44 and has not been spent, because the
sample has yet to show a direction worth scaling.
