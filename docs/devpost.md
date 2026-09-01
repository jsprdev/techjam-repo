# ContextCart

## Inspiration

Shopping rarely starts with a perfect query. You know roughly what you want, you
recognise the right thing when you see it, and the words come later. Keyword
search assumes the opposite. We built ContextCart to hold a conversation instead.

## What it does

A multi-turn shopping agent over a frozen 50,000 product Amazon catalog. Each
turn it reads what the shopper has said, updates a belief over which product they
mean, and either returns a ten product shortlist or asks the question that
narrows the field fastest. Ten turns at most.

TechnicalScore **0.893583** against the organiser's 0.1067 baseline. Hit Rate@10
0.975, MRR 0.7796, mean turns to conversion 2.39 against the baseline's 9.81.

## How we built it

The build was organised around information gain per turn. Before writing any
dialogue policy we measured which questions this shopper can actually answer:
asking about a feature pays in 96 percent of conversations, size pays in one in
twenty two, and three of the ten allowed attributes can never be answered at all.
The agent asks in that order and retires an attribute once it stops paying.

Everything runs in memory. A per-turn router classifies the turn as Browsing or
Buying and sizes the candidate pool. Retrieval is sparse TF-IDF with cosine
similarity over a pooled index across titles, features, categories, descriptions,
stores and details. The ranker blends retrieval relevance, popularity, rating,
exact phrase evidence and an offline language model appeal prior.

Nothing is ever hard filtered. A violated constraint demotes a candidate and
leaves it in the pool, because Amazon metadata has holes and a filter that drops
the target loses the session with no way back. That is what lets the agent
recover when a shopper reverses a preference three turns in.

The brief asks for LLM semantic ranking and the rules warn that scoring may run
with no network. We resolved that by moving the model off the turn path: Claude
Haiku 4.5 scores products once through the Batches API, and the result ships as a
committed artefact the ranker reads as a lookup. A live conversational reranker
also exists and is off by default.

## Challenges we ran into

The real challenge was learning which of our own ideas were wrong. Field-aware
retrieval reached a better Hit Rate in isolation and cost MRR monotonically once
swept against the pooled index, so it ships at zero weight with the measurement
recorded. Route fusion added no recall. A profile preference-tags ranking term
landed inside the noise band and was removed. Three parser improvements that each
read as obvious each lost score.

Dense embeddings were ruled out rather than measured. Phase 0 showed TF-IDF
already reaches the target inside the top 1000 in 100 percent of sessions, so
there was no recall problem for embeddings to solve, and every remaining miss is
a ranking failure.

## Accomplishments that we're proud of

Eight times the starter baseline, reproducible, offline capable, inside the ten
turn limit, with every major choice backed by an ablation.

The one we are proudest of is measurement discipline. Forty of the 200 sessions
were reserved on day one and no tuning decision ever saw them. Spent once at the
end: **0.8801 held out against 0.8969 on what we tuned on**, Hit Rate identical
at 0.975 on both.

We also found a bug that made the score depend on the installed numpy version.
TF-IDF leaves most of the catalog tied at exactly zero, and an unstable partition
meant which tied products entered a wide shortlist varied between builds. Ties
now break by catalog index, verified identical across three Python and numpy
combinations.

## What we learned

Measurement kept overturning our instincts. Retrieval was never the bottleneck,
so ranking and dialogue efficiency mattered far more than adding retrieval
complexity.

The result we report most carefully: our intent router classifies the first turn
correctly every time, and it currently makes no difference to the shortlist,
because the measured optimum gives both tracks the same rerank depth. We tried
three ways to make the tracks diverge and each cost score.

## What's next for ContextCart

Our diagnostics point at one gap. When the target misses first place it is equal
or better on phrase evidence and retrieval similarity, and loses on popularity
alone. The phrase signal has run out of resolution. Breaking that tie needs a
model reading the conversation, which is what the live reranker is for.

Then: expand the semantic prior if a larger sample shows a direction, lift
browsing MRR where it trails buying by 0.15, and move toward semantic retrieval
for shoppers who paraphrase rather than quote.

## Built With

Python 3.11, NumPy, scikit-learn, Anthropic Claude Haiku 4.5 via the Batches API,
the Anthropic Messages API, pytest, Git and GitHub, the Amazon Reviews 2023
Clothing Shoes and Jewelry catalog, and the organiser's TechJam local evaluator.

## Model, cost, tokens and latency

| | |
| --- | --- |
| Model | Claude Haiku 4.5, offline via the Batches API. The optional live reranker is `claude-opus-5`, off by default. |
| Cost | $0.00 per evaluation. The offline artefact cost about $0.14 once. |
| Tokens | 0 prompt, 0 completion, reported in the response `usage` field. |
| Latency | 938 ms p95 per turn on a shared cloud instance, and about 25 seconds for all 200 sessions on an Apple silicon laptop. Latency depends on the hardware. |
| Memory | 964 MB peak across a 200 session run. |
| Network | Not required. `evaluation/verify_offline.py` runs a full session with every socket poisoned, and is negative controlled. |

The offline prior covers 60 of the 50,000 products, a cost controlled sample, and
moves the score by 0.0005. The full pass is costed at about $13.44.
