# Field-aware vs pooled retrieval: the measurement

Written while merging `codex/field-aware-retrieval` into the dialogue and belief work.
All numbers are the 160 session train split, same agent, same ranker, only the retrieval
index differs.

## The question

Jarell's field-aware retriever reaches a better MTTC than the pooled index but a much worse
MRR. Can the two be combined, or does one have to win?

## The mechanism

He builds **six separate TF-IDF indices**, one per catalog field, and scores a query as the
weighted sum of their cosine similarities. The pooled version builds **one index** over all
fields concatenated.

Per-field cosine normalises by that field's own length. A product whose `store` field is the
single word "Casio" scores near 1.0 on that field when the query mentions Casio, while in a
pooled index the same term is diluted across a 1,500 character document. Per-field scoring
therefore over-rewards short-field matches, filling the top of the list with plausible
generics and pushing the exact target down a few places. That is precisely what MRR measures.

It also discards the signal that one document matches *many* query terms at once, because
averaging normalised per-field scores flattens it.

## The test

The pooled index was added back as a seventh weighted field, so the two schemes could be
mixed continuously rather than chosen between. `weight_pooled=0` is pure per-field, high
values approach pure pooled.

| weight_pooled | TechnicalScore | HitRate@10 | MRR | MTTC |
| --- | --- | --- | --- | --- |
| 0.0, pure field-aware | 0.8464 | 0.969 | 0.615 | 2.12 |
| 1.0 | 0.8484 | 0.969 | 0.624 | 2.16 |
| 3.0 | 0.8536 | 0.969 | 0.642 | 2.17 |
| 8.0 | 0.8583 | 0.969 | 0.660 | 2.20 |
| 100, phrase boost off | 0.8898 | 0.975 | 0.766 | 2.37 |
| pure pooled | **0.8951** | 0.975 | **0.790** | 2.48 |

**Monotonic.** Every gram of per-field weighting costs MRR in proportion to its weight, and
there is no interior optimum. The answer is that the two cannot be combined: pooled indexing
wins outright on this catalog and this metric mix.

## Two things worth keeping from the field-aware branch

**The `phrases` seam.** `Retriever.retrieve(query, k, phrases=())` is a good design. The
flattened query loses phrase boundaries, and those boundaries carry the evidence, because
the customer quotes the target product's own text verbatim. The argument is optional, so it
does not break a retriever that ignores it. Kept.

**The hit rate.** Field-aware retrieval reached HitRate@10 of 0.988 at one setting, the best
any configuration has produced here, and its MTTC is consistently better. It finds the target
and then ranks it badly. If that recall can be kept while fixing the ranking, it beats
everything measured so far. That is the interesting thread, not the weighting itself.

## Why swapping rankers did nothing

The first merge attempt kept the field-aware retriever and substituted the belief ranker. The
result was identical to the retrieval branch on all 200 sessions. His retriever applies its
phrase boost *before* truncation, so the top ten is already fixed by the time any reranker
runs. A reranker cannot recover precision that retrieval has already discarded.


## Second attempt: fusing the two routes

Since the 0.988 was a recall win and the MRR loss was a ranking loss, the obvious next move
was to run both routes and take the union, ordering by the pooled route so the field route
could contribute candidates without contributing its ordering. That is also the multi-route
retrieval Pillar I names.

It did not work. The field route contributed no recall the pooled route had not already
found, and every slot given to it cost MRR:

| configuration | Score | HitRate@10 | MRR | MTTC |
| --- | --- | --- | --- | --- |
| pooled only, boost off | **0.8951** | 0.975 | **0.790** | 2.48 |
| fusion, 5% of slots reserved | 0.8940 | 0.975 | 0.782 | 2.40 |
| fusion, 25% reserved | 0.8865 | 0.975 | 0.755 | 2.37 |
| fusion, 25% reserved, boost on | 0.8594 | 0.969 | 0.668 | 2.27 |
| fusion, 40% reserved | 0.8583 | 0.969 | 0.664 | 2.27 |

HitRate never moved above 0.975 in any fused configuration. The 0.988 the field-aware branch
reached is not extra recall that fusion can borrow: it comes from the field route DRIVING the
final ordering, which is the same thing that costs 0.175 of MRR. The two are the same
mechanism seen from two sides, and they are not separable.

Arithmetically that operating point is worse for us: +0.013 HitRate at weight 0.50 is worth
+0.0065, while -0.175 MRR at weight 0.30 costs -0.0525. A net loss of about 0.046.

## What shipped

The pooled configuration, which measures 0.8951 on train and 0.8931 through the official
harness on all 200 sessions.

The field-aware code is kept, not deleted. Every per-field weight defaults to 0.0 and a field
with zero weight is never indexed, so the six unused matrices cost nothing: construction is
17.9 seconds. Anyone who wants to revisit this sets the weights and re-sweeps, with this
table as the baseline to beat.
