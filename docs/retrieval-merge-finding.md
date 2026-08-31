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
