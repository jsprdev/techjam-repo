# Retrieval and ranking experiments: the measurements

Measured attempts to move HitRate or MRR that were rejected, kept here with their numbers
so nobody re-runs them without a new reason. All numbers are the 160 session train split.

1. [Field-aware vs pooled retrieval](#field-aware-vs-pooled-retrieval)
2. [Profile preference-tags in the ranker](#profile-preference-tags-in-the-ranker)

---

## Field-aware vs pooled retrieval

Written while merging `codex/field-aware-retrieval` into the dialogue and belief work.
All numbers are the 160 session train split, same agent, same ranker, only the retrieval
index differs.

### The question

Jarell's field-aware retriever reaches a better MTTC than the pooled index but a much worse
MRR. Can the two be combined, or does one have to win?

### The mechanism

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

### The test

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

### Two things worth keeping from the field-aware branch

**The `phrases` seam.** `Retriever.retrieve(query, k, phrases=())` is a good design. The
flattened query loses phrase boundaries, and those boundaries carry the evidence, because
the customer quotes the target product's own text verbatim. The argument is optional, so it
does not break a retriever that ignores it. Kept.

**The hit rate.** Field-aware retrieval reached HitRate@10 of 0.988 at one setting, the best
any configuration has produced here, and its MTTC is consistently better. It finds the target
and then ranks it badly. If that recall can be kept while fixing the ranking, it beats
everything measured so far. That is the interesting thread, not the weighting itself.

### Why swapping rankers did nothing

The first merge attempt kept the field-aware retriever and substituted the belief ranker. The
result was identical to the retrieval branch on all 200 sessions. His retriever applies its
phrase boost *before* truncation, so the top ten is already fixed by the time any reranker
runs. A reranker cannot recover precision that retrieval has already discarded.

### Second attempt: fusing the two routes

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

### What shipped

The pooled configuration, which measures 0.8951 on train and 0.8931 through the official
harness on all 200 sessions.

The field-aware code is kept, not deleted. Every per-field weight defaults to 0.0 and a field
with zero weight is never indexed, so the six unused matrices cost nothing: construction is
17.9 seconds. Anyone who wants to revisit this sets the weights and re-sweeps, with this
table as the baseline to beat.

---

## Profile preference-tags in the ranker

Considered while answering Pillar III's "distil the profile into retrieval **and ranking**"
(spec 7.1). The retrieval half already exists: `state/slots.py` appends the buyer's
`preference_tags` to the query, and removing them costs 0.037. The ranking half did not:
`rank/baseline.py::believe()` receives `profile` and never reads it.

**Measured, no effect, not merged.** The prototype and its tests were removed after the
sweep below; this section is the record so nobody rebuilds it.

### What was tried

A ranking term `weight_profile_tags * (matched_tags / total_tags)` added to each shortlist
candidate's score in `believe()`, where a tag is "matched" if one of its expansion words
appears in the product text on a word boundary. Tags were expanded to the words that
actually occur in the catalog (`warmth -> warm | warmth | insulated | insulating`,
`weather -> weather | waterproof | water resistant | water-resistant | rainproof`). Word
boundaries mattered: bare `"rain"` is a substring of `"training"` in 890 catalog products,
`"warm"` of `"warm water wash"` care text.

### The tag vocabulary

9 distinct tags across the 200 sessions: `fit` (163), `material` (154), `comfort` (144),
`style` (101), `durability` (47), `performance` (26), `warmth` (18), `weather` (12),
`general shopping` (1). The first four are near-universal in clothing text and carry almost
no discriminative power; the rare four are the only ones that can separate two
lexically-similar products. `fit` is also below `MIN_PHRASE_CHARS` and is dropped.

### Full tag set: monotonically negative

160 train sessions, only `weight_profile_tags` varied:

| weight | TechnicalScore | HitRate@10 | MRR | MTTC |
| --- | --- | --- | --- | --- |
| **0.0** | **0.8963** | 0.975 | **0.794** | 2.48 |
| 0.1 | 0.8931 | 0.975 | 0.784 | 2.48 |
| 0.25 | 0.8885 | 0.975 | 0.769 | 2.48 |
| 0.5 | 0.8831 | 0.975 | 0.753 | 2.52 |
| 1.0 | 0.8663 | 0.969 | 0.714 | 2.61 |
| 2.0 | 0.8270 | 0.938 | 0.662 | 3.01 |

Every increment costs MRR, and at high weight it costs HitRate too. Same mechanism as
field-aware retrieval above: a signal that matches almost everything dilutes the exact
phrase evidence that actually locates the target. `artifacts/sweep_profile_tags.json`.

### Discriminating tags only: a noise-band spike, not a plateau

Restricting the match set to `durability`, `performance`, `warmth`, `weather` (the generic
four neutralised):

| weight | TechnicalScore | MRR |
| --- | --- | --- |
| 0.0 | 0.8963 | 0.794 |
| 0.05 | 0.8964 | 0.795 |
| 0.10 | 0.8970 | 0.796 |
| 0.15 | 0.8969 | 0.796 |
| 0.20 | 0.8971 | 0.797 |
| **0.25** | **0.8982** | **0.802** |
| 0.30 | 0.8964 | 0.796 |
| 0.35 | 0.8955 | 0.793 |

HitRate is flat at 0.975 across the whole range; all movement is MRR; MTTC is unchanged.
The best cell is +0.0019 over baseline, but it is a single point: 0.10 to 0.20 sit inside
the +/-0.001 noise band and 0.30 is already back at baseline. Picking 0.25 would be fitting
the argmax on 160 sessions, exactly the error the popularity sweep (consolidation 4.2)
avoided. `artifacts/sweep_profile_tags_fine.json`.

### Outcome

Nothing shipped. The prototype term, its config field and its tests were removed once the
sweep showed no gain the noise band could not explain. The official score is unchanged at
0.892323, exactly as it was before the experiment.

The honest reading: the profile's soft tags are useful evidence for *retrieval* (removing
them from the query costs 0.037) but not for *ranking* a shortlist retrieval has already
narrowed, where the verbatim phrase overlap dominates. Anyone revisiting this should start
from the discriminating-tags-only table above, not from the full set, and would need a
`rerank_depth` re-sweep alongside it (consolidation 4.4).
