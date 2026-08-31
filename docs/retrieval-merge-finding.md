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

---

## Dual-track routing: does the second track change anything?

The router in `policy/intent.py` classifies every turn Buying or Browsing and scores 1.000
against the session label on turn one (`evaluation/intent_audit.py`). This section asks the
harder question: once it has classified, does the classification change the output?

**Measured answer: no, and every attempt to make it change the output made the score worse.**

### The router's decision is currently inert

The two tracks differ in `Track.width`, the retrieval truncation: 200 for Buying, 800 for
Browsing. But `track_depth_browsing` and `track_depth_buying` both default to 200, so the
ranker only ever rescores the top 200 candidates. The 600 extra items a Browsing turn
retrieves are thrown away before anything looks at them.

Two sweeps confirm it. Neither moves any metric by any amount:

| `truncate_browsing` | TechnicalScore | HitRate@10 | MRR | MTTC |
| --- | --- | --- | --- | --- |
| 200 | 0.9201 | 0.983 | 0.848 | 2.30 |
| 800 | 0.9201 | 0.983 | 0.848 | 2.30 |

(60 train sessions. `artifacts/sweep_route_inert.json`.)

| `intent_constraint_weight` | TechnicalScore | HitRate@10 | MRR | MTTC |
| --- | --- | --- | --- | --- |
| 0.0 | 0.8969 | 0.975 | 0.796 | 2.48 |
| 0.05 | 0.8969 | 0.975 | 0.796 | 2.48 |
| 0.2 (shipped) | 0.8969 | 0.975 | 0.796 | 2.48 |

(160 train sessions. `artifacts/sweep_intent_constraint.json`.) The constraint signal is the
heaviest input to the router, and zeroing it changes nothing downstream, which is the same
result seen from the other side.

### Making the track matter costs score

The obvious repair is to let a Browsing turn actually use its wider pool, by raising the
depth the ranker rescores on Browsing turns only. This is what per-track configuration is
for and it had never been swept.

| `track_depth_browsing` | TechnicalScore | HitRate@10 | MRR | MTTC |
| --- | --- | --- | --- | --- |
| **200 (shipped)** | **0.8969** | 0.975 | **0.796** | 2.48 |
| 400 | 0.8858 | 0.975 | 0.752 | 2.37 |
| 800 | 0.8859 | **0.981** | 0.737 | 2.29 |

(160 train sessions. `artifacts/sweep_track_depth_browsing.json`.)

Widening costs about 0.011. MRR falls from 0.796 to 0.737 while MTTC improves from 2.48 to
2.29, and arithmetically that trade is a loss: MRR carries 0.30 and Efficiency 0.20.

### What this means, stated plainly

The optimal configuration gives both tracks the same rerank depth, so the router's output
does not currently change the shortlist the customer sees. We would rather say that than
leave a traceability table implying otherwise.

Two things are worth keeping from it anyway. The router itself is correct and measured, and
it is the seam any track-specific behaviour would attach to, at the cost of a config change
rather than a rewrite. And the depth-800 row is the same shape as the field-aware finding
above: **HitRate rises to 0.981**, the best any configuration has produced, while MRR falls.
The wider Browsing pool genuinely contains targets the narrow pool misses. Nothing we have
ranks them well enough to profit. That is the same open thread as the field-aware route, and
it is where the remaining score is: browsing MRR is 0.675 against buying's 0.828.

---

## Where the remaining MRR actually goes

`evaluation/rank_diagnostics.py` had never run. It called `run_eval.run()` with an argument
that does not exist, and it read two trace keys the agent never emitted, so it raised before
producing a number either way. Both are fixed and the trace now carries the per-candidate
score breakdown, gated on the trace sink so a scored run allocates nothing extra.

With it working, the picture on 160 train sessions is specific.

### The ranking stage is doing most of the work

| | |
| --- | --- |
| Mean rank of the target when retrieval returns it | **56.5** |
| Mean rank after ranking | **1.74** |
| Sessions where ranking promoted the target | 125 of 156 |
| Sessions where ranking demoted it | **2** |
| Unchanged | 29 |

Retrieval finds the target and puts it around rank 56. Ranking moves it to 1.74, and it makes
things worse in 2 sessions out of 156. That is the answer to "is the LLM Semantic Ranking
stage load bearing", and it is worth more than an architecture diagram.

### When the target loses, it loses on popularity alone

For sessions where the target converted but not at rank one, the mean advantage the item
above it held, per score term:

| Term | Leader's advantage |
| --- | --- |
| **popularity** | **+0.264** |
| appeal (offline LLM prior) | +0.008 |
| rating | +0.000 |
| retrieval | -0.011 |
| **phrase** | **-0.010** |

Negative means the target was ahead on that term. So in the losing cases the target is
**equal or better on both phrase evidence and retrieval similarity**, and loses purely
because something above it is more popular.

Per scenario, the tie is even cleaner:

| Scenario | converted | target at rank 1 | displaced | popularity advantage | phrase advantage |
| --- | --- | --- | --- | --- | --- |
| buying | 63 | 51 | 12 | +0.314 | 0.000 |
| **browsing** | 62 | **36** | **26** | +0.230 | **0.000** |
| intent_override | 23 | 20 | 3 | +0.081 | 0.000 |

Browsing loses the target from rank one 26 times against buying's 12, on the same HitRate.
And in every scenario the phrase advantage is **exactly zero**: the displacing item matched
the customer's quoted phrases just as well as the target did.

### What that implies

The phrase signal has run out of resolution. It is the strongest term we have and by the time
a session is losing, it has already tied. Popularity then decides, and popularity is not a
signal about this customer.

That is not an argument for lowering the popularity weight: it was swept from 0.0 to 5.0 and
zeroing it costs 0.43, because in the many sessions the target does win, popularity is what
wins it. It is an argument for a **tie-breaker that only fires when phrase evidence ties**,
which is precisely the case a model reading the conversation is suited to and a bag of words
is not.

So the live reranker in `src/language/rerank.py` is aimed at a real, measured gap rather than
at the brief's wording. The honest caveat stays: it has never been run against a key, it costs
$0.64 for a full 200-session run, and it may still lose, because this simulator's customer
speaks in verbatim catalog substrings. But the case for trying it is now a measurement.
