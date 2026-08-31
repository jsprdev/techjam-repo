# Handoff: what each role picks up

Written overnight after day 1. Read your own section, then `phase0-findings.md` for the
evidence behind any claim you want to challenge.

## Current branch update: field-aware retrieval

This update supersedes the statements below that describe retrieval as untouched or its
field weights as unused. The original handoff remains as the day 1 baseline.

Branch `codex/field-aware-retrieval`, commit `517ef30`, now has:

- Separate TF-IDF indexes for title, features, categories, description, store, and details.
- Runtime field weighting plus `weight_details` in `Config`.
- Optional phrase evidence passed from `Slots` to retrieval, with a soft exact-phrase boost
  before candidate truncation.
- A backward-compatible retriever seam, sweep-cache support, and focused regression tests.

Validation passed: 51 tests, offline verification, and the query-degeneracy check.

The branch's full public-set result is **0.8487 TechnicalScore**, with 0.9650 HitRate@10,
0.6416 MRR, and 2.31 MTTC. The previous pooled-retrieval result was 0.7889, 0.8600,
0.7367, and 4.11 respectively.

Interpret this correctly: candidate coverage and conversion speed improved substantially,
but MRR regressed. The branch should be pushed to unblock downstream work, but it should
not merge into `main` as the final retrieval configuration. The next diagnostic should trace
the target's retrieval position against its final position. If retrieval already places it
high and reranking drops it, tune Role 3's popularity and phrase calibration. If retrieval
places it low, preserve the new candidate pool while using a more precision-oriented lexical
ordering for the shortlist.

## MRR diagnostic checkpoint

`evaluation/rank_diagnostics.py` is now available for a train-only attribution run. It
enables detailed traces only for that local run and joins those traces to official evaluator
results. The Agent never receives target information and its normal response path does not
construct these diagnostic payloads.

With `retrieval_exact_phrase_boost=0.0` on all 160 train sessions, 144 sessions converted.
The reranker moved the target from mean retrieval position 33.26 to mean final position
1.97: 118 promotions, three demotions, and 23 unchanged. In rank-one misses, the leading
candidate's average advantage was +0.2136 from popularity but only +0.0442 from retrieval;
rating and phrase evidence were neutral. Do not lower the popularity prior globally: the
earlier sweep showed it improves overall quality. The next scoped test is a small
constraint-specificity reranker increment for close candidates, evaluated only on train.

## Where things stand

The pipeline runs end to end through the official command and scores **0.7889** on all 200
public sessions against a 0.107 baseline, with placeholder modules in all three of your
slots. That number is a starting line, not a result. It says the scaffolding is sound and
the headroom is now measurable.

It got there in two steps. The scaffold scored 0.5420 on its own. Then the sweep harness
found that `weight_popularity`, shipped at 0.15, was badly undertuned. See the curve under
role 3.

```bash
pip install -r requirements.txt
pytest                                  # 44 tests, ~2s
python evaluation/run_eval.py           # your baseline to beat, 160 train sessions
```

Current per-scenario breakdown, all 200 through the official harness:

| scenario | n | hit@10 | mrr | mttc |
| --- | --- | --- | --- | --- |
| buying | 80 | 0.875 | 0.780 | 3.74 |
| browsing | 80 | 0.875 | 0.661 | 3.90 |
| intent_override | 30 | 0.900 | 0.900 | 4.80 |
| boundary | 10 | 0.500 | 0.500 | 6.60 |

**Boundary is the weakest bucket at 0.500**, but it is 10 sessions, 5% of the score, and
mostly noise. Do not spend a day there.

**MTTC 4.105 is the biggest remaining block.** Efficiency is 0.690 out of a possible 1.0,
worth 0.20 of the score. Getting MTTC to 2.5 would add roughly 0.032, and it is the one
number no lever has moved much yet. That is role 2, and it is now the best-value work left.

**Retrieval is now implemented but not promoted.** The field-aware branch uses title,
feature, category, description, store, and details scores separately, then boosts disclosed
whole phrases before truncation. It produced a stronger overall score but a lower MRR, so
ranking precision is now the blocking issue rather than candidate coverage.

## One thing to know before you write a test

`Agent.respond` wraps every turn in a blanket `except Exception` and degrades to a
popularity-ordered guess. That is deliberate, since a raise forfeits the session, and it
means **a test whose only assertion is "the response was well formed" passes on a completely
dead pipeline**. Three of my own checks were vacuous for exactly this reason before an
adversarial review caught them.

So: use the `strict_agent` fixture for anything asserting your module works. It re-raises
instead of degrading. Assert something the fallback cannot produce, such as a query-specific
winner, rather than just contract legality. `tests/test_agent.py` has worked examples.

## Rules that bind everyone

Found by auditing the kit. Violating any of these is expensive and silent.

1. **`reset()` must never raise.** The evaluator wraps `respond()` in try/except but calls
   `reset()` bare. One exception there aborts all 200 sessions, not one turn.
2. **`respond()` must never raise, and must return a dict whose `message` is a `str`.** If
   either fails, the evaluator discards the whole response including your recommendations.
3. **Never emit an `ask_attribute` outside the ten-value enum.** The local evaluator silently
   coerces an unknown value to `other`, but the contract enum is closed and the private
   harness may validate it.
4. **Ranking is array order.** The `score` field is never read by the evaluator. Sort before
   returning.
5. **Never edit anything under `techjam-conversational-search/evaluator/`.** The submission
   rules forbid it and `tests/test_entry_point.py` fails if you do.
6. **No network, ever, on the critical path.** Run `python evaluation/verify_offline.py`
   before you push. It runs a ten turn session with every socket poisoned.
7. **Turn 10 is a full scoring turn.** The hit check runs before the loop breaks, so never
   return a terminal message with an empty list.

## Role 1: Retrieval

`src/retrieval/baseline.py`. You own **HitRate@10, weight 0.50**. The field-aware checkpoint
is on `codex/field-aware-retrieval`; the work below describes the original v0 starting point.

The ceiling is 67% at depth 10 and 100% at depth 1000, so the candidates are already there.
Your job is ordering them better, not finding more.

The v0 module left two levers unused. The field-aware branch now implements both as a
separate per-field index and retrieval-stage phrase evidence:

- `weight_title`, `weight_features`, `weight_categories`, `weight_description`,
  `weight_store`. Right now every field is pooled into one bag of words, so a word in a care
  instruction counts the same as one in the title.
- `exact_phrase_boost`. This is the big one. The simulated customer's utterances are literal
  substrings of the target product's own `features` and `details`. A whole-phrase match is
  far stronger evidence than the sum of its tokens, and nothing currently exploits that.

If you change what is read at index build time, add the field to `INDEX_FIELDS` in
`evaluation/sweep.py` or the sweep will serve you a cached index and report a wrong number.

Done when recall@10 is above 75% and recall@1 above 55%, measured on train only.

## Role 2: Dialogue

`src/state/slots.py`. You own **Efficiency and MTTC, weight 0.20**. Currently 0.456 at
MTTC 6.45. This is the largest single block of unclaimed score.

Three things, in order of value:

1. **Question policy.** `pick_attribute()` currently cycles through attributes by catalog
   coverage. The evaluator's `customer_reply` returns up to two undisclosed constraints whose
   `classify_constraint` bucket matches what you asked. `classify_constraint` is a plain
   keyword matcher over a fixed word list, so you can model it exactly rather than guessing.
   Asking an attribute the target has no constraint for wastes the information, though never
   the turn, because asking is free.
   - Note `attribute == "other"` matches **any** undisclosed constraint. That makes it a
     strictly dominant ask under this simulator. `Config.allow_other_fallback` gates it.
     Measure both, and be honest in the writeup about which we shipped and why, because "ask
     the wildcard every turn" is effective and also a worse product.
2. **Override.** `intent_override` sessions flip at turn 3 or 4 with a message beginning
   "Actually, ignore my earlier preference." Hits before the flip are **not counted**, so
   those 30 sessions cannot be won earlier no matter what. Worse, accumulating through the
   flip actively hurts: the old constraint is now wrong and needs erasing. Nothing currently
   detects this.
3. **Decay.** `Config.slot_decay` is unused. A turn one inference should not weigh the same
   at turn eight as something just stated.

Also yours: `_truncation_width()` in `src/agent.py` is a two-line placeholder standing in for
real Buying versus Browsing routing.

Done when MTTC is below 4 and override sessions hit within two turns of the flip.

## Role 3: Ranking and agent shell

`src/rank/baseline.py` and `src/agent.py`. You own **MRR, weight 0.30**. Currently 0.403.

HitRate is 0.66 but MRR is 0.40, so the target is often in the ten and not at the top. That
gap is your whole job.

**The popularity prior is already swept, and it was the single highest-leverage constant in
the system.** Full curve over the 160 train sessions:

| weight_popularity | score | hit@10 | mrr | mttc |
| --- | --- | --- | --- | --- |
| 0.0 | 0.4608 | 0.550 | 0.366 | 7.21 |
| 0.4 | 0.6313 | 0.787 | 0.402 | 5.16 |
| 1.0 | 0.7244 | 0.844 | 0.577 | 4.53 |
| 1.5 | **0.7435** | 0.850 | 0.625 | 4.46 |
| 2.0 | 0.7401 | 0.850 | 0.611 | 4.41 |
| 3.0 | 0.7391 | 0.856 | 0.595 | 4.38 |
| 5.0 | 0.7176 | 0.831 | 0.578 | 4.57 |

Unimodal, plateauing across 1.5 to 3.0. I set the default to **2.0**, the middle of the
plateau rather than its exact argmax, because those three points differ by under 0.005 and
picking the argmax on 160 sessions is fitting noise. It is marked PROVISIONAL in
`config.py` and it is yours to confirm or override.

I also checked it is not a metric-gaming artefact: `evaluation/check_degeneracy.py` shows
five unrelated queries still return completely disjoint top tens even at weight 20, because
the prior only reorders a shortlist retrieval has already filtered. Rerun it if you change
how ranking composes.

### `rerank_depth`: a trade-off I deliberately did not resolve

Swept over the same 160 train sessions. Composite and MRR peak in different places.

| rerank_depth | score | hit@10 | mrr | mttc |
| --- | --- | --- | --- | --- |
| 25 | 0.6216 | 0.681 | 0.604 | 6.01 |
| 50 | 0.6899 | 0.769 | **0.621** | 5.03 |
| 100 (shipped) | 0.7401 | 0.850 | 0.611 | 4.41 |
| 200 | **0.7515** | 0.875 | 0.558 | 3.67 |
| 400 | 0.7438 | 0.881 | 0.504 | 3.39 |

Deeper reranking applies the popularity prior to more candidates, which pulls more plausible
items into the top ten and shortens sessions, but pushes the exact target down *within* those
ten. HitRate and MTTC improve while MRR degrades.

**I left the default at 100.** Moving to 200 buys +0.011 composite, which is close to noise
on 160 sessions, and costs 0.053 MRR, which is not. MRR is 30% of the score and it is yours,
so the call is yours. Rerun with `--split train` after any ranking change, because the shape
of this curve depends entirely on how ranking composes.

Worth knowing when you decide: `rerank_depth` is capped in practice by the truncation width,
which is 200 on the Buying track and 800 on Browsing, so a depth above 200 only affects
Browsing turns.

Features Phase 0 says are available and unused: exact phrase overlap between disclosed
constraints and product text, category path agreement, `average_rating`, and price band.

The shell is already hardened (never raises, never exceeds the cap, always returns a list,
contract-validated). Keep it that way. `_phrase()` is a placeholder and is the visible
surface in the demo video, so it is worth twenty minutes late on.

Any LLM rerank stays behind `Config.use_llm`, default off, with the deterministic ordering as
the fallback. The organiser may score us with networking disabled.

Done when MRR is above 0.45, zero exceptions across 200 sessions, and
`evaluation/verify_offline.py` passes.

## Role 4: Platform, already built

`evaluation/`. Eval wrapper with per-scenario breakdown and traces, stratified held-out
split, config sweep harness, latency and memory instrumentation, offline verification rig,
contract tests. See the README.

The 40 session held-out slice is reserved and stratified. Nobody tunes against it, including
to check. The moment a held-out number informs a decision it stops measuring generalisation.

Remaining for role 4: Devpost text, demo video, and untracking the 58 MB catalog from git.
