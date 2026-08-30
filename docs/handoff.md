# Handoff: what each role picks up

Written overnight after day 1. Read your own section, then `phase0-findings.md` for the
evidence behind any claim you want to challenge.

## Where things stand

The pipeline runs end to end through the official command and scores **0.7511** on all 200
public sessions against a 0.107 baseline, with placeholder modules in all three of your
slots. That number is a starting line, not a result. It says the scaffolding is sound and
the headroom is now measurable.

It got there in two steps. The scaffold scored 0.5420 on its own. Then the sweep harness
found that `weight_popularity`, shipped at 0.15, was badly undertuned. See the curve under
role 3.

```bash
pip install -r requirements.txt
pytest                                  # 63 tests, ~2s
python evaluation/run_eval.py           # your baseline to beat, 160 train sessions
```

Current per-scenario breakdown, all 200 through the official harness:

| scenario | n | hit@10 | mrr | mttc |
| --- | --- | --- | --- | --- |
| buying | 80 | 0.850 | 0.616 | 4.04 |
| browsing | 80 | 0.875 | 0.604 | 4.10 |
| intent_override | 30 | 0.900 | 0.768 | 4.87 |
| boundary | 10 | 0.600 | 0.533 | 5.60 |

**Boundary is now the weakest bucket at 0.600**, but it is only 10 sessions, so it is 5% of
the score and mostly noise. Do not spend a day there.

**MTTC 4.265 is the biggest remaining block.** Efficiency is 0.674 out of a possible 1.0,
worth 0.20 of the score. Getting MTTC to 2.5 would add roughly 0.035. That is role 2.

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

`src/retrieval/baseline.py`. You own **HitRate@10, weight 0.50**. Currently 0.66.

The ceiling is 67% at depth 10 and 100% at depth 1000, so the candidates are already there.
Your job is ordering them better, not finding more.

Two levers are already wired into `Config` and completely unused by the v0 module:

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
