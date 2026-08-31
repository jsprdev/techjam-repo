# Handoff: current checkpoint

Read `CLAUDE.md` first. This document records the current branch state and the
next evidence-backed work; `phase0-findings.md` contains the original measurements.

## Branch state

`codex/field-aware-retrieval` is ready to push as a collaboration checkpoint.
Its relevant commits are:

- `517ef30` — field-aware lexical retrieval
- `79f5ad0` — train-only MRR diagnostics
- `f42b9c4` — rejected specificity-experiment record

The default system is deterministic, in-memory, read-only against the catalog, and has no
network dependency on its critical path. It passes 52 tests and
`evaluation/verify_offline.py`. Run `evaluation/check_degeneracy.py` after changing how
ranking composes.

## What changed in retrieval

The pooled TF-IDF retriever was replaced with independent indexes for `title`, `features`,
`categories`, `description`, `store`, and `details`. Their cosine scores are blended using
the `Config` weights. `Agent` also passes whole disclosed slot constraints into
`Retriever.retrieve(..., phrases=())`; matching a sufficiently long phrase adds soft
evidence before candidate truncation. The public interface remains backward compatible.

The full public-set result for the configured default is:

| Metric | Pooled baseline | Field-aware branch |
| --- | ---: | ---: |
| TechnicalScore | 0.7889 | **0.8487** |
| HitRate@10 | 0.8600 | **0.9650** |
| MRR | **0.7367** | 0.6416 |
| MTTC | 4.11 | **2.31** |

This is a strong retrieval and efficiency checkpoint, but it is not ready to replace
`main` as the final configuration because its MRR is lower. Its measured peak memory is
about 937 MB; the rules mention possible memory restrictions but publish no numeric cap.

## MRR diagnosis and next step

`evaluation/rank_diagnostics.py` is a train-only tool. It enables detailed local traces,
then compares retrieval and reranking only after the official evaluator has finished. The
Agent never receives a target ASIN or diagnostic payload in its normal response path.

With `retrieval_exact_phrase_boost=0.0` on 160 training sessions, 144 sessions converted.
Reranking improved mean target position from 33.26 in the shortlist to 1.97, with 118
promotions and only three demotions. In remaining rank-one misses, the leader's mean score
advantage was mostly popularity (+0.2136), rather than retrieval (+0.0442), rating, or
whole-phrase evidence.

Do not reduce popularity globally: prior train sweeps show it improves overall quality. A
small longer-phrase specificity increment was tried at weights 0–4 on 40 training sessions;
it changed no rank or metric and was removed. The next step is to extend the diagnostic to
compare the target and leader's **field-level constraint matches**, then test only a
field-specific tie-breaker supported by that evidence. Use the 160-session train split to
confirm a promising variant; do not inspect the 40-session holdout before feature freeze.

## Guardrails

1. Preserve the public `Agent` API and the `Retriever.retrieve(query, k, phrases=())` seam.
2. Do not edit `techjam-conversational-search/evaluator/` or mutate the catalog.
3. Keep all retrieval and ranking evidence soft—never hard-filter candidates.
4. `reset()` and `respond()` must never raise; only valid, unique ASINs may be recommended.
5. Ranking is recommendation-array order. The evaluator does not read a score field.
6. Keep the critical path offline and deterministic. Do not add a model download, vector
   store, or second catalog-wide index without first resolving the memory budget.
7. Use `strict_agent` for tests that assert behaviour rather than response shape; the normal
   agent intentionally falls back to a legal popularity list after errors.

## Role 1: Retrieval

Owns `src/retrieval/baseline.py` and contributes **HitRate@10 (weight 0.50)**. Phase 0 found
that plain TF-IDF reaches 100% target recall at depth 1000 once all constraints are known;
the job is therefore candidate ordering and useful truncation, not a denser index or a wider
catalog search.

The field-aware checkpoint implements the two formerly missing levers: independent field
matrices and soft phrase evidence before truncation. All matrices are built independently of
their runtime blending weights, so `evaluation/sweep.py` can reuse the index when only
weights or phrase boosts change (`INDEX_FIELDS` is intentionally empty). If a future change
alters the indexed data or representation, revisit that cache decision before trusting a
sweep.

The remaining Role 1 concern is shortlist precision, coordinated with Role 3. Never solve it
by hard-filtering or by adding an external/dense retriever: both violate the recovery and
offline constraints, and a second full-catalog index is risky at the current memory level.

## Role 2: Dialogue

Owns `src/state/slots.py` and **Efficiency / MTTC (weight 0.20)**. The field-aware checkpoint
already reaches 2.31 overall MTTC, but the following evaluator behaviour remains important
for any dialogue work:

1. `customer_reply` supplies up to two undisclosed constraints whose
   `classify_constraint` bucket matches the requested attribute. That classifier is a fixed
   keyword matcher, so model its actual behaviour rather than an abstract attribute model.
2. Asking is free in the evaluator: recommendations are checked before `ask_attribute`.
   `other` can match any undisclosed constraint, but it is gated by
   `Config.allow_other_fallback`; measure and disclose its use rather than assuming it is
   automatically desirable.
3. Intent-override sessions flip at turn 3 or 4. Hits before the flip do not count, and
   retaining superseded constraints can hurt ranking. Treat the new preference as evidence
   that needs explicit handling rather than ordinary accumulation.
4. `slot_decay` is available for measurement. A recent explicit constraint may deserve more
   influence than an early one, but any decay change must be evaluated with the full pipeline.

Do not hard-filter candidates from inferred slots. The ten-turn cap remains a local
responsibility, not something to delegate to the evaluator.

## Role 3: Ranking and agent shell

Owns `src/rank/baseline.py` and `src/agent.py`, and **MRR (weight 0.30)**. This is where the
MRR fix belongs. Role 1 supplies a broad, high-coverage shortlist; Role 3 reorders it into
the recommendations the evaluator scores.

The popularity prior was the strongest historical deterministic signal and remains useful;
do not remove it globally. Its earlier pre-field-aware sweep plateaued around weights 1.5 to
3.0, which is why the configured default is 2.0. `evaluation/check_degeneracy.py` confirms
that even a strong popularity prior reorders a query-dependent shortlist rather than
replacing relevance.

Historical rerank-depth sweeps also showed a trade-off: deeper reranking can improve
HitRate@10 and MTTC while lowering within-list MRR. Keep `rerank_depth=100` unless a new
train-only measurement beats the current checkpoint acceptance gate; deeper values affect
Browsing more than Buying because truncation is 800 versus 200 candidates.

The next Role 3 task is diagnostic-led: compare field-level constraint matches between the
target and the popular rank-one leader, then trial a small soft tie-breaker only if the
evidence supports it. Keep exact phrase evidence, ratings, and any new signal deterministic.
An LLM rerank may only remain opt-in behind `Config.use_llm`; it cannot be required.

## Role 4: Platform and measurement

Owns `evaluation/`, `src/config.py`, `src/interfaces.py`, `requirements.txt`, and `README.md`.
It provides the official-wrapper evaluation command, scenario metrics, traces, train/holdout
split, config sweep harness, latency and memory reporting, contract tests, and offline probe.

The 40-session holdout is stratified and reserved. Do not tune against it or use it for an
intermediate confidence check: it stops being a generalisation check once it informs a
decision. Role 4 also owns the final reproducibility and submission work, including the
Devpost text, demo material, and submission bundle.

## Evaluation protocol

Tune only on the fixed 160-session train split. A change is accepted only if it improves
train TechnicalScore by at least 0.005, does not lower overall HitRate@10 or MRR, and does
not reduce HitRate@10 by more than 0.025 in a scenario with at least 30 sessions. Record
overall and per-scenario HitRate@10, MRR, MTTC, latency, and memory. The 10-session boundary
bucket should be reported but not used alone to reject a change.

```bash
techjam-conversational-search/.venv/bin/python -m pytest
techjam-conversational-search/.venv/bin/python evaluation/verify_offline.py
techjam-conversational-search/.venv/bin/python evaluation/check_degeneracy.py
techjam-conversational-search/.venv/bin/python evaluation/run_eval.py --split train
```
