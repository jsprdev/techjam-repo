# Build plan: current checkpoint

Companion to `phase0-findings.md`, which carries the evidence for every claim here.
Rendered version: the team artifact linked in the project chat.

## What Phase 0 changed

Three measurements invalidate large parts of the original spec. Each one deletes work
somebody would otherwise have done.

1. **Retrieval is not the bottleneck.** Plain TF-IDF finds the target inside the top 1000 in
   100% of sessions and inside the top 10 in 67% once every constraint is known. No dense
   embeddings, no vector index, no LLM catalog extraction.
2. **The graded run may have no network.** `docs/submission_rules.md` warns that organiser
   policy may disable network access under CPU, memory and timeout limits. The deterministic
   engine is the product, not a fallback. Weights cannot be downloaded at run time.
3. **Asking is free.** The evaluator checks `recommendations` for a hit before it reads
   `ask_attribute`, so one turn carries both. There is no ask-versus-recommend tradeoff.

## Current implementation checkpoint: field-aware retrieval

The retrieval checkpoint is on `codex/field-aware-retrieval` and is ready to push for
collaboration. It is not ready to merge into `main` as the final retrieval setting.

Completed work:

- Separate in-memory TF-IDF indexes for title, features, categories, description, store,
  and details, blended with the weights in `Config`.
- Optional disclosed phrase evidence passed through the retriever interface and applied as
  a soft, whole-phrase score before candidate truncation.
- Separate retrieval and reranker phrase-boost settings, sweep-cache updates, and focused
  field, phrase, and agent integration tests.
- 54 tests, offline verification, and the query-degeneracy check pass.

Full public-set measurement for the branch's configured default:

| Metric | Previous pooled retrieval | Merged retrieval branch |
| --- | ---: | ---: |
| TechnicalScore | 0.7889 | **0.849765** |
| HitRate@10 | 0.8600 | **0.9650** |
| MRR | **0.7367** | 0.630883 |
| MTTC | 4.11 | **2.10** |

This raises coverage and reaches a converting top ten much earlier, but it lowers MRR by
placing the target farther down the first successful list. The merged ask policy and
`rerank_depth=200` improve MTTC further, but MRR remains the active precision problem.

## MRR diagnostic checkpoint

The branch also contains a train-only ranking diagnostic (`evaluation/rank_diagnostics.py`).
It records the retrieval shortlist and deterministic reranker components only when
explicitly requested by the offline evaluator; it does not change the public `Agent`
response or the production scoring path.

On the 160-session training split with retrieval phrase boosting set to zero (the
strongest comparable configuration tested), the target appeared in a converting top ten
in 144 sessions. Reranking improved its mean position from 33.26 in the retrieval
shortlist to 1.97, promoting it in 118 sessions and demoting it in only three. Where the
target still finished below first, the leading candidate's mean advantage was dominated by
the popularity prior (+0.2136); retrieval contributed only +0.0442 and rating and phrase
components were effectively neutral.

The first deliberately narrow experiment, a small deterministic score for longer exact
constraints, was swept from 0 through 4 on an initial 40 training sessions. It did not
change any rank or metric, so it was rejected and not retained in the scoring path. Do not
globally reduce popularity, add a second catalog-wide index, or inspect the holdout before
a promising feature is found. The latter would raise the current approximately 937 MB peak
memory without an announced memory limit.

## Score ownership

| Role | Owns | Weight |
| --- | --- | --- |
| 1. Retrieval | HitRate@10 | 0.50 |
| 2. Dialogue | Efficiency / MTTC | 0.20 |
| 3. Ranking and agent shell | MRR | 0.30 |
| 4. Platform and measurement | every number above, plus the submission | 65% of judging |

## Role 1: Retrieval

Owns `src/retrieval/`, `src/catalog.py`. The index uses weighted matching over title,
features, description, categories, store, and details. `SlotState` owns query construction;
the agent chooses the Buying/Browsing truncation width.

The field-aware index and retrieval-stage phrase evidence are implemented. The remaining
task is precision calibration: keep the candidate coverage gain while handing Role 3 a
shortlist whose strongest lexical and phrase matches rank first.

- Retain a retrieval change only when it meets the checkpoint acceptance gate on the training
  slice and preserves the offline, memory, and interface constraints.
- Never: dense embeddings, external vector stores, anything that downloads weights.

## Role 2: Dialogue

Owns `src/state/`. Slots accumulate, get overridden and decay; the agent routes Buying versus
Browsing per turn. The current branch already reaches 2.10 overall MTTC, so dialogue is not
the immediate blocker. Preserve the ten-turn cap and do not trade away the retrieval gain.

Know this before designing the policy:

- The evaluator classifies constraints with a plain keyword matcher (`classify_constraint`),
  so model the matcher, not the concept.
- Boundary sessions refuse the first question once.
- Override sessions flip at turn 3 or 4, and hits before the flip are not counted. Those
  thirty sessions have a floor of three turns.

- Measure any dialogue change against the train split and retain it only if it preserves the
  current retrieval and ranking safeguards.
- Never: hard filtering. Demote non-matching items, never delete them.

## Role 3: Ranking and agent shell

Owns `src/agent.py`, `src/rank/`, `src/language/`. The shell must remain contract-safe and
the reranker is the active workstream: it needs to preserve the retrieval coverage gain
while improving rank-one precision. Current diagnostic evidence points to popularity winning
otherwise-close comparisons; inspect field-level constraint matches before adding a narrow,
soft tie-breaker. An optional LLM rerank remains disabled by default.

- Done when: a train-validated configuration meets the checkpoint acceptance gate without
  lowering overall HitRate@10 or MRR, and the system remains offline-safe.
- Never: make any LLM call mandatory, or put one on the critical path.

## Role 4: Platform and measurement

Owns `evaluation/`, `src/config.py`, `src/interfaces.py`, `requirements.txt`, `README.md`.
The required measurement stack is already in place:

1. One command evaluation wrapper with per-scenario breakdown and per-session traces.
2. A config sweep harness, so tuning is systematic rather than three people guessing.
3. Instrumentation for latency, token usage and memory, all required disclosures.
4. A contract test running the agent against the official evaluator on every merge.
5. The offline verification rig proving the system survives with the network pulled.

Keep the 40-session held-out slice reserved until feature freeze. Platform work is complete
when a clean clone reproduces the score and the submission bundle is ready.

## Current interfaces

Definitions live in `src/interfaces.py`. Preserve these seams unless a coordinated change is
needed across the owning modules.

- `Candidate`, a `parent_asin` and a score. The only currency between modules.
- `Retriever.retrieve(query, k, phrases=()) -> list[Candidate]`, owned by role 1.
- `SlotState.to_query() -> str`, owned by role 2. The one seam where dialogue feeds retrieval.
- `Ranker.rank(candidates, slots, profile) -> list[str]`, owned by role 3.
- `Agent.respond(session_id, user_message, turn, top_k) -> dict`, fixed by the organiser.
  Nobody may change this one at all.

## Do not build these

| Ruled out | Because |
| --- | --- |
| Dense retrieval | Recall is already 100% at depth 1000, and weights cannot be downloaded offline |
| LLM catalog extraction | The evaluator never reads the fields it would clean, and they are under 5% populated |
| Reinforcement learning | 200 sessions cannot support policy learning without fitting our own simulator |
| Hard constraint filters | A filter that drops the target is unrecoverable. Demote instead |
| Cross-session profiling | The rules define every session as isolated. No eval surface |
| Varying list length | Gaming MRR against MTTC reads as metric gaming to a human judge |
