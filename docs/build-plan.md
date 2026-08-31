# Build plan: four roles, five days

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

The initial retrieval implementation is committed on branch
`codex/field-aware-retrieval` at `517ef30` and is ready to push as a collaboration
checkpoint. It is not ready to merge into `main` as the final retrieval setting.

Completed work:

- Separate in-memory TF-IDF indexes for title, features, categories, description, store,
  and details, blended with the weights in `Config`.
- Optional disclosed phrase evidence passed through the retriever interface and applied as
  a soft, whole-phrase score before candidate truncation.
- Separate retrieval and reranker phrase-boost settings, sweep-cache updates, and focused
  field, phrase, and agent integration tests.
- `pytest` passes with 51 tests. Offline verification and the query-degeneracy check pass.

Full public-set measurement for the branch's configured default:

| Metric | Previous pooled retrieval | Field-aware branch |
| --- | ---: | ---: |
| TechnicalScore | 0.7889 | **0.8487** |
| HitRate@10 | 0.8600 | **0.9650** |
| MRR | **0.7367** | 0.6416 |
| MTTC | 4.11 | **2.31** |

This raises coverage and reaches a converting top ten much earlier, but it lowers MRR by
placing the target farther down the first successful list. The branch is useful as an
integration base for dialogue and ranking work, but the MRR regression fails the promotion
gate below. Do not merge it into `main` until a precision-oriented reranking experiment
preserves the coverage gain without reducing MRR.

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

The first deliberately narrow experiment—a small, deterministic score for longer exact
constraints—was swept from 0 through 4 on an initial 40 training sessions. It did not
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

Owns `src/retrieval/`, `src/catalog.py`. The index and the query. Weighted matching over
title, features, description, categories and store, with heavy boosting for exact phrase
hits, because the simulated customer speaks in phrases copied verbatim out of the target
product's own record. Also owns the query builder and the truncation width that narrows on
Buying and widens on Browsing.

The field-aware index and retrieval-stage phrase evidence are now implemented on
`codex/field-aware-retrieval`. The remaining retrieval task is precision calibration: keep
the candidate coverage gain while handing Role 3 a shortlist whose strongest lexical and
phrase matches rank first.

- Day 1: ship a working retriever before tuning anything, so roles 2 and 3 are unblocked.
- Done when: recall@10 above 75% and recall@1 above 55% with full constraints, on the
  training slice only.
- Never: dense embeddings, external vector stores, anything that downloads weights.

## Role 2: Dialogue

Owns `src/state/`, `src/policy/`. Slots that accumulate, get overridden and decay. Buying
versus Browsing routing per turn. Above all, choosing which attribute to ask each turn so
constraints come out as fast as possible. The hard ten turn cap lives here and must be
enforced in our code, never left to the evaluator.

Know this before designing the policy:

- The evaluator classifies constraints with a plain keyword matcher (`classify_constraint`),
  so model the matcher, not the concept.
- Boundary sessions refuse the first question once.
- Override sessions flip at turn 3 or 4, and hits before the flip are not counted. Those
  thirty sessions have a floor of three turns.

- Done when: MTTC below 4 overall, and override sessions hit within two turns of the flip.
- Never: hard filtering. Demote non-matching items, never delete them.

## Role 3: Ranking and agent shell

Owns `src/agent.py`, `src/rank/`, `src/language/`. Two jobs. First the shell: the `Agent`
class the evaluator imports, schema conformance, usage reporting, and a guarantee that
nothing ever raises or runs past turn ten. A crash costs a whole session. Second the
reranker, which turns 67% at rank ten into a high score at rank one using deterministic
features: exact phrase overlap, category path agreement, popularity, rating, price band. An
optional LLM rerank sits behind a flag that defaults to off.

- Priority: a valid, never-crashing agent running end to end on day 1.
- Done when: zero exceptions across all 200 sessions, MRR above 0.45, and the whole system
  runs with the network unplugged.
- Never: make any LLM call mandatory, or put one on the critical path.

## Role 4: Platform and measurement

Owns `evaluation/`, `src/config.py`, `src/interfaces.py`, `requirements.txt`, `README.md`.
Built first because the other three cannot tune without it. Five technical deliverables:

1. One command evaluation wrapper with per-scenario breakdown and per-session traces.
2. A config sweep harness, so tuning is systematic rather than three people guessing.
3. Instrumentation for latency, token usage and memory, all required disclosures.
4. A contract test running the agent against the official evaluator on every merge.
5. The offline verification rig proving the system survives with the network pulled.

The submission stacks on top, and stacks cleanly because this person already holds every
number the writeup needs to quote. Platform work front-loads onto days 1 and 2, which is
what makes room for the writeup on days 4 and 5.

- Hold out 40 of the 200 sessions on day 1. Nobody tunes against them.
- Done when: any teammate runs one command and gets a per-scenario table, a clean clone
  reproduces the score, and the submission bundle exists 24 hours before the deadline.

## Five days

| Day | Focus | Gate |
| --- | --- | --- |
| 1 | Freeze the seams, unblock everybody | A full 200 session run produces a number |
| 2 | First real system | Score beats the 0.107 baseline |
| 3 | Tune against real numbers | Score above 0.40, every scenario bucket examined |
| 4 | Feature freeze at midday, integration and robustness | Zero crashes, verified offline run |
| 5 | Package and record | Submitted with time to spare |

## Interfaces frozen on day 1

Four people editing one package for five days is where teams lose their weekend. Module
ownership plus these signatures, agreed before anyone writes logic, is what prevents it.
Anyone may propose a change, nobody changes one silently. Definitions live in
`src/interfaces.py`.

- `Candidate`, a `parent_asin` and a score. The only currency between modules.
- `Retriever.retrieve(query, k) -> list[Candidate]`, owned by role 1.
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
