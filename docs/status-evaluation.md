# Status evaluation: every requirement, against what exists

Originally written after the main-only score plateaued at 0.8951 on the train split. The
public metrics below now reflect the merged field-aware retrieval branch; its feature-status
audit should be refreshed before using it as a current implementation inventory. It audits
the code against the organiser's brief (`problem-statement.md`), our own build spec
(`techjam-detailed-agent-spec.md`), and the four-role plan (`docs/build-plan.md`).

Verified by reading the code, not the docstrings. A config field nothing reads is MISSING.

---

## 1. Headline

| | |
| --- | --- |
| TechnicalScore | **0.849765** on 200 public sessions, baseline 0.1067 |
| HitRate@10 | 0.965 |
| MRR | 0.630883 |
| MTTC | 2.10 |
| Efficiency | 0.890 |
| Parameter tuning | The main-only sweep is historical and must be revalidated for field-aware retrieval |
| Pillars fully answered | **1 of 4** |

The number and the pillars have come apart. Almost everything remaining closes a named
requirement without moving the score, and that is the correct thing to spend time on: the
score is one input to Technical Execution, which is 35% of judging.

---

## 2. Pillar I, Core Architecture

| Named requirement | Status | Evidence |
| --- | --- | --- |
| Dual-Track Routing, Buying vs Browsing | **PARTIAL** | `agent.py:_truncation_width` is a 2-line heuristic on constraint count. No intent detection |
| Pipeline base: **keyword** retrieval | BUILT | `src/retrieval/baseline.py`, TF-IDF |
| Pipeline base: **category** retrieval | **MISSING** | No category route exists |
| Pipeline base: **vector** similarity | **MISSING** | No dense route exists |
| Pipeline base: **LLM Semantic Ranking** | **MISSING** | `src/language/` does not exist. Zero LLM calls anywhere |
| In-memory execution | BUILT | No external store, about 937 MB peak |

The brief names the pipeline base literally as "Multi-Route Retrieval then LLM Semantic
Ranking". We have one of three routes and none of the ranking stage. **This is the single
most judge-visible gap**, because it is quoted verbatim in the brief and a judge will look
for it by name.

## 3. Pillar II, Dialog Strategy

| Named requirement | Status | Evidence |
| --- | --- | --- |
| Dynamic State Machine, Information Accumulation | BUILT | `slots.observe`, incremental phrases |
| **Intent Override**, slot erasure and rewriting | **MISSING** | Nothing detects the override turn. Note measurement below |
| Over-Generality retrieval cutoff | **MISSING** | `flat_belief_entropy` has zero readers |
| Proactive structured clarification prompts | **PARTIAL** | Attribute choice is real and measured; the wording in `agent.py:_phrase` is a placeholder |

On the merged branch, `intent_override` remains the strongest MRR bucket at hit 0.966667,
MRR 0.840000, and MTTC 3.8333, against a structural floor of turn 3 or 4. Override handling
still closes a named requirement, but its score impact should be re-measured before making a
claim in the writeup.

## 4. Pillar III, Self-Evolution

**This pillar was dropped from `docs/build-plan.md` entirely.** It appears nowhere in the
plan or the handoff. The plan allocated roles by metric (HitRate, MRR, Efficiency,
platform), and Pillar III moves none of them, so it fell out and nobody was assigned it.
The "Do not build" table cut cross-session profiling, correctly, and that single cut was
allowed to stand in for the whole pillar. Spec section 7 warned against exactly this:

> Tier 7.1 is the runtime answer and must exist. A submission whose only adaptation happens
> between eval runs has not answered the pillar.

The honest twist is that three of the four behaviours in spec 7.1 **already run**, built for
score and never named:

| Spec 7.1 requirement | Status | Evidence |
| --- | --- | --- |
| Reliability reweighting: attributes that fail get downweighted | **BUILT** | `slots._exhausted` retires an attribute the customer cannot answer |
| Workflow re-orchestration re-selected each turn | **PARTIAL** | `_truncation_width` recomputes retrieval width every turn from state |
| Personalised context distillation | **BUILT** | `slots.to_query` compresses profile plus dialogue rather than replaying raw history |
| Belief updating as evidence arrives | **MISSING** | There is no belief object. No distribution, no entropy |

So Pillar III is closer to answered than the audit first suggested, but it is invisible:
unnamed in code, unmeasured in the harness, and absent from every document a judge reads.
Naming and measuring what already exists is the highest value work remaining, and costs
about two hours.

## 5. Pillar IV, Evaluation Matrix

| Named requirement | Status |
| --- | --- |
| Coverage, Hit Rate@K | BUILT, `evaluation/run_eval.py` |
| Precision, MRR | BUILT |
| Efficiency, MTTC | BUILT |
| Per-scenario reporting | BUILT |
| Held-out generalisation check | BUILT and **unspent**, 40 stratified sessions never looked at |

The only pillar fully answered. The recall ceiling diagnostic and the held-out slice both go
beyond what the brief asks for and are worth saying so.

## 6. In-scope items the brief names explicitly

| In-scope item | Status |
| --- | --- |
| Intent-detection modules splitting Buying and Browsing | **MISSING**, only a constraint-count proxy |
| Heterogeneous retrieval routing: weights | BUILT, per-field TF-IDF matrices are blended at runtime |
| Heterogeneous retrieval routing: custom dynamic truncation | BUILT, and swept |
| Heterogeneous retrieval routing: **slot decay over time** | **MISSING**, `slot_decay` has zero readers |
| Runtime-adaptive memory for context distillation | BUILT but unnamed, see Pillar III |
| Prompt strategy or local scoring tuning for the ranking stage | PARTIAL, local scoring tuned and swept; no prompt stage exists |

## 7. Hard limits and rules

| Rule | Status |
| --- | --- |
| Max 10 turns, zero if exceeded | Enforced in `agent.py`, tested |
| Catalog read only, no mock ASINs | Respected |
| No UI, backend only | Respected |
| No base model fine-tuning | Respected |
| No external vector DB, in-memory only | Respected |
| Text only, no multimodal | Respected |
| Evaluator files unmodified | Enforced by pinned hashes in `tests/test_entry_point.py` |
| Runs with network disabled | Verified, negative-controlled, `evaluation/verify_offline.py` |

All clean.

## 8. Submission deliverables

| Deliverable | Status |
| --- | --- |
| Public repo, commented code | BUILT |
| README: overview, setup, reproduce | BUILT |
| README: limitations reflection | BUILT |
| README: team member contributions | **MISSING** |
| Section 13 pillar traceability table in README | **MISSING** |
| Devpost written description | **MISSING** |
| Demo video, YouTube, public | **MISSING** |
| Disclosure of model, cost, tokens, latency | BUILT |

## 9. Dead config: documented intentions, not features

The following configuration fields have no current reader in `src/`, `evaluation/`, or
`tests`:

```
flat_belief_entropy   slot_decay   use_llm   llm_timeout_seconds
```

Each corresponds to a named brief requirement. They are the gap list in miniature.

## 10. What to do, ranked by value per hour

Jarell owns `src/retrieval/` and the `weight_*` fields. **12 of 15 gaps are outside his
territory.** Only the category route, vector route and field weighting collide.

| # | Work | Closes | Score | Effort | Collides |
| --- | --- | --- | --- | --- | --- |
| 1 | Name, measure and document the Pillar III behaviours that already run | Pillar III | 0 | 2h | no |
| 2 | Section 13 traceability table into the README | Judging | 0 | 1h | no |
| 3 | Slot decay and over-generality cutoff | Pillar II, 2 in-scope items | ~0 | 3h | no |
| 4 | Intent override detection | Pillar II | +0.012 max | 3h | no |
| 5 | Team contributions, Devpost, demo video | Submission | 0 | 1 day | no |
| 6 | Offline LLM artefact consumed deterministically at runtime | Pillar I LLM stage | unknown | 1 day | no |
| 7 | Belief object with entropy | Makes the spec's own thesis true | unknown | 1 day | no |
| 8 | Category and vector retrieval routes | Pillar I multi-route | unknown | 1 day | **yes, Jarell** |

Item 6 is the only honest route to the named LLM stage given that official scoring may run
with networking disabled: use an LLM offline to build an artefact, ship the artefact, consume
it deterministically at runtime. It satisfies the requirement, survives a network-disabled
run, and is disclosable without pretending we call a model live.

## 11. Deliberate deviations to disclose

Per spec section 13, these must be stated rather than hidden:

- The Buying track uses steep reweighting, not literal hard filtering (spec 5.4).
- Cross-session long-term profiling is not implemented: the rules define every session as
  isolated and `session_id` is a fresh UUID, so there is no eval surface for it.
- The system makes zero LLM calls. This is a deliberate response to the submission rules
  warning that official scoring may disable network access, not a missing feature.
- The ask policy models the evaluator's constraint classifier. Three of the ten legal
  attributes can never be answered, and we do not ask them. This is fitted to this
  simulator and a real deployment would need a general parser.
