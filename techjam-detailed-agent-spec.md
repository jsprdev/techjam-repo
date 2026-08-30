# Conversational Shopping Agent: Build Specification

**Competition:** TikTok TechJam 2026, Problem Statement 4, Shopping Copilot
**Repo:** `jsprdev/techjam-conversational-search` (fork of `TechJam2026/techjam-conversational-search`)

---

## 0. How to use this document

This is a design spec, not an implementation. It states what to build, why each piece exists, and what order to build it in. Implementation choices (libraries, data structures, file layout) are left to the implementer.

Sections marked **UNVERIFIED** contain assumptions that must be checked against the actual evaluator source before building on them. Do not skip these. One of them can invalidate a major component.

---

## 1. The thesis

> Probability is calibrated but blind. Language is perceptive but badly calibrated. The architecture is the boundary drawn between them.

Every design decision below follows from that sentence. A probabilistic layer maintains a belief over the catalog and makes all decisions about uncertainty. A language layer handles everything requiring world knowledge or linguistic judgment. Neither is allowed to do the other's job.

**Product framing:** a shopping agent that knows how sure it is. Search engines return ten results whether they are confident or guessing. This one measures its own uncertainty and acts on it. Confident, it recommends. Unsure, it asks the single question that resolves the most doubt.

---

## 2. The task contract

### 2.1 Mechanics

Each session is derived from a real Amazon purchase history using a leave-last-out split. The last item the user actually bought is the hidden target. Everything before it is compressed into an anonymised preference profile handed to the agent at session start.

No human is involved. A deterministic evaluator plays the customer. The agent is a Python class.

Per turn, the agent may:
- ask a clarification question in `message`, and tag the field it wants via `ask_attribute`
- return a ranked list of up to 10 `parent_asin` values
- do both

The session ends when the target appears in the scored top 10, or after turn 10. Exceeding 10 turns is a hard zero for that session.

### 2.2 Agent interface

```
reset(session_id, user_profile) -> None
respond(session_id, user_message, turn, top_k) -> dict
```

Response fields: `message`, `ask_attribute`, `recommendations`, `usage`.

`ask_attribute` is one of: `category`, `material`, `color`, `size`, `style`, `brand`, `budget`, `feature`, `use_case`, `other`, or `null`.

The authoritative contract is `docs/agent_api_contract.json`. Read it before implementing.

### 2.3 Scoring

```
TechnicalScore = 0.50 x HitRate@10 + 0.30 x MRR + 0.20 x Efficiency
Efficiency = clip((11 - MTTC) / 10, 0, 1)
```

- **HitRate@10:** fraction of sessions where the target is found within 10 turns
- **MRR:** mean reciprocal rank of the target, a miss contributes zero
- **MTTC:** mean first-hit turn, a miss is assigned turn 11
- Only exact `parent_asin` equality counts as a hit
- Metrics are also reported per scenario

**Baseline to beat:** BM25 starter scores HitRate 0.125, MRR 0.068, MTTC 9.81, for a TechnicalScore of roughly 0.107.

### 2.4 How this score is actually used

The README states explicitly that TechnicalScore is an objective input to the Technical Execution assessment, that it is not a separate judging criterion, and that it does not represent the entire Technical Execution score.

Judging is five human criteria: Technical Execution 35%, Innovation and Problem Insight 20%, Impact and Relevance 20%, Feasibility and Practicality 15%, Presentation and Communication 10% (final event only).

**Design consequence:** do not optimise the number at the cost of looking like a metric gamer. A team scoring 0.42 who can explain their retrieval ceiling, their weakest scenario, and what they fixed beats a team at 0.45 reporting a bare number.

### 2.5 Scenario types

Sessions cover four labelled behaviours, and metrics break down by each:
- **Buying:** high intent, hard constraints
- **Browsing:** open ended, exploratory
- **Intent Override:** the user abandons prior constraints mid-session
- **Boundary:** edge cases, likely over-general or contradictory input

Diagnose per scenario from the start. The weakest bucket is the cheapest source of points.

---

## 3. Data reality

This section is derived from the official Amazon Reviews 2023 schema. The competition catalog is a frozen 50,000-product subset of `Clothing_Shoes_and_Jewelry`. The organiser may have trimmed or renamed fields, so **inspect the actual catalog before assuming any of this holds**.

### 3.1 Item metadata fields

| Field | Type | Use |
| --- | --- | --- |
| `parent_asin` | str | Primary key, the thing you must match exactly |
| `title` | str | Main retrieval text |
| `main_category` | str | Coarse category |
| `categories` | list | Hierarchical category path |
| `features` | list | Bullet points, rich retrieval text |
| `description` | list | Long-form text |
| `price` | float | Budget attribute |
| `store` | str | Brand proxy |
| `details` | dict | Free-form key-value, includes Color, Size, Material, Brand, Style |
| `average_rating` | float | Quality prior |
| `rating_number` | int | Popularity prior |
| `bought_together` | list | Co-purchase signal, often null |
| `images`, `videos` | list | Out of scope, multimodal is explicitly excluded |

### 3.2 The critical finding: the enum maps onto the schema

Seven of the ten `ask_attribute` values map directly onto existing metadata fields:

| ask_attribute | Source |
| --- | --- |
| `category` | `categories`, `main_category` |
| `material` | `details["Material"]` |
| `color` | `details["Color"]` |
| `size` | `details["Size"]` |
| `style` | `details["Style"]` |
| `brand` | `store`, `details["Brand"]` |
| `budget` | `price` |
| `feature` | `features` |
| `use_case` | **Not present. Must be derived.** |
| `other` | Catch-all |

This narrows the LLM extraction job considerably. It is mostly **normalisation and gap-filling** of existing fields, plus **derivation** of `use_case` and any soft attributes like formality or season. It is not extraction from scratch.

`details` is a free-form dict whose keys vary wildly across products. Normalising it into a consistent schema is real work and is the foundation the entire belief layer stands on.

### 3.3 The parent_asin variant problem

The Amazon docs state that products differing in colour, style, or size usually share the same `parent_asin`.

Scoring is exact `parent_asin` equality. Therefore **colour and size are structurally weak discriminators** because variants collapse into a single scoring unit. The `details` dict holds one representative variant's values, not the full range.

**Design consequence:** do not assume the ten attributes are equally useful. Colour and size are likely to be low-information despite being well populated. This must be measured, not assumed. See the attribute reliability model in section 6.2.

### 3.4 Stated assumptions, do not build around these

The brief grants these explicitly. Building for them wastes time.

- Inputs are pre-cleaned text. **No spelling correction, typo handling, or ASR noise handling is needed.**
- Catalog, pricing, and category trees are static for the duration.
- Each session is an isolated single-user interaction. No concurrency handling needed.

### 3.5 Hard limits

- **10 turns maximum.** Exceeding it is forced termination and zero for that session. Enforce a hard stop in the agent itself, do not rely on the evaluator.
- **The catalog is read-only.** No structural mutations, no mock ASIN injection. Derived artefacts such as the extracted attribute table must live in separate files alongside it, never written back into the catalog.

### 3.6 Scale

18.3 MB gzipped, roughly 1.5 KB of raw text per product decompressed. Naive LLM extraction over the full catalog is on the order of 20 million input tokens. Trim to needed fields first. Prototype on 500 products before committing to a full run.

---

## 4. Architecture

Three layers with a strictly enforced boundary.

```
Layer 3  Self-evolution        offline, between runs
Layer 2  Language              semantics, world knowledge
Layer 1  Probability           belief, uncertainty, decisions
```

**The boundary rule:** the language layer never holds the belief state and never decides when to commit. Language models are badly calibrated about their own confidence, and calibration is the entire point of Layer 1.

---

## 5. Layer 1: the deterministic floor

Runs with zero LLM calls. If the API dies mid-demo, this still works. That fallback is a real Feasibility argument and should be stated explicitly in the submission.

This layer is textbook conversational recommender system design. Entropy-based attribute selection has been standard in the CRS literature for years. Treat it as a solid foundation, not as the innovation claim.

### 5.1 Intent routing (Pillar I requirement)

Before retrieval, classify the turn as **Buying** or **Browsing**. The brief calls this Dual-Track Routing and it is a named architectural requirement, not optional.

- **Buying track:** high intent, hard constraints present. Weight the lexical and category routes heavily. Sharpen the belief distribution aggressively around constraint matches. Narrow truncation.
- **Browsing track:** open ended, no firm constraints. Weight the dense route heavily to allow cross-category scenario matching. Keep the distribution flat and the candidate pool wide.

Routing is per turn, not per session. A user can open in Browsing mode and converge to Buying by turn three. The router reads the current message plus the belief state's current entropy.

This routing is what makes the Buying and Browsing scenario buckets score differently, and metrics are reported per scenario.

### 5.2 Retrieval: three routes

The brief specifies keyword, category, and vector similarity. Do not drop the category route.

- **Lexical:** BM25 over title, features, description
- **Category:** structured matching against `categories` and `main_category`
- **Dense:** embeddings over concatenated product text

Merge into a single candidate pool with track-dependent weights from 5.1.

**Dynamic truncation:** the pool size is not fixed. Truncate narrowly on the Buying track and widely on Browsing. This is named in the in-scope list as custom dynamic truncation and should be a tunable parameter, not a constant.

Must run entirely in memory. External vector database clusters are out of scope per the rules.

### 5.3 Belief state

A score over catalog items representing "how likely is this the target." Initialised from retrieval plus priors, updated every turn.

**Priors at session start, before any conversation:**
- **Popularity.** `rating_number` is a strong prior. In leave-last-out Amazon benchmarks, popularity baselines are competitive with far more sophisticated models. Real purchases concentrate on popular items.
- **Profile.** The anonymised profile is prior purchase behaviour, not flavour text. Brand continuity, category continuity, price band continuity. This is collaborative filtering signal available before the user says anything.

### 5.4 Soft reweighting, never hard filtering

**This is the single most important rule in the spec.**

When a constraint arrives, demote non-matching items. Never remove them.

Rationale: Amazon metadata is patchy and inconsistent. A hard filter that drops the target is unrecoverable and costs the full 0.5 weight of HitRate for that session. A demotion is recoverable three turns later. Users also answer imperfectly, and the parent_asin variant problem means attribute values are noisy even when present.

If asked why not just filter: users answer imperfectly and metadata has holes.

**Deliberate deviation from the brief, flag this in the writeup.** Pillar I asks the Buying track to "lock hard constraints." This spec implements that as a very steep reweighting rather than literal deletion. The behaviour is nearly identical, a constraint violation drops an item far down the ranking, but it stays recoverable if the metadata was wrong or the user contradicts themselves later.

Make the sharpness a tunable parameter and measure both settings on the public set. If true hard filtering genuinely scores better, use it. If it does not, you have evidence for why you deviated, which is a stronger answer to a judge than either choice made blindly.

### 5.5 Slot state machine (Pillar II requirement)

The belief over items and the slot state are two different objects. Keep both.

Slots are the explicit accumulated constraints: category, material, colour, size, style, brand, budget, feature, use case. They are what gets shown, logged, and reasoned about. The belief is the item-level distribution the slots feed into.

The state machine must handle three operations:
- **Accumulation:** a new constraint arrives, add it
- **Override:** the user changes direction, erase and rewrite the affected slots (see 6.4)
- **Decay:** slot confidence weakens over turns unless reinforced

Decay is named in the in-scope list as slot decay over time. It matters because an inferred constraint from turn one should not dominate turn eight with the same weight as a constraint the user just stated. Make the decay rate tunable.

### 5.6 Proactive guidance on over-generality (Pillar II requirement)

When the candidate pool is overloaded and the belief is near-flat, the brief specifies an immediate retrieval cutoff followed by a structured clarification prompt. Do not return a wide low-confidence list in that situation, ask instead.

This is the same machinery as the commit policy in 5.8, but the over-generality case is the specific trigger the brief names. Implement it as an explicit condition so it is visible in the code and demonstrable in the demo.

### 5.7 Question selection

Rank the ten legal attributes by how evenly each splits current belief mass. Ask the one nearest a 50/50 split.

Weight each attribute's score by its **reliability** (section 6.2). Information gain computed over a field that is 60% missing, or over an attribute that does not discriminate between parent_asins, is fake information.

Never ask about something already known from the profile or already answered.

### 5.8 Commit policy

Each turn, decide between recommending and asking, based on the shape of the belief distribution.

- Belief peaked, low entropy: commit, recommend
- Belief flat, high entropy: ask one more question

This is grounded in published work on entropy-driven dialogue policies, where the entropy of retrieval score distributions routes between direct recommendation and exploratory questioning.

Frame this formally as a POMDP over the belief state, solved with myopic value of information. That framing is honest, citable, and subsumes the question selection rule.

**Always return recommendations.** There is no penalty for guessing alongside a question, and the session can only end favourably by surfacing the target. Never return an empty list.

### 5.9 What not to do

Do not vary the length of the recommendation list turn by turn to game the MRR-versus-MTTC tradeoff. It reads as metric gaming to a human judge and the README now makes clear the score is only partial evidence toward one third of one criterion.

---

## 6. Layer 2: language

Four jobs. Each is something probability genuinely cannot do.

### 6.1 Offline attribute normalisation and derivation

The highest leverage work in the entire build. It is invisible in the demo, which is why other teams will skip it.

Run once, offline, over the catalog. Produce a clean structured attribute table:
- Normalise the free-form `details` dict into consistent keys and values
- Fill gaps by reading `title`, `features`, and `description`
- Derive attributes that do not exist in the schema: `use_case`, formality, season, gift-suitability

The entropy layer is worthless over missing or inconsistent metadata. This step is what makes Layer 1 function at all.

Cost control: trim input fields, test the prompt on 500 products, verify the extracted fields actually improve the recall diagnostic before running the full catalog.

### 6.2 Attribute reliability model

For each of the ten attributes, estimate two things from data:
- **Coverage:** what fraction of the catalog has a usable value
- **Discriminative power:** how much the attribute actually separates parent_asins, given the variant collapsing problem

Feed these as weights into question selection. This is the empirical answer to section 3.3.

### 6.3 Query understanding

Map vague natural language into a distribution over attributes.

"Something for my sister's beach wedding" implies formal, warm weather, gift, likely not the buyer's own size. That is world knowledge. No retrieval or probability method recovers it, and it is precisely the gap the problem statement opens with.

Output feeds the belief update. The LLM proposes soft evidence, the probabilistic layer decides what to do with it.

### 6.4 Intent override detection

Distinguish "and also" from "actually, forget that."

Accumulation adds constraints. Override discards them. A system that only accumulates will silently fail the entire Intent Override scenario bucket, and metrics are reported per scenario.

The CRS literature notes that most systems assume users always know what they want, which is exactly the assumption this scenario is built to break. Handling it well is a genuine differentiator.

### 6.5 LLM semantic ranking (Pillar I requirement)

The brief specifies the pipeline base as Multi-Route Retrieval followed by LLM Semantic Ranking. This is a named requirement and must exist.

Rerank the top candidates against the current slot state and the user's phrasing. The reranker sees a shortlist, not the catalog. Keep the shortlist small enough that latency and cost stay proportionate, since Feasibility is 15% and token usage must be disclosed.

**The boundary still holds.** The reranker proposes an ordering, it does not set the belief and it does not decide whether to commit. Blend its output into the belief rather than replacing the belief with it. If the reranker is unavailable, the deterministic ordering from Layer 1 ships instead.

The in-scope list explicitly permits fine-tuning prompt strategies or local scoring logic for this stage to compress decision paths. That is a sanctioned place to spend effort.

### 6.6 Question phrasing

The maths picks the attribute. The language layer writes the sentence, grounded in the current candidate set. "Are you thinking more casual or dressy?" rather than "Please specify style."

Small, but it is the visible surface in the demo video.

---

## 7. Layer 3: self-evolution

Three tiers. Be explicit in the writeup about which were actually built.

**Read Pillar III carefully.** It asks for *runtime* adaptation and *runtime* workflow re-orchestration, not just offline improvement between runs. Tier 7.1 is the runtime answer and must exist. Tiers 7.2 and 7.3 are offline and are additive, not substitutes. A submission whose only adaptation happens between eval runs has not answered the pillar.

Note the tension in the brief: Pillar III asks for continuously updated long-term user profiles, but the constraints state each session is an isolated single-user interaction. Long-term profiling has no eval surface here. Handle it by treating the supplied profile as the long-term state and distilling it into the session, then say so plainly rather than faking cross-session memory.

### 7.1 Within session (the runtime answer)

Three things happen at runtime, within a single session:

- **Belief updating** as evidence arrives
- **Reliability reweighting:** attributes that fail to move the distribution when asked get downweighted for the rest of the session
- **Workflow re-orchestration:** the track chosen in 5.1, the truncation width in 5.2, and the ask-versus-commit decision in 5.8 are all re-selected every turn from the current belief state. The pipeline shape is not fixed at session start. This is the honest reading of Adaptive Orchestration.

**Personalised context distillation:** compress the supplied profile plus the accumulated dialog into a compact session state that feeds retrieval and ranking, rather than replaying raw history each turn. This also keeps token usage down, which is a disclosed Feasibility metric.

### 7.2 Across sessions, offline

Learn the attribute reliability model (6.2) from the 200 public sessions rather than hand-tuning it.

### 7.3 The ambitious tier: failure-analysis loop

After each evaluation run, feed the missed sessions to a model. Have it diagnose why the target was never surfaced, and propose revisions to the extraction schema or prompts. Apply, re-run, keep what improves.

This has published precedent in self-refining prompt systems and is a defensible reading of the brief's pillar III on dynamic context programming, as opposed to a buzzword.

### 7.4 Guardrail, non-negotiable

Hold out a slice of the 200 public sessions. Never let the self-evolution loop see it.

Without this you are optimising your own simulator's quirks and shipping that to 800 sessions with different users and different target products. The held-out improvement curve is worth more to judges than the improvement itself, and almost no hackathon team holds out data.

---

## 8. Evaluation harness

Build this first, alongside the diagnostics. You cannot tune a policy you cannot measure.

Requirements:
- Run the agent against all 200 public sessions and report the full metric set
- Break metrics down by scenario type
- Train/validation split, with the validation slice reserved for section 7.4
- Log per-session traces: what was asked, what was believed, where the target ranked at each turn

The per-session trace is what lets you answer a judge's question about why a specific session failed.

---

## 9. Build order

### Phase 0: diagnostics before any building

**Recall ceiling.** Across all 200 public sessions, what fraction have the target inside the top 1000 retrieved candidates? Top 100?

This is the hard cap on HitRate. The dialogue layer can only reorder what retrieval already found. If the ceiling is 45%, then no amount of clever questioning gets you past 45%, and the entire weekend should go into retrieval instead.

Almost no team will run this. It tells you where to spend everything.

**Attribute coverage audit.** For each of the ten attributes, what fraction of the catalog has a usable value, before any LLM extraction.

**Read the evaluator.** See section 10.

### Phase 1: floor
Hybrid retrieval, priors, belief state, soft reweighting. Measure against baseline.

### Phase 2: dialogue policy
Question selection, commit rule. Measure the MTTC and MRR change.

### Phase 3: language layer
Offline extraction, query understanding, override detection. Re-run diagnostics to confirm extraction improved the ceiling.

### Phase 4: self-evolution
Reliability learning, then the failure-analysis loop if time permits.

### Phase 5: submission
Demo video, README, cost and latency disclosure.

---

## 10. Open questions, resolve before building

These are **UNVERIFIED**. They were not resolvable from public sources because GitHub blocks automated access to file contents. Read the files directly.

### 10.1 How does the simulated user respond to `ask_attribute`? (CRITICAL)

Read `evaluator/local_evaluator.py`.

If the simulator reveals the target's value for the attribute you request, then question selection is the core of the system and everything in section 5.4 holds.

If it replays a fixed script regardless of what you ask, then asking is pure cost with zero information gain, and the optimal policy is to never ask and commit immediately every turn. **This would invalidate a major component of this spec.**

The `ask_attribute` field being a constrained enum rather than free text strongly suggests the simulator consumes it programmatically, which supports the first reading. CRS user simulators in the literature conventionally respond with the target's attribute values. But this must be confirmed, not assumed.

### 10.2 Is MRR computed on the terminating turn?

The session ends on first hit. Confirm whether the recorded rank is the rank at that moment. This determines how the MRR and MTTC tradeoff actually behaves.

### 10.3 What does the profile actually contain?

Inspect `data/public_set.jsonl`. The README says raw user IDs, review text, timestamps, and purchase history are never disclosed, so the profile is some derived summary. Its exact contents determine how strong the prior in section 5.2 can be.

### 10.4 Catalog field verification

Confirm which of the fields in section 3.1 survived into the frozen catalog and in what form.

### 10.5 Config and rules

Read `docs/evaluation_config.json` and `docs/submission_rules.md` for anything that overrides the above.

---

## 11. Explicitly rejected

**Reinforcement learning.** 200 sessions cannot support policy learning. Training would happen against a self-built user simulator derived from those same sessions, then deployed to 800 sessions with different users and different targets. That measures the simulator's quirks, not the task.

State this in the writeup. "We framed this as a POMDP and solved it with myopic value of information rather than a learned policy, because 200 episodes cannot support policy learning without overfitting our own simulator" is a stronger answer than naming an algorithm. Technical Execution explicitly rewards deliberate, capable decision making, and knowing when not to reach for a tool demonstrates exactly that.

**Hard constraint filtering.** See 5.3.

**Recommendation-list-length gaming.** See 5.6.

**Multimodal.** Explicitly out of scope per the rules. Ignore the image fields.

**Fine-tuning base models.** Explicitly out of scope.

---

## 12. Deliverables checklist

Per the problem statement, all three are required.

**Devpost written description:** how the solution addresses the problem, development tools, APIs used, libraries and frameworks, datasets and assets.

**Public GitHub repository:**
- Well-structured, commented code covering all components
- README with project overview, setup and installation, steps to reproduce results
- A reflection on limitations and what you would improve with more time. This is named explicitly in the brief. Write it properly, self-awareness reads as maturity to judges.
- Team member contributions
- No committed API keys

**Demo video:** end-to-end, uploaded to YouTube, public, linked from Devpost, no third-party trademarks or copyrighted content. A walkthrough of API usage and result analysis is acceptable for a backend solution.

**Also required by the rules:** disclose model choice, estimated cost, token usage, and latency.

---

## 13. Pillar traceability

Judges will map the submission against the four pillars in the brief. Every named requirement must be locatable. Use this table in the README.

| Brief requirement | Pillar | Spec section |
| --- | --- | --- |
| Dual-track routing, Buying vs Browsing | I | 5.1 |
| Multi-route retrieval: keyword, category, vector | I | 5.2 |
| In-memory pipeline | I | 5.2, constraints |
| LLM semantic ranking | I | 6.5 |
| Dynamic state machine, information accumulation | II | 5.5 |
| Intent override, slot erasure and rewriting | II | 5.5, 6.4 |
| Retrieval cutoff on over-generality | II | 5.6 |
| Proactive structured clarification prompts | II | 5.7, 6.6 |
| Slot decay over time | II, in-scope | 5.5 |
| Custom dynamic truncation | I, in-scope | 5.2 |
| Runtime adaptation | III | 7.1 |
| Personalised context distillation | III | 7.1 |
| Adaptive orchestration, runtime re-orchestration | III | 7.1 |
| Prompt strategy tuning for the ranking stage | in-scope | 6.5 |
| Coverage, Precision, Efficiency metrics | IV | 2.3, 8 |
| Per-scenario reporting | IV | 2.5, 8 |

Deliberate deviations to disclose: the Buying track uses steep reweighting rather than literal hard filtering (5.4), and cross-session long-term profiling is not implemented because the constraints define sessions as isolated (7).

---

## 14. Rubric mapping

| Criterion | Weight | What earns it |
| --- | --- | --- |
| Technical Execution | 35% | Layered architecture, clean code, reliable demo, TechnicalScore as supporting evidence |
| Innovation and Problem Insight | 20% | The calibration boundary, the commit rule, the recall ceiling diagnostic |
| Impact and Relevance | 20% | Writing only. Argue why conversational commerce matters beyond this prompt |
| Feasibility and Practicality | 15% | Zero-LLM fallback, in-memory execution, honest cost and latency disclosure, restrained LLM usage |
| Presentation and Communication | 10% | The one-sentence thesis, and being able to answer why each decision was made |

The pitch narrative: state the thesis, show the demo handling a vague query and then an intent override, then explain what the recall ceiling was and what you did about it.
