# CLAUDE.md

Guidance for Claude Code when working in this repository.

---

## 1. What this repo is

A submission for **TikTok TechJam 2026, Problem Statement 4: Shopping Copilot, AI Conversational Search and Recommendations**.

Two design documents already live here and are the source of truth for architecture:

- `techjam-summary-spec.md`, the pitch-level version
- `techjam-detailed-agent-spec.md`, the full build spec, 14 sections

Read the detailed spec before implementing anything. This file covers the problem statement, the code structure, and the working conventions. It does not restate the architecture.

---

## 2. Problem statement (verbatim scope)

### 2.1 Background

Traditional e-commerce search engines heavily rely on static keyword matching, failing to capture the fluid shifts of genuine consumer psychology and the distinction between open-ended browsing and high-intent buying. In modern conversational commerce, constructing an intelligent agent that leverages dynamic context programming is critical to bridging the gap between ambiguous user queries and complex product catalogs.

### 2.2 The four pillars

**I. Core Architecture: Intent Routing and Hybrid Pipeline**
- *Dual-Track Routing:* detect the user's underlying intent, triggering a high-precision filter track for targeted "Buying" to lock hard constraints, and a diverse dense retrieval track for open-ended "Browsing" to unlock cross-category scenario matching.
- *Pipeline Base:* an in-memory data stream of Multi-Route Retrieval then LLM Semantic Ranking, combining keyword, category, and vector similarity.

**II. Dialog Strategy: Multi-Turn Scenario Evolution**
- *Dynamic State Machine:* a conversational state tracker handling Information Accumulation (incremental slots) and abrupt Intent Override (slot erasure and rewriting).
- *Proactive Guidance:* an immediate retrieval cutoff on Over-Generality (candidate pool overload), followed by structured, proactive clarification prompts that guide user convergence.

**III. Self-Evolution: Dynamic Context Programming**
- *Runtime Adaptation:* use accumulated dialog history for Personalized Context Distillation, continuously updating short-term session state and long-term user profiles.
- *Adaptive Orchestration:* dynamic Context Programming for runtime workflow re-orchestration and strategy alignment, so the agent iteratively refines its own guidance logic.

**IV. Evaluation Matrix: Product and Efficiency Metrics**
Anchored on the final purchased record in the Amazon dataset:
- *Coverage (Hit Rate@K):* catalog recall and boundary capability at the retrieval stage.
- *Precision (MRR / Top-K Hit Rate):* accuracy in pushing the exact purchased item to the top.
- *Efficiency (MTTC, Mean Turns to Conversion):* rewards guiding the user to the correct product in fewer rounds.

### 2.3 Constraints and scope

**In scope**
- Highly sensitive intent-detection modules splitting traffic into Buying and Browsing tracks.
- Heterogeneous retrieval routing: weights, custom dynamic truncation, slot decay over time.
- Runtime-adaptive memory layers for personalized context distillation.
- Prompt strategy or local scoring logic tuning for the LLM ranking stage.

**Out of scope**
- UI/UX development. Evaluation is purely backend APIs and headless pipelines.
- Training or full-parameter fine-tuning of base foundational LLMs.
- Heavy external industrial vector DB clusters. Everything must run in memory.
- Multi-modal processing. Text catalogs, structured metadata, and text dialogs only.

**Hard limits**
- **Max 10 turns per session.** Exceeding it is forced termination and a zero for that session. Enforce the cap inside the agent, do not rely on the evaluator.
- **Catalog is strictly read-only.** No structural mutations, no mock ASIN injection. Derived artefacts live in separate files.

**Allowed assumptions**
- Inputs are pre-cleaned text. No spelling correction, typo handling, or ASR noise handling needed.
- Catalog, pricing, and category trees are static.
- Each session is an isolated single-user interaction. No concurrency handling needed.

### 2.4 Data and resources

- Frozen catalog of 50,000 products from Amazon Reviews 2023 `Clothing_Shoes_and_Jewelry`.
- 200 labeled public development sessions for local iteration.
- 800 private sessions held by the organizer for final evaluation, with separate users and target products.
- Provided kit: weak BM25 starter agent, deterministic local evaluator (Hit Rate@10, MRR, MTTC, Efficiency, TechnicalScore), Python agent interface plus machine-readable API contract, evaluation config, baseline results, data docs, submission rules, SHA256 catalog checksum.
- Upstream repo: `https://github.com/TechJam2026/techjam-conversational-search`
- Participant kit release: same repo, tag `participant-kit`
- Data source docs: `https://amazon-reviews-2023.github.io/`

The starter agent may be modified or replaced, but the official local evaluator stays. Keyword, rule-based, dense, hybrid, reranking, local models, and external model APIs are all permitted. The organizer provides no hosted model access, API keys, or credits. A paid LLM is not required. Never commit secrets.

### 2.5 Deliverables

1. **Devpost written description:** how the solution addresses the problem, development tools, APIs used, libraries and frameworks, datasets and assets.
2. **Public GitHub repo:** well-structured commented code covering all components, plus a README with project overview, setup and installation, steps to reproduce results, a reflection on limitations and what we would improve with more time, and team member contributions.
3. **Demo video:** end-to-end, on YouTube, public, linked from Devpost, no third-party trademarks or copyrighted content. For a backend solution, a walkthrough of API usage, inference examples, and result analysis is accepted.

Also required by the rules: disclose model choice, estimated cost, token usage, and latency.

### 2.6 Judging criteria

| Criterion | Weight |
| --- | --- |
| Technical Execution | 35% |
| Innovation and Problem Insight | 20% |
| Impact and Relevance | 20% |
| Feasibility and Practicality | 15% |
| Presentation and Communication (final event only) | 10% |

TechnicalScore is an objective input to Technical Execution, not a separate criterion and not the whole of that criterion. Do not optimise the number at the cost of looking like a metric gamer.

---

## 3. Code structure

The layering in the spec (probability, language, self-evolution) maps onto directories. The rule that matters: **the language layer never holds belief state and never decides when to commit.** If a module under `language/` imports and mutates the belief, that is a bug, not a shortcut.

```
src/
  agent.py            Entry class the evaluator imports. Thin orchestration only.
  config.py           Every tunable in one dataclass. No magic numbers elsewhere.
  catalog.py          Loads the frozen catalog plus derived attributes. Read-only.

  retrieval/
    lexical.py        BM25 over title, features, description
    category.py       Structured matching on categories and main_category
    dense.py          In-memory embeddings
    fuse.py           Weighted merge and dynamic truncation

  state/
    belief.py         Score over catalog items, soft reweighting, entropy
    slots.py          Slot state machine: accumulate, override, decay
    session.py        Distilled session context, turn counter, hard turn cap

  policy/
    intent.py         Per-turn Buying vs Browsing routing
    question.py       Attribute selection by expected information gain
    commit.py         Ask vs recommend, over-generality cutoff

  language/
    client.py         The single LLM entry point. Timeouts, retries, fallbacks.
    understand.py     Utterance to soft attribute evidence
    override.py       Accumulation vs override detection
    rerank.py         Semantic rerank of the shortlist
    phrase.py         Question wording

offline/
  extract_attributes.py   Catalog normalisation and derivation
  reliability.py          Per-attribute coverage and discriminative power
  failure_loop.py         Optional: diagnose misses, propose schema revisions

evaluation/
  run_eval.py         All 200 public sessions, full metric set
  diagnostics.py      Recall ceiling, attribute coverage audit
  traces.py           Per-session trace logging

data/                 Frozen catalog and sessions. Never written to.
artifacts/            Derived tables and caches. Gitignored if large.
tests/
```

Notes on the layout:

- `agent.py` stays thin. It sequences calls and owns the turn counter. Logic lives in the modules.
- Every LLM call goes through `language/client.py`. One place to add timeouts, log tokens, and degrade gracefully. The agent must still produce a valid response when the API is down, since the zero-LLM fallback is a real Feasibility argument.
- `offline/` scripts write to `artifacts/`, never to `data/`.
- Build `evaluation/` first. A policy that cannot be measured cannot be tuned.
- Create directories when the code arrives, not upfront. Empty packages with placeholder files are clutter.

---

## 4. Code conventions

**Keep it modular and clean, and do not overengineer.** These pull against each other, so the tie-break is: prefer the simplest thing that keeps the layer boundary intact.

Do:
- Type hints on every public function and dataclass.
- Dataclasses for state (`SlotState`, `Belief`, `SessionContext`). Plain dicts for wire-format payloads.
- Pure functions where practical. State mutation belongs in `state/`.
- Every tunable in `config.py` with a comment on what moving it does.
- Docstrings that explain *why*, since the *what* should be readable from the code.
- Small, focused modules. If a file passes roughly 300 lines, it is probably two things.
- Deterministic seeds anywhere randomness appears.
- Tests for the parts where a silent bug is expensive: slot override and decay, belief updates, the turn cap, response schema conformance.
- Log per-session traces from day one. They answer a judge asking why a specific session failed.

Do not:
- Add abstract base classes, registries, plugin systems, or factories for a single implementation.
- Add a dependency injection layer, a config framework, or a custom ORM.
- Cache before profiling shows a need.
- Introduce a persistence layer. Sessions are isolated and in-memory.
- Leave commented-out code or dead branches. Delete them, git remembers.
- Build for hypothetical future requirements. The hackathon ends.
- Hard-filter the candidate pool. Demote instead, per spec section 5.4.
- Commit API keys, or anything from `data/` that the rules keep private.

**Writing style, everywhere:**
- **Never use em dashes.** Not in code comments, docstrings, commit messages, the README, the Devpost description, generated user-facing text, or chat replies. Use a comma, a colon, parentheses, or two sentences.
- Prose in docs and comments stays plain and direct. No filler adjectives.

**Commits:** imperative mood, one logical change each, and a subject line that says what changed and why it was needed.

---

## 5. Before building

Spec section 10 lists open questions marked UNVERIFIED. Resolve them against the actual kit before writing anything that depends on them. The critical one is 10.1: whether the simulated user actually reveals the target's attribute value in response to `ask_attribute`. If it replays a fixed script, question selection is pure cost and a major component of the spec is invalid.

Then run the Phase 0 diagnostics in spec section 9, the recall ceiling and the attribute coverage audit, before committing effort anywhere else. The dialogue layer can only reorder what retrieval already found.
