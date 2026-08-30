# Conversational Shopping Agent

TikTok TechJam 2026, Problem Statement 4: Shopping Copilot, AI Conversational Search and
Recommendations.

A multi-turn shopping agent that finds a hidden target product in a frozen 50,000 item
Amazon catalog by talking to a simulated customer, in at most ten turns.

**Current score: 0.7511 TechnicalScore** on all 200 public sessions, measured through the
official harness, against a 0.107 BM25 baseline. Runs fully offline with no LLM dependency.

| Metric | Value | Weight | Baseline |
| --- | --- | --- | --- |
| HitRate@10 | 0.855 | 0.50 | 0.125 |
| MRR | 0.630 | 0.30 | 0.068 |
| Efficiency | 0.674 | 0.20 | 0.119 |
| MTTC | 4.265 | | 9.81 |
| **TechnicalScore** | **0.7511** | | **0.1067** |

| scenario | n | hit@10 | mrr | mttc |
| --- | --- | --- | --- | --- |
| buying | 80 | 0.850 | 0.616 | 4.04 |
| browsing | 80 | 0.875 | 0.604 | 4.10 |
| intent_override | 30 | 0.900 | 0.768 | 4.87 |
| boundary | 10 | 0.600 | 0.533 | 5.60 |

Every tuning decision was made on the 160 session train split. The 40 session held-out
slice has not been looked at, and will not be until the final check.

## Setup

Python 3.11 or newer.

```bash
pip install -r requirements.txt
```

Then fetch the frozen catalog, which is not committed because it is 58 MB:

```bash
cd techjam-conversational-search/data
curl -L -o catalog.jsonl.gz \
  https://github.com/TechJam2026/techjam-conversational-search/releases/download/participant-kit/catalog.jsonl.gz
gunzip catalog.jsonl        # expect 50,000 rows
```

## Reproducing the score

The official command, exactly as the organiser documents it:

```bash
cd techjam-conversational-search
python3 -m evaluator.local_evaluator --catalog data/catalog.jsonl
```

`starter/agent.py` is a bridge that re-exports our implementation from `src/`, because the
evaluator hardcodes `from starter.agent import Agent`. The evaluator itself is unmodified,
and `tests/test_entry_point.py` asserts both facts on every run.

Our own runner adds the train and held-out split, per-scenario breakdown, traces and
instrumentation:

```bash
python evaluation/run_eval.py                    # 160 train sessions
python evaluation/run_eval.py --split holdout    # 40 held-out sessions
python evaluation/run_eval.py --split all        # all 200, comparable to the baseline
python evaluation/run_eval.py --traces artifacts/traces.json
```

Tuning and verification:

```bash
python evaluation/sweep.py --grid weight_popularity=0.0,0.2,0.4 --limit 60
python evaluation/verify_offline.py              # proves no network dependency
python evaluation/diagnostics.py                 # catalog field and coverage audit
python evaluation/recall_ceiling.py              # the retrieval ceiling
pytest                                           # 63 tests, about two seconds
```

## How it works

Three layers with one rule between them: **language proposes, probability decides.** The
language layer never holds belief state and never decides when to commit, because models are
badly calibrated about their own confidence and calibration is the entire point.

```
src/
  agent.py          Entry class. Thin orchestration, owns the turn counter.
  interfaces.py     The four frozen seams between modules.
  config.py         Every tunable, in one dataclass.
  catalog.py        Read-only loader for the frozen catalog.
  response.py       Contract-legal response construction and validation.
  trace.py          Per-turn traces.
  retrieval/        Role 1. Lexical index and query building.
  state/            Role 2. Slots, override, decay, question policy.
  rank/             Role 3. Reranking the shortlist.
evaluation/         Role 4. Eval wrapper, splits, sweep, offline rig, diagnostics.
```

The system is deterministic and makes zero LLM calls. That is a deliberate architectural
choice, not a missing feature: the submission rules warn that official scoring may run with
network access disabled, so any model dependency is a liability rather than an asset.

## What the diagnostics found

`phase0-findings.md` carries the full evidence. Three measurements shaped the design:

1. **Retrieval is not the bottleneck.** Plain TF-IDF reaches 100% recall at depth 1000 and
   67% at depth 10 once every constraint is disclosed. There is no case for dense embeddings
   or a vector index, and none is used.
2. **Asking is free.** The evaluator checks recommendations for a hit before it reads
   `ask_attribute`, so one turn carries both. Every turn asks a question and returns a full
   ranked list.
3. **The metadata the enum implies is mostly absent.** `details` populates material at 4.3%,
   colour 4.9%, style 3.5% and size 1.8% across the catalog, and `use_case` has no source
   field at all. Attribute selection is weighted by measured coverage rather than by the
   attribute list in the brief.

## Model choice, cost, tokens and latency

Required disclosures.

| | |
| --- | --- |
| Model | None. Fully deterministic, zero LLM calls. |
| Estimated cost | $0.00 |
| Token usage | 0 prompt, 0 completion, reported as such in `usage` |
| Latency | 265 ms p95 per turn, 22 s one-off index build |
| Memory | 876 MB peak across a 200 session run |
| Network | Not required. Verified by `evaluation/verify_offline.py` |

## Limitations

- The score is measured against the organiser's deterministic simulator, whose customer
  utterances are drawn verbatim from the target product's own catalog record. Real shoppers
  paraphrase, so the lexical approach that works here would need semantic retrieval in
  production.
- `intent_override` sessions cannot be won before turn 3 or 4 by construction, which puts a
  floor under MTTC that no policy can move.
- Long-term cross-session user profiling is not implemented. The rules define every session
  as isolated and `session_id` is a fresh UUID per sample, so there is no eval surface for it.
  We say so rather than faking it.
