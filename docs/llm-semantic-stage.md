# The LLM semantic ranking stage

How the submission answers Pillar I's named pipeline component without putting the
graded run at risk.

## The conflict

Pillar I names the pipeline base literally: **"Multi-Route Retrieval then LLM Semantic
Ranking"**. Pillar IV describes MRR as evaluating "the LLM's accuracy in pushing the exact
purchased item to the absolute top". The brief assumes a model is in the loop.

`docs/submission_rules.md` says the opposite: *"For official final scoring, organizer policy
may disable network access"*, under CPU, memory and timeout limits. A model call on the turn
path is then a way to score zero, not a way to score higher.

Both are real requirements and they conflict only if the call happens per turn.

## Two stages, not one

The brief's phrase "LLM Semantic Ranking" describes a model that READS THE
CONVERSATION and reorders the shortlist for this customer. There are now two
stages, and they answer different halves of the problem.

| | Offline prior | Live rerank |
| --- | --- | --- |
| File | `offline/build_semantic_prior.py` | `src/language/rerank.py` |
| Runs | Once, before submission | Every turn |
| Sees | One product, alone | The conversation plus the shortlist |
| Reacts to what the customer said | No | **Yes** |
| Needs network at run time | No | Yes |
| Default | **On** | **Off** |

The live rerank is what spec 6.5 actually specified. It is off by default, and
that is not a hedge: `docs/submission_rules.md` warns official scoring may run
with network access disabled, so a model on the turn path is a way to score
zero. The spec anticipated exactly this and said the deterministic ordering
ships instead when the reranker is unavailable.

Enable it with `Config.use_llm = True` and an `ANTHROPIC_API_KEY`. With it off,
the reported token usage is 0 and the score is unchanged.

## The resolution for the graded run

Move the model off the turn path. It runs once, offline, before submission.

```
offline/build_semantic_prior.py     the LLM pass. Batches API, runs once
        writes
artifacts/semantic_prior.json       the artefact. Committed, ~4 KB
        read once at construction
src/semantic.py                     deterministic lookup, no network
        feeds
src/rank/baseline.py                blended into the ranking score
```

At run time the agent performs a dictionary lookup. `evaluation/verify_offline.py` passes
with every socket poisoned, and that check is negative-controlled: injecting a real socket
call makes it fail.

## What the model is asked

Deliberately not things the catalog already carries, and not anything a regex could recover.
Three judgments that need world knowledge:

| Field | Why it needs a model |
| --- | --- |
| `appeal` | Would a typical shopper buy this or scroll past it? The evaluation target is a real purchase, so this is the closest query-independent proxy for what is scored. `rating_number` captures popularity; this is meant to catch desirability the count misses, such as a well-reviewed but niche novelty item |
| `use_case` | Measured at **0% catalog coverage** in `phase0-findings.md`. There is nothing to normalise, it has to be derived |
| `formality` | Same. It is what separates two otherwise identical dresses |

## Cost

The Batches API runs asynchronously at 50% of standard rates, which suits a one-off offline
pass with no latency requirement. Products are trimmed to title, category path and the first
two features, keeping input near 120 tokens against roughly 60 output tokens.

| Scope | Tokens | Batched cost, Claude Haiku 4.5 |
| --- | --- | --- |
| 500 product prototype | 0.1M in, 0.03M out | **$0.14** |
| Full 50,000 catalog | 11.9M in, 3.0M out | **$13.44** |

Reproduce the estimate without spending anything:

```bash
python offline/build_semantic_prior.py --estimate-only
```

## Measured effect

Swept on the 160 session train split:

| weight_appeal | TechnicalScore | MRR |
| --- | --- | --- |
| 0.0, stage disabled | 0.8951 | 0.790 |
| 0.3 | 0.8960 | 0.793 |
| 1.0, shipped | **0.8969** | **0.796** |

Monotonic, and all of the movement is in MRR, which is where a semantic judgment should show
up. Through the official harness on all 200 sessions: 0.893083 to **0.893583**.

**That is a small gain and it should be read as such.** It is however produced by an artefact
covering **60 of 50,000 products**, 0.12% of the catalog, reaching 37 of the 200 targets. The
signal is directionally positive at that coverage. Whether it scales is untested, and the
enriched products are the most reviewed ones, which the popularity prior already ranks well,
so the marginal value of enriching the tail may differ in either direction.

## The live rerank, and what is tested

`src/language/rerank.py` sends the customer's disclosed constraints and up to 20
candidates, and asks for a reordering. The layer boundary from spec section 4
holds: it proposes an ordering, it does not set the belief and it does not
decide whether to commit, so a confident model cannot override a confident
belief, only reorder what the belief surfaced.

Nothing about it is load-bearing. Every failure path returns the input order
unchanged: no key, no network, timeout, malformed JSON, a reply that is not a
list, a safety refusal, or a model that drops, duplicates or invents indices.
`tests/test_rerank.py` covers each of those by injecting a fake client, so the
tests need neither a key nor a network.

**What is NOT tested: a single real API call.** No key was available here. The
request shape follows the current SDK documentation and the failure handling is
covered, but nobody has yet watched it succeed against the live API. Anyone
enabling it should run one session with a key before trusting it, and one of the
failure tests already caught a real bug: a non-list `order` silently rebuilt the
input order while still reporting that the model had run, which would have
overstated the stage in the usage disclosure.

## Honest limitations

- **The committed artefact was not produced by running the pipeline.** No API key was
  available in the development environment, and the organiser provides none. The 60 entries
  were authored by a language model working from the same trimmed product text the pipeline
  sends, to the same schema. The pipeline is real, runnable and costed, but a judge
  reproducing it needs a key and $13.44.
- **60 products is underpowered.** A gain of 0.0005 on the official harness is within the
  range where noise cannot be excluded.
- **The stage is optional by construction.** `Config.weight_appeal = 0.0` disables it
  entirely, and a missing or malformed artefact degrades to no signal rather than an error.
  `tests/test_semantic.py` covers each degradation path, including the agent running normally
  with the artefact deleted.

## What this does and does not claim

It **does** mean the submission has a real LLM stage: a costed pipeline, a committed artefact,
a deterministic consumer, and a disclosed answer to the Devpost question about APIs used.

It does **not** mean the score depends on a model. The system scores 0.8931 with the stage
disabled. The honest framing for the writeup is that the deterministic engine is the product,
the LLM contributes a small offline prior, and the architecture is built so that losing the
model costs 0.0005 rather than the whole run.
