# Consolidation: what was built, why, what is left

Last verified 31 August 2026 against `main` at `d476b36`. Section 4.11 (profile
preference-tags in the ranker: considered, measured, not merged) added
afterwards on this machine, which reproduces a train baseline of 0.8963 and an
all-200 official score of 0.892323 rather than the 0.893583 recorded below; see
the note at the end of section 1.

This is the single document to read if you are picking the project up, joining
it, or judging it. It supersedes nothing: `docs/handoff.md`,
`docs/build-plan.md`, `docs/status-evaluation.md`,
`docs/retrieval-merge-finding.md` and `docs/llm-semantic-stage.md` are still the
detailed record. This one puts them in order, in plain English, and says what is
unfinished.

---

## 1. Verification performed for this document

Everything below was re-run, not recalled from memory.

| Check | Result |
| --- | --- |
| All remote branches enumerated, unmerged commits found | 3 branches ahead of `main`, all accounted for in section 3 |
| Test suite on `main` | 89 passed (94 after the section 4.11 work) |
| Offline probe (`evaluation/verify_offline.py`) | PASS, 10 turns with every socket blocked |
| Official evaluator on `main` | TechnicalScore **0.893583**, 1 minute 40 seconds |
| Fresh clone, judge simulation | identical 0.893583, so nothing depends on this working copy |
| Working tree | clean, no uncommitted changes |

**Baseline reproduction, added later.** `requirements.txt` pins nothing
(`numpy>=1.26`, `scikit-learn>=1.4`). On a machine with numpy 2.5.2 / scikit-learn
1.9.0 the same commit scores **0.892323** on all 200 (MRR 0.7754 vs 0.7796,
browsing MRR 0.6706 vs 0.675) — TF-IDF tie-breaks shift with the sklearn version.
Sweep *deltas* on one machine are still valid; comparing an absolute score to the
0.893583 above is not, until the versions that produced it are pinned. This is a
project-wide housekeeping item, not a regression.

Headline numbers from the official run:

| Metric | Value |
| --- | --- |
| Hit Rate@10 | 0.975 |
| MRR | 0.779609 |
| MTTC | 2.39 turns |
| Efficiency | 0.861 |
| **TechnicalScore** | **0.893583** |
| Reported token usage | 0 (the shipped path makes no network calls) |

By scenario: buying 0.975 hit / 0.828 MRR, browsing 0.975 hit / 0.675 MRR,
intent override 0.967 hit / 0.917 MRR, boundary 1.000 hit / 0.817 MRR.

---

## 2. What the product is, in one paragraph

A judge clones the repository, installs the requirements, and runs
`python3 -m evaluator.local_evaluator` from inside
`techjam-conversational-search/`. That command drives our agent through 200
simulated shopping conversations. On each turn the agent reads what the customer
said, updates what it believes about the target product, decides whether it knows
enough to recommend or should ask one more question, and returns either a
ten-product shortlist or a clarifying question. It scores 0.893583 against the
organiser's baseline of 0.1067. It needs no API key, no network, and finishes in
under two minutes.

---

## 3. Every branch, and where its work ended up

Verified by enumerating all remote refs and diffing each against `main`.

| Branch | Author | State |
| --- | --- | --- |
| `main` | | The submission. `d476b36`. |
| `codex/field-aware-retrieval` | Jarell | 7 commits ahead of `main`, but **all of its code is already on `main`**. Verified file by file: `src/interfaces.py` and `evaluation/rank_diagnostics.py` are byte-identical; `src/retrieval/baseline.py` on `main` is his file plus the pooled route added on top. What remains unmerged is only his edits to four documentation files, which were later rewritten. Nothing of his is lost. |
| `jasper/day1-scaffold`, `jasper/phase0-diagnostics` | Jasper | 1 commit each ahead: the merge commits from the closed PRs #5 and #6. No content. Safe to delete. |
| `jasper/*` (9 others) | Jasper | Fully merged into `main`. Safe to delete. |
| `claude/*` (3 remotes) | | Dead, left over from the branch rename. Deletion returned 403. Cosmetic only. |

Contributors on `main`: 40 commits authored by the agent, 12 by Jasper (as
author and merger), 2 by Jarell.

Merged pull requests: #4 Phase 0 diagnostics, #7 agent and platform and tests,
#8 ask policy, #10 retrieval merge, #11 README and handoff correction, #12
offline LLM prior, #13 artefact shipping fix, #14 live reranker. Closed without
merging: #1, #2, #3 superseded; #5 and #6 were a stacked-PR mistake that
targeted feature branches instead of `main`.

---

## 4. The design decisions, and the evidence for each

Every one of these was measured on the 160 training sessions, never on the 40
held-out ones. Where a number appears, it is a TechnicalScore unless stated.

### 4.1 The entry point had to be bridged

`evaluator/local_evaluator.py` hardcodes `from starter.agent import Agent`, and
that file held the organiser's BM25 baseline. Until it was bridged to our code,
every local number any of us produced measured a code path the graded command
never executed. Fixing this alone took the score from 0.107 to 0.542.

`tests/test_entry_point.py` now pins SHA256 hashes of the evaluator files, so
nobody can quietly edit the grader and report the result as ours.

### 4.2 Popularity gets real weight (`weight_popularity = 2.0`)

The single largest tuning win, worth about +0.21. Swept 0.0 to 5.0: the curve
rises from 0.4608 at zero to a plateau near 0.743 between 1.5 and 3.0, then
falls to 0.7176 at 5.0. We shipped 2.0, the middle of the plateau, rather than
the exact argmax, because 1.5, 2.0 and 3.0 differ by less than 0.005 and picking
the peak would be fitting noise on 160 sessions.

It is not a degenerate "always show bestsellers" prior.
`evaluation/check_degeneracy.py` shows unrelated queries still return completely
disjoint top tens even at weight 20, because the prior only reorders a shortlist
retrieval has already filtered.

### 4.3 Exact phrase overlap is the strongest single signal (`exact_phrase_boost = 4.0`)

The organiser's simulated customer does not speak naturally. Its utterances are
verbatim substrings of the target product's own catalog record. So when a whole
disclosed phrase appears word for word in a product's text, that product is very
likely the target. Worth about +0.038.

This fact drives several other decisions below, and it is the main reason to be
sceptical of semantic reasoning here.

### 4.4 Rerank depth is 200, and the answer reversed once phrase overlap existed

| Depth | Score | Hit | MRR | MTTC |
| --- | --- | --- | --- | --- |
| 100 | 0.8540 | 0.919 | 0.804 | 3.33 |
| **200** | **0.8951** | 0.975 | 0.790 | 2.48 |
| 400 | 0.8737 | 0.975 | 0.704 | 2.24 |
| 800 | 0.8670 | 0.981 | 0.664 | 2.14 |

Every remaining miss at depth 100 was a target sitting in the candidate pool
beyond rank 100, so it was never reranked and never got its phrase evidence.
Widening to 200 rescues them. Past 200 the extra candidates dilute rank one
faster than they add hits. Worth about +0.041.

This is a case where the earlier sweep gave the opposite answer. Re-sweeping
after a change to the ranker was not optional.

### 4.5 The ask policy is ordered by measured yield, not by intuition

`ATTRIBUTES_BY_YIELD = (feature, material, color, style, size, use_case)`, and
`UNANSWERABLE = (category, brand, budget)`. We measured which attributes the
simulator actually answers with new information and asked those first, and
stopped asking the three it never usefully answers. Worth about +0.06.

This resolved the critical open question from the spec (section 10.1): the
simulator does reveal attribute values, so question selection is real work and
not pure cost.

### 4.6 Intent Override demotes rather than erases (`override_demote = 0.5`)

The brief describes "slot erasure". We do not erase, and that is measured rather
than a soft reading. In 28 of the 30 public `intent_override` sessions, the
preference the customer says to ignore is itself lifted from the target
product's own record and appears verbatim in that product's text. Erasing it
deletes true evidence.

| Setting | Behaviour | Score | MRR |
| --- | --- | --- | --- |
| 1.0 | ignore the pivot entirely | 0.8951 | 0.790 |
| **0.5** | demote, shipped | **0.8951** | 0.790 |
| 0.0 | literal erasure | 0.8925 | 0.782 |

Reproduce with `evaluation/override_audit.py`. We ship the middle value because
it implements the brief's intent without paying for the literal reading.

### 4.7 Field-aware retrieval is merged, and switched off

Jarell's per-field TF-IDF reached a 0.988 hit rate in isolation, better than our
0.975. We merged it and swept the full mixing range. Field-aware scoring costs
MRR monotonically: there is no interior optimum, every step away from pooled
indexing moves the target further down the list. Route fusion added no recall
the pooled route had not already found.

The cause is mechanical. Per-field cosine normalises by that field's own length,
so a product whose short `store` field contains a query term scores near 1.0 on
it, while the same term is diluted across a long description. That
systematically over-rewards short-field matches and crowds the top of the list
with plausible generics.

So the code is merged, tested, and defaulted to zero weight. A field with weight
zero is not indexed at all, so it costs nothing at runtime. If someone improves
the field weighting, turning it on is a config change, not a rewrite. Full sweep
in `docs/retrieval-merge-finding.md`.

### 4.8 Seven ideas were measured and rejected

Recorded so nobody re-tries them without new information:

1. **Dense embeddings.** No gain over the lexical route on a customer who quotes
   catalog text verbatim.
2. **LLM catalog attribute extraction.** Did not beat the parser.
3. **Field-aware retrieval weighting.** Section 4.7.
4. **Route fusion.** Section 4.7.
5. **Rarity-weighted phrases.** +0.0003, inside noise.
6. **Three parser "improvements"** that each read as obvious and each lost
   score: dropping "no additional preference" replies (0.7369 against 0.7422 for
   keeping them), deduplicating repeated phrases, and a narrower opening regex.
7. **Profile preference-tags in the ranker.** Section 4.11. Monotonically
   negative with the full tag set; a noise-band spike with the discriminating
   subset. Prototype and tests removed; nothing shipped.

The "no preference" result is worth remembering. Those replies look like pure
noise, but their tokens carry almost no inverse document frequency, so removing
them mostly just shortens the query.

### 4.9 The LLM never sits on the critical path

Two LLM stages exist. Both are off by default and both degrade to the
deterministic ordering.

**Offline semantic prior** (`offline/build_semantic_prior.py`, artefact at
`artifacts/semantic_prior.json`). The model runs once, ahead of time, over
catalog products and writes a JSON file of appeal and use-case judgments. At
runtime the agent just reads the file. No call, no key, no latency.

**Live conversational reranker** (`src/language/rerank.py`). Reads the actual
conversation and reorders the top 20. Off by default (`use_llm = False`). Every
failure path returns the input order and reports `used_llm=False`.

The reason for this design: the organiser may score us with networking disabled,
and the hard limits mean a model on the critical path is a way to score zero
rather than a way to score higher. The zero-LLM fallback is also a real
Feasibility argument, which is 15% of the grade.

### 4.10 Forty sessions were reserved and have never been touched

`holdout_size = 40`, stratified, seeded at 20260101. Every tuning decision above
was made on the other 160. The held-out curve is worth more to a judge than the
score itself, and it is only worth something if it is spent once.

### 4.11 Profile preference-tags feed retrieval but not ranking

Pillar III (spec 7.1) asks for the profile to be distilled into "retrieval **and
ranking**". The retrieval half is already there: `state/slots.py` appends the
buyer's `preference_tags` to the query, and removing them costs 0.037. The
ranking half is not, because `rank/baseline.py::believe()` receives `profile` and
never reads it.

A prototype ranking term was built to close that: it scored a shortlist candidate
by how many of the buyer's tags (word-bounded, expanded to the words that occur
in the catalog) appear in its text. Swept on the 160 train sessions. The full
tag set is **monotonically negative** (0.8963 at 0.0, 0.8931 at 0.1, 0.827 at
2.0): the generic tags `material`, `comfort`, `style` match nearly everything and
dilute the verbatim phrase evidence. Restricting to the four discriminating tags
(`durability`, `performance`, `warmth`, `weather`) turns it marginally positive
at low weight, peaking at 0.8982 at weight 0.25, but that is a single-cell spike
with no plateau, so shipping it would be fitting the argmax on 160 sessions.

**The prototype, its config field and its tests were removed.** Nothing shipped;
the score is unchanged. Full tables and the mechanism in
`docs/retrieval-merge-finding.md`. The retrieval half stays; the honest position
for the writeup is that the profile informs retrieval and there was no ranking
gain to be had on top of it.

---

## 5. Honest weaknesses

Stated plainly, because a judge will find them anyway.

- **The semantic prior covers 60 products out of 50,000.** It was built as a
  cost-controlled sample run, not a full pass. Its contribution to the score is
  0.0005 (0.893583 with it, 0.893083 without). The code path is real and the
  full run is costed at about $13.44 on Haiku 4.5, but as shipped this stage is
  a demonstration, not a contributor. `src/semantic.py` defaults an unknown
  product to the mean of the enriched ones, so partial coverage is at least not
  itself a ranking signal.
- **The live reranker has never been executed against a real key.** It is
  tested, it fails safe, and it has never once produced a number. We do not know
  whether it helps.
- **All 13 remaining misses are ranking failures, not retrieval failures.** The
  target was in the candidate pool every single time. The ceiling here is
  ordering, not recall.
- **Browsing MRR (0.675) trails buying MRR (0.828) by a wide margin.** Open-ended
  sessions are where the remaining points are. One attempt at it, feeding the
  buyer's profile tags into the ranker, was measured and rejected (section 4.11).
- **We tuned against a deterministic simulator.** The 800 private sessions use
  different users and different target products. Decisions that exploit the
  simulator's verbatim-quoting habit may transfer less well than the training
  numbers suggest. Section 4.3 is the biggest exposure.

---

## 6. What is left to do

### 6.1 Required deliverables, not started

These carry more of the grade than the score does. Judging is Technical
Execution 35%, Innovation 20%, Impact 20%, Feasibility 15%, Presentation 10%,
and TechnicalScore is one input to the first of those, not the whole of it.

1. **Devpost written description.** Must cover: how the solution addresses the
   problem, development tools, APIs used, libraries and frameworks, datasets and
   assets. The rules additionally require disclosing model choice, estimated
   cost, token usage, and latency. We have all four numbers: model
   `claude-opus-5` configured with Haiku 4.5 used for the offline artefact, zero
   tokens on the shipped path, and 1 minute 40 seconds for all 200 sessions.
2. **Demo video.** End to end, on YouTube, public, linked from Devpost, no
   third-party trademarks or copyrighted content. For a backend solution a
   walkthrough of API usage, inference examples and result analysis is accepted.
3. **README completeness pass.** The README needs project overview, setup and
   installation, steps to reproduce, a reflection on limitations and what we
   would improve with more time, and team member contributions. The first three
   exist. The reflection and the contributions table need writing, and the
   README still reports 0.8931 where the current number is 0.893583.

### 6.2 Technical work worth doing, in priority order

1. **Spend the holdout.** `python evaluation/run_eval.py --split holdout`. One
   command, exactly once, at the very end. Running it early destroys its only
   value. This is the number that tells us whether we overfit.
2. **Run the live reranker once with a real key.** 478 calls at about 1,045
   input tokens each: **$0.64 on Haiku 4.5**, $1.29 on Sonnet 5, $3.22 on Opus
   5. Sixty-four cents to replace a guess with a fact.
3. **Attack browsing MRR.** It is 0.675 against buying's 0.828 and it is half
   the sessions. This is where the remaining score is.
4. **Optionally complete the semantic prior** over the full catalog, about
   $13.44. Only worth it if the 60-product sample shows a measurable direction
   first.

### 6.3 Housekeeping

- Delete the three dead `claude/*` remote branches (previously returned 403,
  needs someone with the right permission).
- Delete the eleven merged `jasper/*` branches.
- Decide whether to keep or delete `codex/field-aware-retrieval`. Its code is on
  `main`; only superseded documentation would be lost.

---

## 7. What needs to be tested before submission

Written as a checklist someone can actually run.

### 7.1 The four things that must never break

Run these before any push. They are the whole submission.

```
cd techjam-conversational-search && python3 -m evaluator.local_evaluator
```
Must print `recommended_technical_score` at or above 0.8931. If it drops,
something regressed.

```
python -m pytest tests/ -q
```
Must be 89 passed. `tests/test_entry_point.py` in particular proves the graded
command still reaches our agent, and pins the evaluator's own hashes.

```
python evaluation/verify_offline.py
```
Must print PASS. Proves the agent completes 10 turns with every socket blocked,
which is the scenario where the organiser scores us with no network.

```
python evaluation/check_degeneracy.py
```
Proves the popularity prior has not collapsed into "always return bestsellers".
It is negative-controlled, so it can actually fail.

### 7.2 The judge simulation, which catches what the above cannot

The single most valuable test we have, and the one that caught the worst bug.
Clone the repository into a clean directory, follow the README exactly as
written, and run the official command. Do not use the working copy.

```
git clone <repo> /tmp/judgecheck && cd /tmp/judgecheck/techjam-conversational-search
python3 -m evaluator.local_evaluator
```

This is how we found that `artifacts/` had been gitignored since day one,
silently excluding `artifacts/semantic_prior.json`. A judge's fresh clone had no
LLM stage at all and scored 0.893083 instead of 0.893583, with nothing failing
loudly to explain why. Verified passing for this document: the fresh clone
reproduces 0.893583 exactly.

Re-run this after **any** change to `.gitignore`, to file paths, or to anything
loaded from disk.

### 7.3 Tested before shipping, so do not remove these

- **Slot override and decay.** A silent bug here corrupts 30 sessions.
- **The turn cap.** Exceeding 10 turns is forced termination and a zero for that
  session. It is enforced inside the agent, not trusted to the evaluator.
- **`reset()` is total and cannot raise.** The evaluator calls it bare, so one
  bad session would otherwise cascade into all 200.
- **Response schema conformance.** A malformed payload scores zero regardless of
  how good the ranking was.
- **The reranker reports `used_llm` honestly.** A non-list `order` used to
  silently rebuild the input order while still reporting `used_llm=True`, which
  would have overstated the LLM stage in the required usage disclosure. Caught
  by a test before merge.

### 7.4 Not yet tested, and known to be untested

- The live reranker against a real API key. Zero real executions.
- The full-catalog semantic prior. Only 60 products have ever been enriched.
- Behaviour on the 800 private sessions. Unknowable until submission; the
  holdout run in 6.2 is the closest proxy we have.
