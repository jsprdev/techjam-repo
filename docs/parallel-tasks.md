# Four parallel tracks

Derived from `docs/consolidation.md`. Written so four people can work at the
same time without stepping on each other.

The rule that makes this work: **each track owns a disjoint set of files.** If
you need to change a file another track owns, say so in the group chat first
rather than merging and finding out.

Read `docs/consolidation.md` before picking a track. It has the numbers and the
reasons behind every decision you are about to build on.

---

## The gate: do not touch the holdout

Forty sessions were reserved and have never been looked at. They are worth
something only because of that.

```
python evaluation/run_eval.py --split holdout
```

**One person runs this, once, at the very end, after all four tracks have
merged.** Not to check your work in progress. Not to compare two options. If you
tune against it, it stops being a generalisation check and becomes another
training set, and we lose the one honest number we have to show a judge.

Owner: whoever owns the final merge (Jasper). Everyone else uses
`--split train`, which is the default.

---

## Track A: Devpost write-up and README

**This is the highest-value track.** Presentation and the written submission
carry more of the grade than the score does. Judging is Technical Execution 35%,
Innovation 20%, Impact 20%, Feasibility 15%, Presentation 10%, and
TechnicalScore is one input to the first of those, not the whole of it. Right
now this deliverable is at zero.

**Files you own:** `README.md`, `docs/devpost.md` (new). Nothing else.

**What to produce:**

1. `docs/devpost.md`, the Devpost written description. Required content: how the
   solution addresses the problem, development tools, APIs used, libraries and
   frameworks, datasets and assets. The rules separately require disclosing
   model choice, estimated cost, token usage, and latency. All four numbers
   exist: the shipped path uses **zero tokens** and **no API key**, a full
   200-session run takes **1 minute 40 seconds**, the offline artefact was built
   on Haiku 4.5, and `llm_model` is configured as `claude-opus-5` for the
   optional live stage.
2. Finish the README. It currently has overview, setup and reproduction steps.
   It is missing the **reflection on limitations and what we would improve with
   more time**, and the **team member contributions** table. Both are explicitly
   required by the rules.
3. Fix the stale number. The README says 0.8931 in five places. That is the
   score with the semantic prior switched off. The real number is **0.893583**.
   Browsing MRR also reads 0.671 where the current value is 0.675.

**Where to get the material:** section 4 of `docs/consolidation.md` is the
design decisions with evidence, section 5 is the honest weaknesses list, which
is most of your limitations reflection already written.

**What makes this good rather than adequate:** lead with the measurements, not
the architecture. "We swept popularity weight from 0 to 5 and found a plateau,
so we shipped the middle of it rather than the argmax, because picking the peak
on 160 sessions is fitting noise" is a stronger paragraph than any description
of module layout. Judges see a lot of architecture diagrams and very few
ablation tables. Section 4.8, the six ideas we measured and rejected, is
probably the single most persuasive thing we have.

**Done when:** `docs/devpost.md` is ready to paste into the submission form, the
README has all six required sections, and no document in the repo reports a
score that a fresh clone does not reproduce.

---

## Track B: Demo video

**Files you own:** `docs/demo-script.md` (new). No code at all.

**What to produce:** an end-to-end demo video, on YouTube, public, linked from
Devpost, with no third-party trademarks or copyrighted content. For a backend
submission the rules accept a walkthrough of API usage, inference examples, and
result analysis, so no interface work is needed.

**Suggested shape, about four minutes:**

1. The problem in one sentence, and the baseline number: BM25 scores 0.1067.
2. A live terminal run of the official command, unedited, ending on 0.893583.
   Show the clock. Under two minutes is itself a Feasibility argument.
3. One real session traced turn by turn. Show the customer's utterance, the
   agent's belief updating, the decision to ask rather than recommend, and the
   target arriving in the top ten. `--traces artifacts/traces.json` on
   `run_eval.py` gives you the raw material.
4. One or two ablations. The rerank-depth table in section 4.4 of the
   consolidation is a good one, because the answer reversed when a different
   part of the system changed, which is a genuinely interesting thing to say.
5. The offline claim, demonstrated rather than asserted: run
   `python evaluation/verify_offline.py` and let it print PASS with every socket
   blocked.

**Coordinate with Track A** on the numbers so the video and the write-up do not
disagree. Numbers only, not files: you own different files.

**Done when:** uploaded, public, and the link is in `docs/demo-script.md` so
Track A can reference it.

---

## Track C: Browsing MRR

**This is where the remaining score actually is.** Best suited to whoever is
strongest on retrieval and ranking, which is Jarell.

**Files you own:** `src/rank/`, `src/policy/`, `src/state/`,
`src/retrieval/`, `evaluation/rank_diagnostics.py`, and the
retrieval/dialogue/ranking sections of `src/config.py`.

**The problem, stated precisely:**

| Scenario | n | Hit@10 | MRR |
| --- | --- | --- | --- |
| buying | 80 | 0.975 | **0.828** |
| browsing | 80 | 0.975 | **0.675** |

Same recall, 0.153 lower MRR, across 40% of the sessions. Browsing finds the
target just as often and buries it further down the list. MRR carries 0.30 of
the score.

**What we already know, so you do not repeat it:**

- All 13 remaining misses across the whole set are **ranking failures, not
  retrieval failures**. The target was in the candidate pool every single time.
  Do not go looking for recall.
- Six things have been measured and rejected. Section 4.8 of the consolidation
  lists them: dense embeddings, LLM catalog extraction, field-aware weighting,
  route fusion, rarity-weighted phrases, and three parser changes that each read
  as obvious and each lost score. Do not re-try these without a new reason.
- The field-aware route is merged and defaulted to zero weight because it costs
  MRR monotonically across the whole mixing range. If you find a weighting that
  does not, turning it back on is a config change, not a rewrite. See
  `docs/retrieval-merge-finding.md`.
- Rerank depth is 200 and the sweep **reversed** when exact phrase overlap was
  added. Any change to the ranker means re-sweeping depth, not assuming it.

**Where to start:** `evaluation/rank_diagnostics.py` on the browsing subset
only. The specific question is where the target sits when it is not at rank one,
and what outranks it. Track the browsing MRR, and watch that buying MRR and
MTTC do not fall to pay for it: the composite is
`0.50 x HitRate + 0.30 x MRR + 0.20 x Efficiency`, so a change that lifts
browsing MRR by 0.05 while costing half a turn of MTTC is roughly break-even.

**Rules:** train split only. Every claim gets a sweep, not an argument. Anything
you reject goes in `docs/retrieval-merge-finding.md` with its number, so the
next person does not spend a day rediscovering it.

**Done when:** browsing MRR is up and the composite on the train split is above
0.8936, or you have written down what you tried and why it did not work. A
documented negative result is worth real marks under Innovation and is not a
failed track.

---

## Track D: Prove the LLM stages

Two LLM stages are built, tested, and have produced almost nothing. This track
turns them from a claim into evidence, and it is the direct answer to the
concern Jarell raised about the project not making LLM calls.

**Files you own:** `src/language/`, `src/semantic.py`, `offline/`,
`artifacts/`, and only the `use_llm`, `llm_model`, `llm_timeout_seconds` and
`weight_appeal` fields of `src/config.py`.

**Three jobs, in order:**

1. **Run the live reranker against a real key, once.** It has never been
   executed. It is fully built (`src/language/rerank.py`), reads the actual
   conversation, reorders the top 20, and fails safe to the input order. Cost
   for a full 200-session run: **$0.64 on Haiku 4.5**, $1.29 on Sonnet 5, $3.22
   on Opus 5. That is sixty-four cents to replace a guess with a fact.

   Set `use_llm=True` and run `--split train`. **Check `total_tokens` is
   non-zero in the output.** If the score comes back as exactly the
   deterministic number, the call silently fell back and you measured nothing.

   Be prepared for it to lose. The simulated customer speaks in verbatim catalog
   substrings rather than natural language, so exact phrase matching may simply
   beat semantic reasoning here, and the reranker only sees the top 20 so it
   cannot find anything new. If it loses, leave it off by default and write down
   the number. "We built it, measured it, and it did not help on this simulator"
   is a real finding and reads far better than an unmeasured stage.

2. **Decide the semantic prior's fate.** It currently covers **60 products out
   of 50,000**, which is 0.12%, and contributes 0.0005 to the score. As shipped
   it is a demonstration, not a contributor. A full catalog pass costs about
   $13.44 on Haiku 4.5 through the Batches API. Only spend that if you can first
   show a measurable direction on a larger sample, say 2,000 products for about
   $0.54. If it does not scale, say so and leave the sample in as an honest
   demonstration of the pipeline.

3. **Produce the disclosure numbers**, which the rules require and which Track A
   needs: model choice, estimated cost, token usage, latency. Give them the
   shipped-path figures (zero tokens, 1m40s) and the optional-path figures from
   job 1.

**Done when:** every LLM stage in the repo has a real measured number attached
to it, and the default configuration is whatever those numbers justify.

---

## Coordination

**File ownership at a glance:**

| Track | Owns |
| --- | --- |
| A | `README.md`, `docs/devpost.md` |
| B | `docs/demo-script.md` |
| C | `src/rank/`, `src/policy/`, `src/state/`, `src/retrieval/`, `evaluation/rank_diagnostics.py` |
| D | `src/language/`, `src/semantic.py`, `offline/`, `artifacts/` |

**The one shared file is `src/config.py`.** C and D both touch it, in different
sections. C owns the retrieval, dialogue and ranking blocks; D owns the four
LLM fields. Git will usually merge these cleanly, but pull before you push.

**Nobody edits `docs/consolidation.md`** during the sprint. It is the shared
reference point, and a document four people are editing is not a reference
point. Update it once at the end.

**Nobody edits anything under `techjam-conversational-search/evaluator/` or
`data/`.** The submission rules forbid it, and `tests/test_entry_point.py` pins
SHA256 hashes of the evaluator files, so a test will fail loudly if anyone
tries.

### Before every push

```
cd techjam-conversational-search && python3 -m evaluator.local_evaluator
python -m pytest tests/ -q
python evaluation/verify_offline.py
```

Expect 0.893583 or better, 89 tests passing, and PASS. If the score drops,
something regressed and it is yours to find.

### Before submission, once

Clone the repository into a clean directory, follow the README exactly as
written, and run the official command. Do not use your working copy. This is the
check that caught the worst bug we had: `artifacts/` was gitignored, so a
judge's fresh clone silently had no LLM stage at all and scored 0.893083 with
nothing failing to explain why.

---

## Open pull requests

**#9, "Field aware retrieval"** by Jarell, open against `main` since 31 August.

It should be closed rather than merged, and Jarell should be the one to close
it. **All of its code is already on `main`.** It was brought over in PR #10:
`src/interfaces.py` and `evaluation/rank_diagnostics.py` are byte-identical to
his versions, and `src/retrieval/baseline.py` on `main` is his file with the
pooled route added on top. What remains unmerged is only his edits to four
documentation files, which were rewritten afterwards.

Merging it now would be actively harmful. The branch was cut before PRs #11
through #14, so merging it would delete the offline LLM prior, the live
reranker, the artefact shipping fix and roughly 3,500 lines of the platform.

The branch shows 7 unmerged commits, which reads like the work was dropped. It
was not. Worth saying out loud rather than leaving a stale PR to imply
otherwise.
