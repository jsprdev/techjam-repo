# CLAUDE.md, competition kit directory

Rules for working inside `techjam-conversational-search/`. The repo root `CLAUDE.md`
carries the problem statement and the wider conventions; this file covers what is
specific to the organiser's kit and to a repo four people are editing at once.

---

## 1. This is a live four-person repo. Look before you write.

Somebody else has almost certainly touched this since you last looked. Before starting
anything, and again before you push:

```bash
git fetch --prune origin
git log --oneline origin/main -10
git for-each-ref --sort=-committerdate --format='%(committerdate:relative) %(authorname) %(refname:short)' refs/remotes/origin | head
```

Also check open pull requests. If someone has a branch touching the module you are about
to edit, talk to them rather than racing them. Module ownership is in `docs/handoff.md`,
and the seams between modules are frozen in `src/interfaces.py`: anyone may propose a
change there, nobody changes one silently.

Rebase or merge `main` in before you push. A branch that is a day stale is a merge
conflict waiting to happen.

## 2. What must never break

**Never modify anything under `evaluator/`.** The submission rules forbid it and treat a
modified run as invalid. `tests/test_entry_point.py` pins the file hashes and fails if
anyone does, including on a committed change.

**`starter/agent.py` must keep re-exporting `src.agent.Agent`.** The evaluator hardcodes
`from starter.agent import Agent` with no override. This was already wrong once: the file
held the organiser's BM25 baseline while our own runner imported `src.agent` directly, so
every local number measured a code path the graded command never took. If the score
suddenly reads about 0.107, check this first.

**`reset()` must never raise.** The evaluator wraps `respond()` in try/except but calls
`reset()` bare, so one exception there aborts all 200 sessions rather than costing a turn.

**`respond()` must never raise and must return a dict whose `message` is a string.**
Otherwise the evaluator discards the whole response, recommendations included.

**Never let a graded path touch the network.** Official scoring may run with networking
disabled. `python evaluation/verify_offline.py` proves it, and is negative-controlled.

**The catalog is read only.** No mutation, no injected ASINs. Derived files go in
`artifacts/`.

## 3. Do not introduce regressions

The score is the product. Before every push:

```bash
pytest                                    # ~2s
python evaluation/run_eval.py --split train   # ~2min, compare against the last number
```

If your change moves the score down, it does not ship, however good the idea was. Three
changes have already been reverted on exactly that basis, all of them things that
obviously should have helped. Measure, do not reason.

Never tune against `--split holdout`. Those 40 sessions are the only evidence we did not
simply fit the public simulator, and that evidence is spent the moment a number from them
influences a decision.

## 4. Keep it modular, do not over-engineer

One module, one job, behind the seam in `src/interfaces.py`. If your change needs to reach
into another module's internals, the seam is wrong; fix the seam and say so, do not reach
through it.

Do not add abstract base classes, registries, plugin systems or factories for a single
implementation. Do not add a config framework. Do not cache before profiling. Every
tunable goes in `src/config.py` with a comment recording what moving it did, so the next
person is reading a measurement rather than guessing at a constant.

Tests earn their place by catching something that would cost score or corrupt a
measurement. Nothing else. The suite was cut from 81 to 44 on that basis and is better for
it. Note that `respond()` degrades to a popularity guess on any failure, so a test whose
only assertion is "the response was well formed" passes on a completely dead pipeline: use
the `strict_agent` fixture for anything asserting your module works.

## 5. Build the thing that shows what we added

The submission is judged on whether a human can see the value, not on the number alone.
TechnicalScore is one input to Technical Execution, which is 35%; the writeup, the demo
and the reasoning carry the rest.

So when you finish a piece of work, be able to say in one sentence what it added and show
the measurement. "Ordering the asks by measured answerability took MTTC from 4.23 to 3.33,
because three of the nine attributes can never be answered and we were asking two of them
first" is worth more than a bare number, and it is the same sentence that goes in the
writeup.

Prefer the change a judge can follow over the change that is merely clever. A deliberate
deviation you can explain and measure beats an unexplained gain.

## 6. Writing style

Never use em dashes, anywhere: code comments, docstrings, commit messages, the README, the
Devpost description, generated user-facing text. Use a comma, a colon, parentheses, or two
sentences. Keep prose plain and direct.
