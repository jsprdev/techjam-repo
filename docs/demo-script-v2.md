# Demo video script, version 2

Version 1 is kept at `docs/demo-script.md`. This one is a different film. V1 is a
methodology video: it spends 30 seconds on a hyperparameter sweep table and
never once names a pillar from the problem statement. V2 shows the product
working, says what we built against each pillar, and uses the numbers as
evidence rather than as the subject.

Final public video URL: `ADD_YOUTUBE_URL_AFTER_UPLOAD`

---

## How long should this be?

**There is no time limit in the rules.** The problem statement says only:

> Submit a short video that demonstrates your solution working end-to-end.

No maximum is stated anywhere in the problem statement, the participant kit, or
the kit docs. V1 claims "the official rules cap the demo video at 3 minutes" and
lists "Maximum 3 minutes" as an official requirement. **That cap is invented.**
If Devpost's own listing states one, follow that instead and tell the team,
because nothing in the material we were given does.

**Target 3 minutes to 3 minutes 30.** Reasoning rather than convention:

- Four pillars, plus a working demo, plus the numbers, does not compress below
  about three minutes without becoming a list of claims nobody can check.
- Presentation and Communication is 10% of the grade and only applies at the
  final event. Length is not itself scored, so there is no prize for brevity and
  no penalty for a fourth minute. There is a real penalty for a judge losing
  interest.
- The one hard constraint is self-imposed: every claim on camera must be one the
  repository can reproduce. That is what sets the floor.

Do not pad to fill time. If it lands at 2:40 and covers everything, ship it.

## What the rules actually require

From `problem-statement.md`:

- Demonstrates the solution working end to end (inference results, model
  predictions)
- Uploaded to YouTube, set to public visibility
- Linked in the Devpost description
- No third-party trademarks or copyrighted content without permission
- Backend track: with no front end, a walkthrough of API usage, inference
  examples or result analysis is explicitly accepted

Everything below is terminal only, which that last point permits.

---

## Before you record

```bash
git clone <repo-url> ~/techjam-demo
cd ~/techjam-demo
python3 -m venv .venv && source .venv/bin/activate    # Python 3.11, 3.12 or 3.13
pip install -r requirements.txt
```

Recording from a fresh clone doubles as the judge simulation: it proves a clone
reproduces the score with nothing local propping it up.

Then generate the traces Scenes 2 and 3 read. **Do this before recording**, it
takes about five minutes and there is nothing to watch:

```bash
python3 -m pytest tests/ -q                    # 97 tests
python3 evaluation/run_eval.py --split train --traces /tmp/traces.json
```

Save the two scene scripts below to `/tmp/scene.py` before you start, so on
camera you are running one short command and not pasting a heredoc.

**Time one evaluator run on this machine before cutting scene lengths.** It is
about 25 seconds on an Apple silicon laptop and over 7 minutes on a loaded cloud
box. Wall clock is a property of the hardware, not of the code.

Canvas 1080p, terminal font 20px or larger, and check the prompt shows no real
name or personal path.

---

## Scene plan

| Scene | Length | What it does |
| --- | --- | --- |
| 1 | 25 s | The problem, in the problem statement's own terms, and what we built |
| 2 | 45 s | The product working: a vague shopper converging to one product |
| 3 | 40 s | The hard case: the shopper changes their mind |
| 4 | 40 s | What is under it, mapped to the four pillars |
| 5 | 35 s | The official number |
| 6 | 30 s | What we can prove, and what we cannot |

About 3 minutes 15. Narration is written to roughly 150 words per minute.

---

### Scene 1. The problem and the build. 25 seconds.

Title card:

```text
Conversational Shopping Agent
TikTok TechJam 2026, Problem Statement 4

50,000 products.  At most 10 turns.  No network at scoring time.
TechnicalScore 0.893583      Organiser baseline 0.1067
```

Narration, 62 words:

> The brief opens by saying traditional e-commerce search relies on static
> keyword matching, and fails to capture how real shoppers actually behave: the
> shift between open-ended browsing and high-intent buying, and the moment
> someone changes their mind halfway through.
>
> So we built an agent that finds one hidden product among fifty thousand by
> talking to the shopper, in at most ten turns. Here it is working.

### Scene 2. The product working. 45 seconds.

```bash
python3 /tmp/scene.py public_0012
```

Verified output:

```text
session public_0012   scenario: browsing

TURN 1  customer: I'm looking for Women Dresses, but I'm still exploring.
         agent asks: feature   route=browsing   pool_overloaded=True
         target rank: outside top 10

TURN 2  customer: For that, what matters is: Imported; Wrap closure.
         agent asks: material   route=buying   pool_overloaded=False
         target rank: 1
```

Narration, 105 words:

> One real conversation. The shopper opens vague: women's dresses, still
> exploring. Two things happen that the brief asks for by name.
>
> The router reads that turn as Browsing, not Buying. And the candidate pool
> comes back overloaded, which is the over-generality condition in Pillar Two. So
> instead of guessing at a top ten, the agent asks which feature matters.
>
> The answer, imported and wrap closure, flips the router to Buying, the overload
> clears, and the target goes from outside the top ten to rank one.
>
> One question, and the right product out of fifty thousand. That is the whole
> product in two turns.

### Scene 3. The shopper changes their mind. 40 seconds.

```bash
python3 /tmp/scene.py public_0002
```

Verified output:

```text
session public_0002   scenario: intent_override

TURN 1  customer: I'm looking for Accessories Belts. Buckle closure
         agent asks: feature   route=buying   pool_overloaded=False
         target rank: outside top 10

TURN 2  customer: For that, what matters is: Imported; Buckle closure.
         agent asks: material   route=buying   pool_overloaded=True
         target rank: 3

TURN 3  customer: Actually, ignore my earlier preference. What I need is: leather.
         agent asks: color   route=buying   pool_overloaded=False
         target rank: 2
```

Narration, 100 words:

> This is the case the brief calls Intent Override. On turn three the shopper
> abandons what they asked for and says something different. A system that
> filters hard has already deleted the answer and cannot recover.
>
> Ours never filters. Constraints demote candidates, they never remove them, so
> the target survives the pivot and comes back at rank two.
>
> The brief describes this as slot erasure. We demote the old preference rather
> than erasing it, and that is a measured decision: in thirty of thirty of these
> sessions, the preference the shopper abandons is still true of the product they
> end up buying.

### Scene 4. What is under it. 40 seconds.

No command. A single slide, on screen for the whole scene:

```text
PILLAR I    Core architecture
            Per-turn Buying / Browsing routing        policy/intent.py
            In-memory TF-IDF retrieval, 50k products  retrieval/baseline.py
            LLM semantic ranking, computed offline    offline/ + src/semantic.py

PILLAR II   Dialog strategy
            Slot state machine: accumulate, override  state/slots.py
            Over-generality cutoff, then ask          policy/commit.py

PILLAR III  Self-evolution
            Belief distribution with entropy          state/belief.py
            Session distillation, attribute retiring  state/session.py

PILLAR IV   Evaluation
            HitRate@10, MRR, MTTC, per scenario       evaluation/run_eval.py
            97 tests, offline probe, held-out split
```

Narration, 98 words:

> Everything you just saw maps onto the four pillars in the brief.
>
> Pillar One, the routing and the in-memory retrieval, no database anywhere.
> Pillar Two, the slot state machine and the over-generality cutoff you watched
> fire. Pillar Three, a belief distribution the agent updates every turn, and a
> session it distils as evidence arrives. Pillar Four, the full metric set, per
> scenario, plus a held-out split we reserved on day one.
>
> On the LLM ranking stage the brief names: we run it, but offline, ahead of
> time. The result is committed, so the graded run reads it without a network
> call.

### Scene 5. The official number. 35 seconds.

```bash
cd techjam-conversational-search
python3 -m evaluator.local_evaluator --catalog data/catalog.jsonl
cd ..
```

Leave it uncut. Narration while it runs, 55 words:

> This is the organiser's evaluator, unmodified, driving our agent through two
> hundred conversations. It is the path that gets graded, not a substitute
> metric. No network, no API key, no database. The rules say scoring may run with
> networking disabled, so nothing in the shipped path needs it.

Then point at the figures, 30 words:

> Hit Rate at ten, 97.5 percent. MRR, 0.78. Turns to conversion, 2.39, against
> the baseline's 9.81. TechnicalScore 0.893583, against 0.1067. Token usage,
> zero. Cost, nothing.

### Scene 6. What we can prove, and what we cannot. 30 seconds.

```bash
python3 evaluation/verify_offline.py
```

Leave on screen:

```text
OK 10 turns completed with every socket blocked
[offline] PASS. Safe to submit as running fully offline.
```

Narration, 92 words:

> Two things we would rather show than assert.
>
> That probe runs a full session with every socket blocked, because the rules
> allow scoring with no network. And forty of the two hundred sessions were
> reserved on day one. No tuning decision ever saw them. We spent them once, at
> the end: 0.8801 held out, against 0.8969 on the sessions we tuned on, with Hit
> Rate identical on both.
>
> The limit we know about: browsing MRR is 0.675 against buying's 0.828. We find
> the product every time and rank it second. That is where we go next.

---

## The two scene scripts

Save this as `/tmp/scene.py` before recording:

```python
import json, sys
sample_id = sys.argv[1] if len(sys.argv) > 1 else "public_0012"
traces = json.load(open("/tmp/traces.json", encoding="utf-8"))
session = next(r for r in traces if r["sample_id"] == sample_id)
with open("techjam-conversational-search/data/public_set.jsonl", encoding="utf-8") as f:
    samples = {r["sample_id"]: r for r in map(json.loads, f)}
target = samples[sample_id]["ground_truth"]["parent_asin"]
print(f"session {sample_id}   scenario: {samples[sample_id]['scenario_type']}\n")
for t in session["turns"]:
    top = t["top_recommendations"]
    rank = top.index(target) + 1 if target in top else "outside top 10"
    print(f"TURN {t['turn']}  customer: {t['user_message']}")
    print(f"         agent asks: {t['ask_attribute']}   route={t['extra']['track']}"
          f"   pool_overloaded={t['extra']['overloaded']}")
    print(f"         target rank: {rank}\n")
```

The target is read only to print its rank, after the agent has answered. The
agent never sees it. Say that on camera if there is room.

---

## Numbers to say, and nothing else

| Metric | Value |
| --- | --- |
| TechnicalScore | **0.893583** |
| Hit Rate@10 | 0.975 |
| MRR | 0.7796 |
| MTTC | 2.39 turns |
| Efficiency | 0.861 |
| Held out, 40 unseen sessions | 0.8801 |
| Organiser baseline | 0.1067 |
| Baseline MTTC | 9.81 |
| Token usage | 0 |
| Cost per evaluation | $0.00 |
| Tests | 97 |

The score is machine independent, verified across three Python and numpy
combinations. Wall clock is not: quote it with the hardware beside it.

## Corrections carried over from v1

Fixed here, still wrong in `docs/demo-script.md`:

- **The 3 minute cap does not exist.** V1 states it twice, once as an official
  requirement. Nothing in the problem statement or the kit sets a limit.
- **"89 tests must pass"** is stale. There are 97.
- **"MRR, 0.775"** in the Scene 2 narration is stale, from before the tie-break
  fix. It is 0.779609, and v1's own numbers table already says so, so v1
  contradicts itself on camera.

## Before uploading

1. No API keys, notifications, personal paths, product images or brand logos.
2. The score spoken on camera matches the README and the Devpost description.
3. YouTube, Public, and check playback while signed out.
4. Put the URL at the top of this file and give it to whoever submits Devpost.
