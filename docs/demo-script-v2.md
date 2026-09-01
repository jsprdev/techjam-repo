# Demo video script, pillar version

Version 1 is kept at `docs/demo-script.md`. This version is organised around the
four pillars in the problem statement, so a judge can see each requirement
answered on screen.

Video URL: `ADD_YOUTUBE_URL_AFTER_UPLOAD`

## Length

The rules set no time limit. The problem statement asks only for "a short video
that demonstrates your solution working end-to-end". Target **2 minutes 30**.

## Before recording

```bash
git pull origin main
pip install -r requirements.txt
python3 evaluation/run_eval.py --split train --traces artifacts/traces.json
```

The trace run takes about five minutes and produces the file the scenes read.
Then create the two helper scripts. Copy this whole block and paste it once:

```bash
mkdir -p artifacts
python3 - <<'MAKE'
from pathlib import Path
Path("artifacts/scene.py").write_text('''import json, sys
sample_id = sys.argv[1] if len(sys.argv) > 1 else "public_0012"
traces = json.load(open("artifacts/traces.json", encoding="utf-8"))
session = next(r for r in traces if r["sample_id"] == sample_id)
with open("techjam-conversational-search/data/public_set.jsonl", encoding="utf-8") as f:
    samples = {r["sample_id"]: r for r in map(json.loads, f)}
target = samples[sample_id]["ground_truth"]["parent_asin"]
print(f"session {sample_id}    scenario: {samples[sample_id]['scenario_type']}\\n")
for t in session["turns"]:
    e = t["extra"]
    top = t["top_recommendations"]
    rank = top.index(target) + 1 if target in top else "outside top 10"
    print(f"TURN {t['turn']}  customer: {t['user_message']}")
    print(f"   I   route={e['track']}  pool={e['width']}")
    print(f"   II  pool_overloaded={e['overloaded']}  asks={t['ask_attribute']}  pivot={e['pivot']}")
    print(f"   III belief_entropy={e['entropy']}  retired={e['retired'] or 'none'}  distilled={e['compression']}")
    print(f"   IV  target_rank={rank}\\n")
''')
Path("artifacts/pillar3.py").write_text('''import json
traces = json.load(open("artifacts/traces.json", encoding="utf-8"))
sessions = len(traces)
retired = sum(1 for r in traces if r["turns"][-1]["extra"]["retired"])
pivots = sum(1 for r in traces if any(t["extra"]["pivot"] for t in r["turns"]))
overload = sum(1 for r in traces if any(t["extra"]["overloaded"] for t in r["turns"]))
comp = [t["extra"]["compression"] for r in traces for t in r["turns"]]
ents = [t["extra"]["entropy"] for r in traces for t in r["turns"]]
print(f"train sessions                                  {sessions}")
print(f"sessions that retired an unproductive attribute {retired}")
print(f"sessions where the over-generality cutoff fired {overload}")
print(f"sessions that detected an intent override       {pivots}")
print(f"mean context distillation ratio                 {sum(comp)/len(comp):.3f}")
print(f"mean belief entropy across all turns            {sum(ents)/len(ents):.3f}")
''')
print("created artifacts/scene.py and artifacts/pillar3.py")
MAKE
```

Then dry run each one and clear the terminal.

Everything lives under `artifacts/`, which the repository already ignores, so
nothing here gets committed and the paths work the same on macOS, Linux and
Windows. Run every command from the repository root, because the scripts open
`artifacts/traces.json` and the public session file by relative path.

Time one evaluator run on the recording machine before fixing scene lengths. It
takes about 25 seconds on an Apple silicon laptop and several minutes on a
loaded cloud instance.

---

## Scene plan

| Scene | Length | What it does |
| --- | --- | --- |
| 1 | 40 s | What we made, and the idea the build was organised around |
| 1b | 25 s | The five stages, optional if time is tight |
| 2 | 45 s | One real conversation, found in two turns |
| 3 | 40 s | The shopper changes their mind and the agent recovers |
| 4 | 25 s | How it adapts across 160 conversations |
| 5 | 40 s | The score, and the sessions we held back |

About three minutes with scene 1b, two minutes thirty without it. Narration runs
at roughly 150 words per minute.

The pillar labels stay on screen through scenes 2 to 4, so a judge can follow
the brief's four pillars while the narration talks about the product.

---

## Scene 1. What we made. 40 seconds.

Title card:

```text
Conversational Shopping Copilot
TikTok TechJam 2026, Problem Statement 4

A shopper who half knows what they want, and an agent
that asks the one question that finds it fastest.
```

Narration, 150 words:

> Most product search assumes you can already describe what you want. Real
> shopping works the other way around. You know roughly what you are after, you
> recognise the right thing when you see it, and the words come later.
>
> So we built a shopping copilot that holds a conversation. It reads what you
> have told it so far, works out what it still does not know, and asks the one
> question that narrows fifty thousand products fastest.
>
> That idea drove the whole build. We measured which questions a shopper can
> actually answer before writing any policy, and the answer reshaped the system.
> Asking about a feature gets a useful reply in ninety six percent of
> conversations. Asking about size gets one in twenty two. Three of the ten
> attributes we were allowed to ask about can never be answered at all, so every
> turn spent on them is a turn thrown away.
>
> So the agent asks in that order, and it stops asking anything that stops paying.

Say over the title card, then cut straight to the product working.

## Scene 1b. How it is put together. 25 seconds.

Optional if you have room. Otherwise fold the first two sentences into Scene 2.

```text
route  ->  retrieve  ->  believe  ->  decide  ->  ask
```

Narration, 85 words:

> Underneath, every turn runs the same five stages, and the shape of each one is
> chosen fresh rather than fixed at the start.
>
> The agent decides whether this turn is browsing or buying, pulls candidates
> from an in memory index over the catalog, scores them into a belief about which
> product is meant, checks whether that belief is decided enough to answer, and
> if it is not, picks the question worth asking.
>
> Nothing runs on a server and nothing calls a model while it is being scored.

## Scene 2. One real conversation. 45 seconds.

```bash
python3 artifacts/scene.py public_0012
```

Verified output:

```text
session public_0012    scenario: browsing

TURN 1  customer: I'm looking for Women Dresses, but I'm still exploring.
   I   route=browsing  pool=800
   II  pool_overloaded=True  asks=feature  pivot=False
   III belief_entropy=0.9211  retired=none  distilled=0.4727
   IV  target_rank=outside top 10

TURN 2  customer: For that, what matters is: Imported; Wrap closure.
   I   route=buying  pool=200
   II  pool_overloaded=False  asks=material  pivot=False
   III belief_entropy=0.8621  retired=none  distilled=0.4571
   IV  target_rank=1
```

Narration, 130 words:

> Here is that idea running. The labels down the left mark which of the brief's
> four pillars each line belongs to, so you can follow both at once.
>
> The shopper opens without a clear target, so the agent treats the turn as
> browsing and casts a wide net, eight hundred candidates out of fifty thousand.
> The pool comes back crowded, meaning nothing has separated yet, and this is
> where most systems return a weak top ten and hope. Instead the agent asks about
> a feature, which is the question the measurement said pays best.
>
> The shopper answers. The agent now reads the turn as buying, tightens to two
> hundred candidates, and the product they are actually looking for moves from
> nowhere to first place.
>
> One question, and the right answer out of fifty thousand.

## Scene 3. When the shopper changes their mind. 40 seconds.

```bash
python3 artifacts/scene.py public_0002
```

Verified output:

```text
TURN 2  customer: For that, what matters is: Imported; Buckle closure.
   II  pool_overloaded=True  asks=material  pivot=False
   IV  target_rank=3

TURN 3  customer: Actually, ignore my earlier preference. What I need is: leather.
   II  pool_overloaded=False  asks=color  pivot=True
   IV  target_rank=2
```

Narration, 105 words:

> The brief calls this Intent Override, and it is the case that breaks a system
> built on hard filters, because a filter that has already removed the answer has
> no way back to it.
>
> Our constraints demote candidates and always leave them in the pool, so the
> target survives the pivot and returns at rank two. The pivot flag on turn three
> shows the state machine detecting the change of direction.
>
> The brief describes this as slot erasure. We demote the abandoned preference
> instead, because we measured that in thirty of thirty of these sessions it
> remains true of the product the shopper eventually buys.

## Scene 4. It adapts as the conversation goes on. 25 seconds.

```bash
python3 artifacts/pillar3.py
```

Verified output:

```text
train sessions                                  160
sessions that retired an unproductive attribute 23
sessions where the over-generality cutoff fired 79
sessions that detected an intent override       24
mean context distillation ratio                 0.754
mean belief entropy across all turns            0.896
```

Narration, 90 words:

> Pillar Three asks the agent to adapt at runtime and to distil the conversation
> as it accumulates.
>
> Across a hundred and sixty sessions, the agent retired an attribute in
> twenty-three of them, meaning it asked about something, learned the shopper had
> nothing to say about it, and stopped asking. The over-generality cutoff fired
> in seventy-nine. It distilled each conversation to about three quarters of its
> raw length before querying, and it carries a belief distribution over the
> catalog that it updates every single turn.

## Scene 5. Does it actually work. 40 seconds.

```bash
cd techjam-conversational-search
python3 -m evaluator.local_evaluator --catalog data/catalog.jsonl
```

Leave the run uncut. Narration, 115 words:

> Pillar Four is the measurement, and this is the organiser's own evaluator
> running unmodified against two hundred conversations.
>
> Coverage, Hit Rate at ten, is 97.5 percent. Precision, measured as mean
> reciprocal rank, is 0.78. Efficiency, mean turns to conversion, is 2.39 against
> the baseline's 9.81. TechnicalScore is 0.893583 where the organiser's baseline
> scores 0.1067. Token usage is zero and the run costs nothing, because the
> language model stage runs offline ahead of time and the result ships as a
> committed artefact.
>
> Forty of these sessions were held back from day one and no tuning decision ever
> saw them. Spent once at the end, they score 0.8801, with coverage identical on
> both halves.

---

## The two scripts

`artifacts/scene.py`:

```python
import json, sys
sample_id = sys.argv[1] if len(sys.argv) > 1 else "public_0012"
traces = json.load(open("artifacts/traces.json", encoding="utf-8"))
session = next(r for r in traces if r["sample_id"] == sample_id)
with open("techjam-conversational-search/data/public_set.jsonl", encoding="utf-8") as f:
    samples = {r["sample_id"]: r for r in map(json.loads, f)}
target = samples[sample_id]["ground_truth"]["parent_asin"]
print(f"session {sample_id}    scenario: {samples[sample_id]['scenario_type']}\n")
for t in session["turns"]:
    e = t["extra"]
    top = t["top_recommendations"]
    rank = top.index(target) + 1 if target in top else "outside top 10"
    print(f"TURN {t['turn']}  customer: {t['user_message']}")
    print(f"   I   route={e['track']}  pool={e['width']}")
    print(f"   II  pool_overloaded={e['overloaded']}  asks={t['ask_attribute']}"
          f"  pivot={e['pivot']}")
    print(f"   III belief_entropy={e['entropy']}  retired={e['retired'] or 'none'}"
          f"  distilled={e['compression']}")
    print(f"   IV  target_rank={rank}\n")
```

`artifacts/pillar3.py`:

```python
import json
traces = json.load(open("artifacts/traces.json", encoding="utf-8"))
sessions = len(traces)
retired = sum(1 for r in traces if r["turns"][-1]["extra"]["retired"])
pivots = sum(1 for r in traces if any(t["extra"]["pivot"] for t in r["turns"]))
overload = sum(1 for r in traces if any(t["extra"]["overloaded"] for t in r["turns"]))
comp = [t["extra"]["compression"] for r in traces for t in r["turns"]]
ents = [t["extra"]["entropy"] for r in traces for t in r["turns"]]
print(f"train sessions                                  {sessions}")
print(f"sessions that retired an unproductive attribute {retired}")
print(f"sessions where the over-generality cutoff fired {overload}")
print(f"sessions that detected an intent override       {pivots}")
print(f"mean context distillation ratio                 {sum(comp)/len(comp):.3f}")
print(f"mean belief entropy across all turns            {sum(ents)/len(ents):.3f}")
```

The target product is read only to print its rank, after the agent has already
answered. The agent itself never sees it.

## Numbers to say

TechnicalScore 0.893583 against a baseline of 0.1067. Hit Rate at ten 0.975,
MRR 0.7796, MTTC 2.39 against 9.81. Held out 0.8801. Token usage zero, cost
zero, 97 tests.

## Before uploading

Check the frame for API keys, notifications and personal paths. Upload to
YouTube as Public, confirm playback while signed out, and put the URL at the top
of this file for whoever submits the Devpost description.
