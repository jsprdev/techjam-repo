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
Save both scripts at the end of this file to `artifacts/scene.py` and
`artifacts/pillar3.py`, then dry run each one and clear the terminal.

Everything lives under `artifacts/`, which the repository already ignores, so
nothing here gets committed and the paths work the same on macOS, Linux and
Windows. Run every command from the repository root, because the scripts open
`artifacts/traces.json` and the public session file by relative path.

Time one evaluator run on the recording machine before fixing scene lengths. It
takes about 25 seconds on an Apple silicon laptop and several minutes on a
loaded cloud instance.

---

## Scene 1. The four pillars. 20 seconds.

Title card:

```text
Conversational Shopping Agent
TikTok TechJam 2026, Problem Statement 4

I    Intent routing and hybrid retrieval pipeline
II   Multi-turn dialog strategy
III  Self-evolution through context programming
IV   Coverage, precision and efficiency
```

Narration, 55 words:

> The brief describes shopping search that relies on static keyword matching and
> misses how people actually shop. It asks for four things: a pipeline that
> routes on intent, a dialog strategy that survives a shopper changing their
> mind, an agent that adapts as the conversation goes on, and measurement across
> coverage, precision and efficiency. We will take them in order.

## Scene 2. Pillars I and II, working. 45 seconds.

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

Narration, 120 words:

> Every line here is labelled with the pillar it answers.
>
> Pillar One is the routing. The shopper opens without a clear target, so the
> agent routes this turn to the Browsing track and widens the candidate pool to
> eight hundred. Retrieval runs entirely in memory over fifty thousand products
> with no database and no network.
>
> Pillar Two is the dialog strategy. The candidate pool comes back overloaded,
> which is the over-generality condition in the brief. The agent responds by
> asking which feature matters rather than returning a weak top ten.
>
> The shopper answers, the router moves the turn to the Buying track, the pool
> narrows to two hundred, and the target product rises from outside the top ten
> to rank one.

## Scene 3. Pillar II, the shopper changes their mind. 40 seconds.

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

## Scene 4. Pillar III, self-evolution. 25 seconds.

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

## Scene 5. Pillar IV, the metrics. 40 seconds.

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
