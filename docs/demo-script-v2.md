# Demo video script, short version

Version 1 is kept at `docs/demo-script.md`. This is the simple cut: show the
product working, say what we built, quote the number, stop.

Video URL: `ADD_YOUTUBE_URL_AFTER_UPLOAD`

## Length

**No time limit exists in the rules.** The problem statement says only "submit a
short video that demonstrates your solution working end-to-end". V1's claim of a
3 minute cap is invented.

**Target 2 minutes.** Four scenes.

## Before recording

```bash
python3 evaluation/run_eval.py --split train --traces /tmp/traces.json
```

Takes about 5 minutes, nothing to watch, do it beforehand. Save the script at
the bottom of this file to `/tmp/scene.py`.

---

## Scene 1. What it is. 15 seconds.

```text
Conversational Shopping Agent
TikTok TechJam 2026, Problem Statement 4

50,000 products.  At most 10 turns.  No network at scoring time.
```

> The brief says keyword search fails real shoppers, who browse before they buy
> and change their minds halfway through. We built an agent that finds one
> hidden product in fifty thousand by talking to them. Here it is working.

## Scene 2. The product working. 40 seconds.

```bash
python3 /tmp/scene.py public_0012
```

```text
TURN 1  customer: I'm looking for Women Dresses, but I'm still exploring.
         agent asks: feature   route=browsing   pool_overloaded=True
         target rank: outside top 10

TURN 2  customer: For that, what matters is: Imported; Wrap closure.
         agent asks: material   route=buying   pool_overloaded=False
         target rank: 1
```

> The shopper opens vague. The router reads it as Browsing, and the candidate
> pool comes back overloaded, which is the over-generality condition in Pillar
> Two. So instead of guessing, the agent asks which feature matters.
>
> That answer flips the router to Buying, the overload clears, and the target
> goes from outside the top ten to rank one. One question, one product out of
> fifty thousand.

## Scene 3. It recovers when they change their mind. 35 seconds.

```bash
python3 /tmp/scene.py public_0002
```

```text
TURN 2  customer: For that, what matters is: Imported; Buckle closure.
         target rank: 3

TURN 3  customer: Actually, ignore my earlier preference. What I need is: leather.
         target rank: 2
```

> This is Intent Override in the brief. A system that filters hard has already
> deleted the answer. Ours never filters: constraints demote candidates, they
> never remove them, so the target survives the pivot.
>
> Under this sit the four pillars. Buying and Browsing routing, and in-memory
> retrieval over fifty thousand products with no database. The slot state
> machine and the cutoff you just watched fire. A belief the agent updates every
> turn. And the full metric set, per scenario. The LLM ranking stage the brief
> names, we run offline and commit the result, so the graded run needs no
> network call.

## Scene 4. The number. 30 seconds.

```bash
cd techjam-conversational-search
python3 -m evaluator.local_evaluator --catalog data/catalog.jsonl
```

> The organiser's evaluator, unmodified, two hundred conversations. Hit Rate at
> ten, 97.5 percent. Turns to conversion, 2.39 against the baseline's 9.81.
> TechnicalScore 0.893583 against 0.1067. Zero tokens, no API key.
>
> Forty sessions were held out from day one and spent once at the end: 0.8801,
> against 0.8969 on what we tuned on. The gap we know about is browsing, where
> we find the product every time and rank it second.

---

## Numbers to say

TechnicalScore **0.893583**, baseline 0.1067. HitRate@10 0.975, MRR 0.7796,
MTTC 2.39 against 9.81. Held out 0.8801. Tokens 0, cost $0.00, 97 tests.

## The scene script

Save as `/tmp/scene.py`:

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

Ground truth is read only to print the rank, after the agent has answered. The
agent never sees it.

## Before uploading

No keys, notifications or personal paths on screen. YouTube, Public. Check
playback signed out. Put the URL at the top of this file.
