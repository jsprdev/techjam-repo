# Three-minute demo video script

Target length: **2 minutes 50 seconds**. The official rules cap the demo video at
3 minutes, so 2:50 leaves headroom for the cut points rather than risking a
disqualifying overrun.

Final public video URL: `ADD_YOUTUBE_URL_AFTER_UPLOAD`

Use only a terminal and plain text. Do not show product images, company logos,
API keys, notifications, or licensed music.

## Official requirements this script satisfies

From the organiser's `problem-statement.md` and the Devpost listing:

- Maximum 3 minutes, uploaded to YouTube, set to public visibility
- Demonstrates the solution working end to end
- Linked from the Devpost written description
- No third-party trademarks or copyrighted content, music included
- Backend track note: with no front end, a walkthrough of API usage, inference
examples and result analysis is explicitly accepted, which is what this is

---

## Measured runtimes, so the scene budgets are real

Every number below was measured on the recording machine (Apple silicon, Python
3.12.13, numpy 2.5.2, scikit-learn 1.9.0) in a fresh clone at `~/techjam-demo`.
Wall clock varies a lot with the machine and its load: the same commit has
timed at 25 seconds here and 7 minutes on a loaded shared cloud box. Re-time on
the actual recording machine before cutting the scene lengths, and quote the
latency in the Devpost description with the hardware beside it.


| Command                              | Wall clock                                        |
| ------------------------------------ | ------------------------------------------------- |
| `pytest tests/`                      | 0.9 s warm, 10.9 s on the first run in a new venv |
| Official evaluator, all 200 sessions | **24.9 s**                                        |
| `run_eval.py --split train --traces` | 22.7 s                                            |
| `verify_offline.py`                  | 8.3 s                                             |


**The official run takes 25 seconds here, against 1 minute 40 on another
machine and 7 minutes on a loaded shared cloud box.** Wall clock is a property
of the hardware, not of the code. The scene breakdown below is
built around 25 seconds; do not pad narration to fill time that no longer exists.

## Numbers to say on camera

These are what this machine produces. Say these and nothing else.


| Metric                    | Value         |
| ------------------------- | ------------- |
| TechnicalScore            | **0.893583**  |
| Hit Rate@10               | 0.975         |
| MRR                       | 0.779609      |
| MTTC                      | 2.39 turns    |
| Efficiency                | 0.861         |
| Reported token usage      | 0             |
| Baseline TechnicalScore   | 0.1067        |
| Baseline MTTC             | 9.81          |
| Browsing MRR / buying MRR | 0.675 / 0.828 |


**Consistency gate: closed.** This script previously said to expect 0.892323 on
the recording machine and 0.893583 elsewhere, and to pin versions before
recording. The drift was ours, not the dependencies': `_top_k` selected the
candidate pool with `np.argpartition`, which orders tied scores however its
internal introselect leaves them. TF-IDF puts most of the catalog at exactly
0.0, so a wide shortlist is mostly tied items, and 35 of 800 slots resolved
differently between numpy builds.

Ties now break by catalog index, which the frozen catalog fixes. Verified on
three environments, all producing **0.893583** exactly:

| Environment | Score |
| --- | --- |
| Python 3.11, numpy 2.2.6, scikit-learn 1.9.0 | 0.893583 |
| Python 3.11, numpy 2.4.6, scikit-learn 1.9.0 | 0.893583 |
| Python 3.12.3, numpy 2.5.2, scikit-learn 1.9.0 | 0.893583 |

So 0.893583 is the number to say on camera, and it is the number the README,
the Devpost description and a judge's fresh clone all produce. Say it with
confidence; it is no longer machine dependent.

## Holdout note

The all-200 official aggregate has already been run and recorded, so Scene 2 is
safe to record now. What must not happen before the final merge is
`python evaluation/run_eval.py --split holdout`, which is a different command and
is not in this script. Use `--split train` for anything exploratory.

---

## 1. Prepare

Record from a fresh clone, not the working copy. This removes the working
directory's trailing space from the on-screen prompt, and it doubles as the
judge simulation required before submission: it proves a clone reproduces the
score without anything local propping it up.

```bash
git clone <repo-url> ~/techjam-demo
cd ~/techjam-demo
python3 -m venv techjam-conversational-search/.venv   # 3.11, 3.12 or 3.13
source techjam-conversational-search/.venv/bin/activate
pip install -r requirements.txt
```

Then warm the caches and generate the traces Scene 3 reads. Do this **before**
recording: the first `pytest` in a new venv takes 10.9 seconds while it compiles
bytecode, and 0.9 seconds every time after.

```bash
git status --short
python3 -m pytest tests/ -q
python3 evaluation/run_eval.py \
  --split train \
  --traces /tmp/techjam-demo-traces.json \
  --output /tmp/techjam-demo-train-results.json
```

`git status` must print nothing. Commit or stash any working-copy edits before
cloning, or the clone will not contain them. 89 tests must pass, and the trace
file must hold 160 sessions.

Canvas: 1080p, terminal font at least 20 pixels, 30 frames per second.

---

## 2. Scene breakdown


| Scene | Window      | Length | On screen                                  |
| ----- | ----------- | ------ | ------------------------------------------ |
| 1     | 0:00 – 0:14 | 14 s   | Title card                                 |
| 2     | 0:14 – 0:52 | 38 s   | Official evaluator, uncut, plus the result |
| 3     | 0:52 – 1:32 | 40 s   | One real conversation, traced turn by turn |
| 4     | 1:32 – 2:02 | 30 s   | The rerank-depth sweep                     |
| 5     | 2:02 – 2:32 | 30 s   | Offline proof and required disclosures     |
| 6     | 2:32 – 2:50 | 18 s   | Honest limits and what is next             |


Narration is written to roughly 150 words per minute. Word counts are given per
scene so you can check pacing before recording rather than after.

### Scene 1, title card, 0:00 to 0:14

```text
Conversational Shopping Agent
Intent routing -> in-memory retrieval -> semantic ranking
50,000 products, at most 10 turns
TechnicalScore: 0.893583      Baseline: 0.1067
```

Narration, 35 words:

> Traditional e-commerce search matches keywords. Real shoppers change their
> minds mid-conversation. This is a shopping agent that finds one hidden product
> in fifty thousand, in at most ten turns. The organiser's BM25 baseline scores
> 0.1067.

### Scene 2, the official run, 0:14 to 0:52

The run itself is 25 seconds. Keep it uncut, and let the result sit on screen
while you read the metrics off it.

```bash
cd techjam-conversational-search
/usr/bin/time -p python3 -m evaluator.local_evaluator \
  --catalog data/catalog.jsonl \
  --output /tmp/techjam-demo-official-results.json
```

Narration while it runs, 56 words, about 22 seconds. It is written to cover the
full 25-second run so there is no dead air waiting for the result:

> This is the organiser's own evaluator, unmodified, driving our agent through
> two hundred simulated conversations against the read-only catalog. It is the
> exact path that gets graded, not a substitute metric. Every one of those
> conversations runs in memory, on CPU, with no network access, because the rules
> say official scoring may disable the network entirely.

When the JSON appears, point at each figure as you say it. Kept deliberately
short, 25 words, so the pointing rather than the talking fills the time:

> Twenty-five seconds. Hit Rate at ten, 97.5 percent. MRR, 0.775. Turns to
> conversion, 2.39 against the baseline's 9.81. TechnicalScore, 0.893583 against
> 0.1067. Token usage, zero.

Return to the repository root:

```bash
cd ..
```

### Scene 3, one real conversation, 0:52 to 1:32

This is the most persuasive scene, which is why it now gets 40 seconds instead
of 30. The architecture is explained here, anchored to output on screen, rather
than narrated over a progress spinner.

```bash
python3 - <<'PY'
import json

sample_id = "public_0012"
traces = json.load(open("/tmp/techjam-demo-traces.json", encoding="utf-8"))
session = next(row for row in traces if row["sample_id"] == sample_id)
with open("techjam-conversational-search/data/public_set.jsonl", encoding="utf-8") as source:
    samples = {row["sample_id"]: row for row in map(json.loads, source)}
target = samples[sample_id]["ground_truth"]["parent_asin"]

for turn in session["turns"]:
    top = turn["top_recommendations"]
    rank = top.index(target) + 1 if target in top else "outside top 10"
    print(f"TURN {turn['turn']}: {turn['user_message']}")
    print(
        f"route={turn['extra']['track']} width={turn['extra']['width']} "
        f"overloaded={turn['extra']['overloaded']} asks={turn['ask_attribute']} "
        f"target_rank={rank}\n"
    )
PY
```

Verified output:

```text
TURN 1: I'm looking for Women Dresses, but I'm still exploring.
route=browsing width=800 overloaded=True asks=feature target_rank=outside top 10

TURN 2: For that, what matters is: Imported; Wrap closure.
route=buying width=200 overloaded=False asks=material target_rank=1
```

Narration, 100 words:

> One real session. The customer opens vague: women's dresses, still exploring.
> The router reads that as browsing, opens an eight-hundred product route for
> coverage, and finds the candidate pool overloaded. So instead of guessing, it
> asks which features matter.
>
> The answer, imported and wrap closure, flips the router to buying. The route
> narrows to two hundred for precision, the overload clears, and the target moves
> from outside the top ten to rank one.
>
> Nothing is hard filtered. Constraints demote candidates rather than deleting
> them, so when a shopper contradicts themselves the agent recovers instead of
> returning an empty list.

Ground truth is joined only for this offline explanation. Say that if there is
room; drop it if the scene is running long.

### Scene 4, a measured design choice, 1:32 to 2:02

```bash
sed -n '/^### 4\.4 /,/^### 4\.5 /p' docs/consolidation.md | sed '$d'
```

This prints the rerank-depth table: 100 scores 0.8540, 200 scores 0.8951, 400
scores 0.8737, 800 scores 0.8670.

Narration, 78 words:

> Every parameter here was swept, not argued. Rerank depth: at one hundred we
> lose targets that retrieval had already found. At four hundred and eight
> hundred, extra plausible products dilute rank one. Two hundred is the peak,
> worth about four points of score.
>
> What matters more is that this sweep reversed once we added exact phrase
> overlap. Re-running it after changing the ranker was not optional. Six other
> ideas were measured the same way and rejected.

### Scene 5, offline proof and disclosure, 2:02 to 2:32

The probe takes 8 seconds, so it fits comfortably inside the scene.

```bash
python3 evaluation/verify_offline.py
```

Leave these lines visible:

```text
OK 10 turns completed with every socket blocked
[offline] PASS. Safe to submit as running fully offline.
```

Narration, 76 words:

> The rules say official scoring may run with network access disabled, so we
> tested that rather than assuming it. This probe poisons every socket entry
> point and runs a full ten-turn session. It is negative-controlled: inject a
> real network call and the check goes red.
>
> Disclosure: the shipped path uses zero tokens and costs zero dollars. An
> offline Haiku 4.5 artifact is a local lookup at run time. An optional live
> reranker fails safe to deterministic order.

### Scene 6, close, 2:32 to 2:50

Stay on the terminal. No new command.

Narration, 46 words:

> Two honest limits. This simulator quotes the target's own catalog text, so
> lexical matching flatters us more than real paraphrasing would. And browsing
> MRR is 0.675 against buying's 0.828, across half the sessions. That gap is
> where the remaining score is, and where we go next.

---

## 3. Publish

1. Confirm the total is under 3 minutes. This is a hard cap in the rules.
2. Keep the evaluator run uncut. Remove only dead air between scenes.
3. Confirm every command and metric is readable at normal playback speed.
4. Export 1080p H.264 with AAC audio and no unlicensed music.
5. Check the final cut for secrets, notifications, personal paths, images and
  logos. The prompt should read `~/techjam-demo`, never a home directory with a
   real name or a stray trailing space in it.
6. Confirm the score spoken on camera matches the README and the Devpost
  description exactly.
7. Upload to YouTube as **Public** and verify playback while signed out.
8. Replace `ADD_YOUTUBE_URL_AFTER_UPLOAD` above with the public URL and give the
  same URL to whoever is writing the Devpost description.

