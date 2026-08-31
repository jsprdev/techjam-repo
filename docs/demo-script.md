# Three-minute demo video script

Target length: 2 minutes 55 seconds to 3 minutes 10 seconds.

Final public video URL: `ADD_YOUTUBE_URL_AFTER_UPLOAD`

Use only a terminal and plain text. Do not show product images, company logos,
API keys, notifications, or licensed music.

## Holdout gate

The official evaluator uses all 200 public sessions, including the 40 reserved
sessions. Record that scene only after all tracks are merged and the final merge
owner approves spending the holdout. Use only `--split train` while preparing.

## 1. Prepare

Use a 1080p canvas, a terminal font of at least 20 pixels, and 30 frames per
second. From the repository root, activate the environment and run the checks:

```bash
cd /home/blackvader/CodingProjects/tiktoktechjam2026/techjam-repo
source techjam-conversational-search/.venv/bin/activate
git status --short
python3 -m pytest tests/ -q
python3 evaluation/run_eval.py \
  --split train \
  --traces /tmp/techjam-demo-traces.json \
  --output /tmp/techjam-demo-train-results.json
```

The Git status should be empty, 89 tests should pass, and the trace file should
contain 160 train sessions.

## 2. Record

### Scene 1, result, 0:00 to 0:15

Screen:

```text
Conversational Shopping Agent
Intent routing -> in-memory retrieval -> semantic ranking
50,000 products, at most 10 turns
TechnicalScore: 0.893583
Baseline: 0.1067
```

Narration:

> Our answer to the Core Architecture pillar is a dual-track, in-memory
> retrieval and ranking pipeline. It searches 50,000 products in at most ten
> turns and scores 0.893583 against the organiser's 0.1067 baseline.

### Scene 2, official end-to-end run, 0:15 to 1:58

Run this only after the holdout gate is cleared. Keep the complete run on
screen. The `time` wrapper shows the wall clock without changing the evaluator.

```bash
cd techjam-conversational-search
/usr/bin/time -p python3 -m evaluator.local_evaluator \
  --catalog data/catalog.jsonl \
  --output /tmp/techjam-demo-official-results.json
```

Narration while it runs:

> This is the organiser's evaluator, running all 200 conversations against the
> read-only catalog. It is the exact submitted path, not a replacement metric.
>
> The problem statement asks for dual-track routing followed by in-memory
> retrieval and semantic ranking. Our router re-evaluates intent every turn
> using the current wording, accumulated constraints, and previous belief
> entropy. Browsing opens an 800-product route for coverage. Buying narrows it
> to 200 for precision.
>
> Retrieval represents title, features, category, description, store, and
> details as sparse TF-IDF vectors in memory, then scores cosine similarity. We
> also implemented a field-aware route and route fusion. It raised recall but
> lowered MRR in every tested mixture, so the shipped configuration keeps the
> stronger pooled route while preserving field fusion behind a config switch.
>
> Ranking rescales the top 200 using retrieval score, popularity, rating,
> recency-weighted phrase overlap, and a small offline LLM appeal prior. Nothing
> is hard filtered. An optional live LLM can read the conversation and reorder
> the top 20, but it defaults off so a missing network cannot break the graded
> path. We tuned on 160 sessions and reserved 40 for one final check.

When the result appears, point to the score, metrics, token usage, and time:

> In under two minutes, the agent reaches 97.5 percent Hit Rate at ten, 0.779609
> MRR, 2.39 turns to conversion, and a 0.893583 TechnicalScore with zero tokens.

Return to the repository root:

```bash
cd ..
```

### Scene 3, one real conversation, 1:58 to 2:28

Run this prepared trace view:

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

Narration:

> This real session starts with a vague request for a dress. The browsing route
> searches 800 products, finds an overloaded pool, and asks about features. The
> answer adds imported and wrap closure. The system switches to the 200-product
> buying route, clears the overload, and moves the target from outside the top
> ten to rank one. Ground truth is joined only for this offline explanation.

### Scene 4, measured design choice, 2:28 to 2:45

```bash
sed -n '/^### 4\.4 /,/^### 4\.5 /p' docs/consolidation.md | sed '$d'
```

Narration:

> The rerank-depth sweep selected 200. A depth of 100 loses targets already
> found by retrieval, while 400 and 800 dilute MRR with extra plausible
> products. We reran this sweep after phrase scoring changed the result.

### Scene 5, offline proof and disclosure, 2:45 to 3:08

```bash
python3 evaluation/verify_offline.py
```

Leave these final lines visible:

```text
OK 10 turns completed with every socket blocked
[offline] PASS. Safe to submit as running fully offline.
```

Narration:

> This probe blocks every socket and validates ten turns. At runtime, the Haiku
> 4.5 semantic artifact is only a local lookup. The optional claude-opus-5 live
> reranker fails safely to deterministic order. The shipped path therefore uses
> zero tokens and costs zero dollars. Our next priority is browsing MRR.

## 3. Publish

1. Keep the evaluator run uncut. Remove only pauses between scenes.
2. Confirm that all commands and metrics are readable at normal playback speed.
3. Export 1080p H.264 video with AAC audio and no unlicensed music.
4. Check the final video for secrets, notifications, personal paths, images, or
   logos.
5. Upload it to YouTube as **Public** and verify playback while signed out.
6. Replace `ADD_YOUTUBE_URL_AFTER_UPLOAD` with the public URL and give the same
   URL to the Track A owner for Devpost.
