# What to run on camera

Everything is terminal based. The rules accept that: for a backend solution, a
walkthrough of API usage, inference examples and result analysis is explicitly
allowed, and there is no UI in scope.

Five commands, in this order. Nothing needs preparing beforehand and nothing
writes to the repository.

---

## Before you press record

```bash
cd <repo>
pip install -r requirements.txt
clear
```

Font at 20px or larger, 1080p, and check the prompt does not show a real name or
a home directory you would rather not publish.

**Time one full evaluator run first.** It takes about 25 seconds on an Apple
silicon laptop and over 7 minutes on a loaded cloud box. Cut your scene lengths
to what your own machine actually does.

---

## 1. The product, working. About 60 seconds.

This is the one to lead with. It is a real conversation, not a metrics dump.

```bash
python3 demo/show.py --session public_0012
```

A vague opener, the agent asking rather than guessing, and the target arriving at
rank one on turn two. The index build at the top takes about 25 seconds and is
worth showing once: 50,000 products, in memory, no database and no network.

Say while it builds: the customer is simulated by the organiser's own code,
imported rather than reimplemented, so this is the same conversation the
official run scores.

## 2. The hard case. About 45 seconds.

```bash
python3 demo/show.py --scenario intent_override
```

The customer changes their mind mid-session. Watch turn 2: the target is already
in the top ten and the evaluator does not count it, because the pivot has not
happened yet. Then the customer says "actually, ignore my earlier preference",
and the agent recovers to rank 2 rather than collapsing.

Worth saying: we demote the superseded preference instead of erasing it. The
brief says erasure. We measured erasure and it scores worse, because in 30 of 30
of these sessions the preference the customer abandons is still true of the
target product.

## 3. Type at it yourself. About 30 seconds, optional.

```bash
python3 demo/show.py --chat
```

Then, one line at a time:

```
I need a leather belt
with a buckle closure, brown
full grain, handmade
```

The list converges onto a handmade full grain belt by the third line, and the
session state prints under each answer so the accumulation is visible.

Skip this scene if you are tight on time. Keep it if you want to show the system
responding to something that is obviously not scripted.

## 4. The number. About 30 seconds, or however long your machine takes.

```bash
cd techjam-conversational-search
python3 -m evaluator.local_evaluator --catalog data/catalog.jsonl
cd ..
```

The organiser's evaluator, unmodified, all 200 sessions. Leave it uncut.

**0.893583** against the starter baseline's **0.1067**. Hit Rate@10 0.975, MRR
0.7796, MTTC 2.39 turns against the baseline's 9.81.

## 5. Two claims a judge would otherwise have to take on trust. About 30 seconds.

```bash
python3 evaluation/verify_offline.py
```

Ten turns with every socket blocked. The rules say scoring may run with
networking disabled, so we tested it rather than assuming it, and the probe is
negative controlled: inject a real network call and it goes red.

Then the honest close, no command needed:

> Forty of the 200 sessions were reserved on day one and no tuning decision ever
> saw them. We spent them once, at the end: 0.8801 held out against 0.8969 on
> the sessions we tuned on. Hit Rate is identical on both, so nothing about
> retrieval was fitted to the training set.
>
> And the limit we know about. Browsing MRR is 0.675 against buying's 0.828.
> When the target does not come first, it is equal or better on phrase evidence
> and on retrieval similarity, and loses on popularity alone. The phrase signal
> has run out of resolution. Breaking that tie needs a model reading the
> conversation, and that is where we go next.

---

## Numbers to say, and nothing else

| Metric | Value |
| --- | --- |
| TechnicalScore | **0.893583** |
| Hit Rate@10 | 0.975 |
| MRR | 0.779609 |
| MTTC | 2.39 turns |
| Efficiency | 0.861 |
| Held out (40 unseen sessions) | 0.8801 |
| Baseline TechnicalScore | 0.1067 |
| Baseline MTTC | 9.81 |
| Reported token usage | 0 |
| Cost per evaluation | $0.00 |

The score is machine independent, verified on three Python and numpy
combinations. Wall clock is not: quote it with the hardware beside it.

## Before uploading

1. Under 3 minutes. Hard cap in the rules.
2. No API keys, notifications, personal paths, product images or brand logos.
3. The score spoken on camera matches the README and the Devpost description.
4. YouTube, Public, and check playback while signed out.
5. Put the URL in `docs/demo-script.md` and give it to whoever submits Devpost.
