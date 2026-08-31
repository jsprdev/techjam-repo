"""Recall ceiling: how deep must retrieval go before it contains the target?

This is the hard cap on HitRate. The dialogue layer can only reorder what
retrieval already found, so measure this before tuning any policy.

Two queries are scored per session. The turn 1 query is what the agent knows
from the opening message alone. The full query adds every constraint the
simulated customer would ever disclose, which is the best case the dialogue
could ever reach. The gap between them is what the dialogue layer is worth.

Usage:
    python evaluation/recall_ceiling.py
"""
import json, random, sys
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "techjam-conversational-search"))
from evaluator.local_evaluator import intent_card, coarse_category, searchable_text

KIT = Path(__file__).resolve().parent.parent / "techjam-conversational-search"
rows = [json.loads(l) for l in (KIT / "data/catalog.jsonl").open(encoding="utf-8")]
asins = [str(r["parent_asin"]) for r in rows]
pos = {a: i for i, a in enumerate(asins)}
samples = [json.loads(l) for l in (KIT / "data/public_set.jsonl").open(encoding="utf-8")]

print("indexing 50k products ...")
corpus = [searchable_text(r) for r in rows]
vec = TfidfVectorizer(lowercase=True, sublinear_tf=True, min_df=2, max_features=300_000,
                      ngram_range=(1, 2), strip_accents="unicode")
X = vec.fit_transform(corpus)
print(f"  matrix {X.shape}, nnz {X.nnz/1e6:.1f}M")

def ranks_for(queries: list[str], targets: list[str]) -> np.ndarray:
    Q = vec.transform(queries)
    out = []
    # chunk to bound memory
    for start in range(0, Q.shape[0], 25):
        S = (Q[start:start+25] @ X.T).toarray()
        for j in range(S.shape[0]):
            t = pos[targets[start + j]]
            # rank = how many products score strictly higher
            out.append(int((S[j] > S[j][t]).sum()) + 1)
    return np.array(out)

turn1, full, targets, scen = [], [], [], []
for s in samples:
    t = str(s["ground_truth"]["parent_asin"]); p = rows[pos[t]]
    card = intent_card(p)
    cat = coarse_category(p.get("categories") or [])
    opening = cat
    if s["scenario_type"] == "buying" and card["hard_constraints"]:
        opening += " " + str(card["hard_constraints"][0])
    everything = " ".join([cat, *card["hard_constraints"], *card["soft_preferences"]])
    turn1.append(opening); full.append(everything); targets.append(t); scen.append(s["scenario_type"])

for label, qs in (("turn 1 opening only", turn1), ("all constraints revealed", full)):
    r = ranks_for(qs, targets)
    print(f"\n=== {label} ===")
    for k in (1, 10, 100, 1000):
        print(f"  recall@{k:<5d} {(r <= k).mean():6.1%}")
    print(f"  median rank {int(np.median(r))}")
    if label.startswith("all"):
        import collections
        for name in sorted(set(scen)):
            m = np.array([x == name for x in scen])
            print(f"    {name:16s} n={m.sum():3d}  recall@10 {(r[m] <= 10).mean():6.1%}  recall@1000 {(r[m] <= 1000).mean():6.1%}")
