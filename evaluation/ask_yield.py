"""Which ask_attribute values can the simulated customer actually answer?

The ask policy is only as good as its model of what the customer will respond
to. This measures that directly instead of assuming it, which matters because
the first version of the policy assumed wrong and spent its two most valuable
early turns on questions with no possible answer.

The customer answers an attribute only when one of the target's own constraint
phrases falls into that bucket. So: derive each target's constraints exactly as
the evaluator does, classify them the same way, and count.

    python evaluation/ask_yield.py
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
KIT = REPO_ROOT / "techjam-conversational-search"
for _entry in (str(REPO_ROOT), str(KIT)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from evaluator.local_evaluator import classify_constraint, intent_card  # noqa: E402

# Every value the contract allows, so the audit can report which of them never
# appear rather than silently omitting them.
ALL_ATTRIBUTES = (
    "category", "material", "color", "size", "style",
    "brand", "budget", "feature", "use_case",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=KIT / "data/catalog.jsonl")
    parser.add_argument("--dataset", type=Path, default=KIT / "data/public_set.jsonl")
    args = parser.parse_args()

    by_asin = {}
    with args.catalog.open(encoding="utf-8") as handle:
        for line in handle:
            product = json.loads(line)
            by_asin[str(product["parent_asin"])] = product
    samples = [json.loads(l) for l in args.dataset.open(encoding="utf-8") if l.strip()]

    counts: collections.Counter[str] = collections.Counter()
    sessions_answerable: collections.Counter[str] = collections.Counter()
    for sample in samples:
        card = intent_card(by_asin[str(sample["ground_truth"]["parent_asin"])])
        constraints = card["hard_constraints"] + card["soft_preferences"]
        buckets = {classify_constraint(c) for c in constraints}
        counts.update(classify_constraint(c) for c in constraints)
        sessions_answerable.update(buckets)

    total = sum(counts.values()) or 1
    print(f"across {len(samples)} sessions\n")
    print(f"  {'attribute':<12}{'constraints':>13}{'share':>8}{'answerable in':>15}")
    print("  " + "-" * 48)
    for name in sorted(ALL_ATTRIBUTES, key=lambda a: -counts[a]):
        answerable = sessions_answerable[name]
        print(
            f"  {name:<12}{counts[name]:>13}{counts[name] / total:>8.1%}"
            f"{answerable / len(samples):>14.1%}"
        )

    never = [a for a in ALL_ATTRIBUTES if not counts[a]]
    if never:
        print(f"\n  Never answerable, so always a wasted ask: {', '.join(never)}")
    print("\n  src/state/slots.py orders its asks by this table.")


if __name__ == "__main__":
    main()
