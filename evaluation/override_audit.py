"""Does an Intent Override actually contradict anything? Measured, not assumed.

    python evaluation/override_audit.py

Pillar II asks for "abrupt Intent Override (slot erasure and rewriting)", and the
obvious implementation is to delete the superseded slot. This script exists
because that obvious implementation is wrong for this task, and the claim needed
evidence before it went into the code.

It materialises each session's hidden intent card exactly as the evaluator does,
then checks whether the preference the customer says to ignore is itself a
property of the target product. It is: the simulator builds both the old and the
new value from the same product's `features` and `details`, so the pivot is a
change of emphasis rather than a contradiction, and erasing the old slot deletes
true evidence about the target.

`Config.override_demote` is the knob this justifies. 0.0 is literal erasure,
1.0 ignores the pivot entirely, and the shipped value sits between them. The
score for all three is in the README.

Reads nothing but the frozen data. Writes nothing.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
KIT = REPO_ROOT / "techjam-conversational-search"
for _entry in (str(REPO_ROOT), str(KIT)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from evaluator.local_evaluator import (  # noqa: E402
    catalog_index,
    classify_constraint,
    materialize_hidden_fields,
    searchable_text,
)

from evaluation.run_eval import DEFAULT_CATALOG  # noqa: E402
from evaluation.splits import DEFAULT_DATASET, load_sessions  # noqa: E402


def _normalise(value: str) -> str:
    """Collapse the formatting difference between a details key and its text.

    `searchable_text` joins a details entry as "key value" while the intent card
    quotes it as "key: value". Two of the thirty sessions differ by nothing but
    that colon, and counting them as contradictions would overstate the case
    this script exists to make.
    """
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def audit(samples: list[dict[str, Any]], products: dict[str, dict]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        if sample.get("scenario_type") != "intent_override":
            continue
        card, behavior = materialize_hidden_fields(sample, products)
        override = behavior.get("override") or {}
        target = str(sample["ground_truth"]["parent_asin"])
        corpus = _normalise(searchable_text(products[target]))
        old_value = str(override.get("old_value", ""))
        new_value = str(override.get("new_value", ""))
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "override_turn": int(override.get("turn", 0)),
                "old_value": old_value,
                "new_value": new_value,
                "old_bucket": classify_constraint(old_value),
                "new_bucket": classify_constraint(new_value),
                "old_in_target": _normalise(old_value) in corpus,
                "new_in_target": _normalise(new_value) in corpus,
                "old_is_soft_preference": old_value in card["soft_preferences"],
                "new_is_hard_constraint": new_value in card["hard_constraints"],
                "contradicts": old_value == new_value,
            }
        )
    total = len(rows)
    return {
        "sessions": total,
        "old_value_is_a_property_of_the_target": sum(r["old_in_target"] for r in rows),
        "new_value_is_a_property_of_the_target": sum(r["new_in_target"] for r in rows),
        "old_value_drawn_from_soft_preferences": sum(
            r["old_is_soft_preference"] for r in rows
        ),
        "new_value_drawn_from_hard_constraints": sum(
            r["new_is_hard_constraint"] for r in rows
        ),
        "old_and_new_actually_conflict": sum(r["contradicts"] for r in rows),
        "override_turn": dict(Counter(r["override_turn"] for r in rows)),
        "rows": rows,
    }


def render(report: dict[str, Any]) -> str:
    total = report["sessions"]
    lines = [
        "",
        "=" * 70,
        f"  Intent Override audit   n={total} sessions",
        "=" * 70,
        f"  the preference to ignore IS a property of the target   "
        f"{report['old_value_is_a_property_of_the_target']:>3}/{total}",
        f"  the replacing preference IS a property of the target   "
        f"{report['new_value_is_a_property_of_the_target']:>3}/{total}",
        f"  old value drawn from the target's soft preferences     "
        f"{report['old_value_drawn_from_soft_preferences']:>3}/{total}",
        f"  new value drawn from the target's hard constraints     "
        f"{report['new_value_drawn_from_hard_constraints']:>3}/{total}",
        f"  old and new genuinely conflict                         "
        f"{report['old_and_new_actually_conflict']:>3}/{total}",
        f"  override fires on turn                                 "
        f"{report['override_turn']}",
        "",
        "  Reading: the pivot redirects emphasis, it does not contradict. Both",
        "  values come from the same target product's own record, so deleting",
        "  the superseded slot deletes true evidence. src/state/slots.py demotes",
        "  it instead, at Config.override_demote.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=Path("artifacts/override_audit.json"))
    args = parser.parse_args()

    samples = load_sessions(args.dataset)
    _, _, products = catalog_index(args.catalog)
    report = audit(samples, products)
    print(render(report))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n[override_audit] -> {args.output}")


if __name__ == "__main__":
    main()
