"""How good is the Buying versus Browsing router? Scored against the labels.

    python evaluation/intent_audit.py
    python evaluation/intent_audit.py --ablate

The agent never sees `scenario_type`. This file does, once, after the fact, which
is the only way to put a number on a routing module rather than assert that it
exists. Pillar I calls for "highly sensitive intent-detection modules splitting
traffic into Buying and Browsing tracks", and sensitivity is a measurement.

It replays every public session's dialogue through the same simulator logic the
evaluator uses, so the utterances the router is scored on are exactly the ones it
will see, then compares the route it chose against the session's true scenario.

**Turn one is the only turn where the label is the answer.** `scenario_type` is a
property of the session, not of the turn, and spec 5.1 asks the router to move: a
customer who opens Browsing and states two constraints by turn three should be
routed Buying by turn three. Scoring every turn against the session label would
therefore mark the intended behaviour wrong. So turn one is scored as a
classification, and what happens afterwards is reported as a trajectory rather
than an error rate.

**The ablation** re-scores turn one with the opener patterns removed. Those three
patterns match the simulator's exact sentence forms and are the part of this
router fitted to this harness. A reader is entitled to know how much of the
accuracy survives without them, so both numbers are printed and both belong in
the README.

Replays the dialogue only. It never runs retrieval, so it costs seconds rather
than minutes.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
KIT = REPO_ROOT / "techjam-conversational-search"
for _entry in (str(REPO_ROOT), str(KIT)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from evaluator.local_evaluator import (  # noqa: E402
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    materialize_hidden_fields,
)

from evaluation.run_eval import DEFAULT_CATALOG  # noqa: E402
from evaluation.splits import DEFAULT_DATASET, load_sessions  # noqa: E402
from src.config import Config  # noqa: E402
from src.policy.intent import _OPENERS, BROWSING, BUYING, route  # noqa: E402
from src.state import Slots  # noqa: E402

# The track each scenario should open on. `buying` and `browsing` are named by
# the brief. `intent_override` opens by stating a preference, which is Buying
# behaviour even though the preference is the one about to be abandoned.
#
# `boundary` is deliberately absent. Those sessions are labelled by how the
# customer answers questions, not by how they open, so there is no correct
# opening track to score them against, and folding them in would move a number
# that means something into one that does not.
EXPECTED = {"buying": BUYING, "browsing": BROWSING, "intent_override": BUYING}

# The ask the replay issues each turn. The router does not read it, but the
# simulator needs one to decide what to disclose, and the yield order is what
# the shipped policy actually asks.
REPLAY_ASKS = ("feature", "material", "color", "style", "size", "use_case", "other")


def replay(sample: dict[str, Any], products: dict[str, dict], categories: dict[str, list[str]],
           config: Config, turns: int, openers: tuple = _OPENERS) -> list[tuple[int, str, str]]:
    """Reproduce one session's customer utterances and route each of them.

    Returns (turn, utterance, chosen track). Belief entropy is unavailable here
    because nothing is retrieved, so the router runs without it, which measures
    the wording and constraint signals in isolation.
    """
    card, behavior = materialize_hidden_fields(sample, products)
    effective = {**sample, "intent_card": card, "behavior": behavior}
    target = str(sample["ground_truth"]["parent_asin"])
    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"

    message = initial_message(effective, coarse_category(categories.get(target, [])), disclosed)
    slots = Slots(config=config)
    out: list[tuple[int, str, str]] = []
    for turn in range(1, turns + 1):
        slots.observe(message, turn)
        track = route(
            message=message,
            constraint_count=len(slots.constraints()),
            previous_entropy=None,
            config=config,
            openers=openers,
        )
        out.append((turn, message, track.name))

        override = effective.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            message = str(override.get("message", ""))
        else:
            ask = REPLAY_ASKS[min(turn - 1, len(REPLAY_ASKS) - 1)]
            message, boundary_used = customer_reply(effective, ask, disclosed, boundary_used)
    return out


def score(samples: list[dict[str, Any]], products: dict[str, dict],
          categories: dict[str, list[str]], config: Config, turns: int,
          openers: tuple = _OPENERS) -> dict[str, Any]:
    """Turn one accuracy, plus the trajectory every turn after it."""
    opening = Counter()
    per_scenario: dict[str, Counter] = {}
    buying_share: dict[str, dict[int, Counter]] = {}
    switches: list[int] = []

    for sample in samples:
        scenario = str(sample["scenario_type"])
        routed = replay(sample, products, categories, config, turns, openers)
        expected = EXPECTED.get(scenario)
        if expected is not None:
            verdict = "correct" if routed[0][2] == expected else "wrong"
            opening[verdict] += 1
            per_scenario.setdefault(scenario, Counter())[verdict] += 1
        for index, (turn, _, track) in enumerate(routed):
            buying_share.setdefault(scenario, {}).setdefault(turn, Counter())[track] += 1
            if index and track != routed[index - 1][2]:
                switches.append(turn)

    def accuracy(counter: Counter) -> float:
        total = counter["correct"] + counter["wrong"]
        return counter["correct"] / total if total else 0.0

    def share(counter: Counter) -> float:
        total = sum(counter.values())
        return counter[BUYING] / total if total else 0.0

    return {
        "turn_one_accuracy": round(accuracy(opening), 4),
        "turn_one_scored_sessions": opening["correct"] + opening["wrong"],
        "per_scenario_turn_one": {
            name: round(accuracy(counter), 4) for name, counter in sorted(per_scenario.items())
        },
        "buying_share_by_turn": {
            scenario: {turn: round(share(counter), 4) for turn, counter in sorted(turns_.items())}
            for scenario, turns_ in sorted(buying_share.items())
        },
        "track_switches": len(switches),
        "switch_turn": dict(sorted(Counter(switches).items())),
    }


# Each ablation removes one signal so its contribution is visible rather than
# asserted. `no openers` is the one that matters to a reader: those three
# patterns match the simulator's exact sentence forms, and everything left after
# removing them is ordinary shopping English that a real deployment would run on.
ABLATIONS: dict[str, tuple[dict[str, Any], tuple]] = {
    "full router": ({}, _OPENERS),
    "no openers": ({}, ()),
    "no wording cues": ({"intent_cue_weight": 0.0}, _OPENERS),
    "no constraint count": ({"intent_constraint_weight": 0.0}, _OPENERS),
    "no openers, no cues": ({"intent_cue_weight": 0.0}, ()),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument("--ablate", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/intent_audit.json"))
    args = parser.parse_args()

    samples = load_sessions(args.dataset)
    _, categories, products = catalog_index(args.catalog)
    base = Config()
    report = score(samples, products, categories, base, args.turns)

    print(f"\n{'=' * 70}")
    print(f"  Intent router audit   n={len(samples)} sessions, {args.turns} turns each")
    print("=" * 70)
    print(f"  turn one track matches the session label   "
          f"{report['turn_one_accuracy']:.3f}"
          f"   over {report['turn_one_scored_sessions']} scored sessions")
    print(f"  track switches after turn one              "
          f"{report['track_switches']}  at turns {report['switch_turn']}")
    print("\n  turn one accuracy by scenario")
    print(f"  {'-' * 34}")
    for scenario, value in report["per_scenario_turn_one"].items():
        print(f"  {scenario:<22}{value:>10.3f}")

    print("\n  share of turns routed Buying, by turn")
    turns_seen = sorted({t for row in report["buying_share_by_turn"].values() for t in row})
    header = "".join(f"{t:>7}" for t in turns_seen)
    print(f"  {'scenario':<18}{header}")
    print(f"  {'-' * (18 + 7 * len(turns_seen))}")
    for scenario, row in report["buying_share_by_turn"].items():
        cells = "".join(f"{row.get(t, 0.0):>7.2f}" for t in turns_seen)
        print(f"  {scenario:<18}{cells}")
    print("\n  Browsing sessions climbing towards Buying is the router working,")
    print("  not the router failing: spec 5.1 asks it to follow the customer.")

    if args.ablate:
        report["ablations"] = {}
        print(f"\n  {'ablation':<24}{'turn 1':>10}")
        print(f"  {'-' * 34}")
        for label, (overrides, openers) in ABLATIONS.items():
            variant = replace(base, **overrides) if overrides else base
            result = score(samples, products, categories, variant, args.turns, openers)
            report["ablations"][label] = result
            print(f"  {label:<24}{result['turn_one_accuracy']:>10.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n[intent_audit] -> {args.output}")


if __name__ == "__main__":
    main()
