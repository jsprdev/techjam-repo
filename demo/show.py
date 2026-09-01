"""Live demo driver. One process, one index build, several scenes.

    python3 demo/show.py                    # replay a labelled session, then wait
    python3 demo/show.py --list             # sessions worth showing, by scenario
    python3 demo/show.py --session public_0012
    python3 demo/show.py --chat             # type your own messages at the agent

Built for recording. The catalog index takes about 25 seconds to build and that
cost is paid once at startup, so every scene after it is instant and you are
never narrating over a progress spinner.

The customer's replies come from the organiser's own simulator, imported from
`evaluator.local_evaluator` rather than reimplemented, so a replayed session is
the same conversation the official run scores and not a scripted mock. The turn
loop mirrors `evaluate()` including the intent override branch. Ground truth is
used only to print the target's rank, after the agent has answered, exactly as
the offline diagnostics do. The agent never sees it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
KIT = REPO_ROOT / "techjam-conversational-search"
for path in (REPO_ROOT, KIT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluator.local_evaluator import (  # noqa: E402
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    materialize_hidden_fields,
    normalize_recommendations,
)

from src.agent import Agent  # noqa: E402
from src.config import Config  # noqa: E402

CATALOG = KIT / "data" / "catalog.jsonl"
PUBLIC_SET = KIT / "data" / "public_set.jsonl"

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
OFF = "\033[0m"


def rule(title: str = "") -> None:
    print(f"\n{DIM}{'=' * 74}{OFF}")
    if title:
        print(f"  {BOLD}{title}{OFF}")
        print(f"{DIM}{'=' * 74}{OFF}")


def load_samples() -> dict[str, dict]:
    with PUBLIC_SET.open(encoding="utf-8") as handle:
        return {row["sample_id"]: row for row in map(json.loads, handle)}


def product_line(products: dict[str, dict], asin: str, rank: int, is_target: bool) -> str:
    title = str(products.get(asin, {}).get("title", "")).strip()
    if len(title) > 58:
        title = title[:55] + "..."
    mark = f"{GREEN}<- target{OFF}" if is_target else ""
    colour = GREEN if is_target else ""
    end = OFF if is_target else ""
    return f"   {colour}{rank:>2}. {asin}  {title:<58}{end} {mark}"


def replay(agent: Agent, sample: dict, categories, products, catalog_ids, top_n: int) -> None:
    """Drive one labelled session exactly as the official evaluator does."""
    session_id = f"demo_{uuid.uuid4().hex[:8]}"
    agent.reset(session_id, sample["user_profile"])
    target = str(sample["ground_truth"]["parent_asin"])
    card, behavior = materialize_hidden_fields(sample, products)
    effective = {**sample, "intent_card": card, "behavior": behavior}

    rule(f"{sample['sample_id']}   scenario: {sample['scenario_type']}   difficulty: {sample.get('difficulty_bucket', '?')}")
    print(f"  {DIM}Hidden target, known to this script only: {target}{OFF}")
    print(f"  {DIM}{str(products.get(target, {}).get('title', ''))[:70]}{OFF}")

    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    user_message = initial_message(effective, coarse_category(categories.get(target, [])), disclosed)

    for turn in range(1, MAX_TURNS + 1):
        print(f"\n{BOLD}  TURN {turn}{OFF}")
        print(f"  {CYAN}customer:{OFF} {user_message}")

        started = time.perf_counter()
        response = agent.respond(session_id, user_message, turn, TOP_K)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
        rank = ranked.index(target) + 1 if target in ranked else None

        print(f"  {YELLOW}agent:{OFF}    {response.get('message', '')}")
        asks = response.get("ask_attribute")
        print(
            f"  {DIM}asks_about={asks}  returned={len(ranked)}  "
            f"target_rank={rank if rank else 'not in list'}  {elapsed_ms:.0f} ms{OFF}"
        )
        for index, asin in enumerate(ranked[:top_n], 1):
            print(product_line(products, asin, index, asin == target))

        if override_applied and target in ranked:
            print(f"\n  {GREEN}{BOLD}CONVERTED on turn {turn} at rank {rank}.{OFF}")
            print(f"  {DIM}Reciprocal rank {1.0 / rank:.3f}. The official run stops the session here.{OFF}")
            return
        if target in ranked:
            # Worth saying out loud: on an intent_override session the evaluator
            # does not count a hit until the customer has actually pivoted, so a
            # target sitting in the top ten here scores nothing. Without this
            # line the next frame looks like the agent found it and carried on.
            print(f"  {DIM}The target is in the list, but this is an intent_override session and{OFF}")
            print(f"  {DIM}the customer has not pivoted yet, so the evaluator does not count it.{OFF}")
        if turn == MAX_TURNS:
            break

        override = effective.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            user_message = str(override.get("message", "Actually, please ignore my earlier preference."))
            print(f"\n  {DIM}The customer is about to change their mind. This is the Intent Override case.{OFF}")
        else:
            user_message, boundary_used = customer_reply(
                effective, response.get("ask_attribute"), disclosed, boundary_used
            )

    print(f"\n  Session ended after {MAX_TURNS} turns without the target reaching the top ten.")


SUGGESTED = (
    "I need a leather belt",
    "with a buckle closure, brown",
    "full grain, handmade",
)


def chat(agent: Agent, products, catalog_ids, top_n: int) -> None:
    """Free-form conversation. No ground truth, no simulator, just the agent.

    Prints the accumulated constraints each turn. Without them the state
    machine is invisible: on a vague opener the popularity prior holds the top
    of the list steady for a turn or two, and a viewer cannot tell whether the
    agent heard the second message or ignored it. The constraint line shows the
    session state growing even when the visible ordering has not moved yet.
    """
    rule("Interactive. Type what a shopper would say. Blank line or Ctrl-D to stop.")
    print(f"  {DIM}Specific wording converges fastest, because the customer\'s own words are{OFF}")
    print(f"  {DIM}the evidence. Try, one line at a time:{OFF}")
    for line in SUGGESTED:
        print(f"    {DIM}{line}{OFF}")

    session_id = f"chat_{uuid.uuid4().hex[:8]}"
    agent.reset(session_id, {"preference_tags": []})
    for turn in range(1, MAX_TURNS + 1):
        try:
            message = input(f"\n{CYAN}  you:{OFF} ").strip()
        except EOFError:
            print()
            return
        if not message:
            return
        started = time.perf_counter()
        response = agent.respond(session_id, message, turn, TOP_K)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
        print(f"  {YELLOW}agent:{OFF} {response.get('message', '')}")
        constraints = agent.constraints_for(session_id)
        if constraints:
            print(f"  {DIM}session state: {' | '.join(constraints)}{OFF}")
        print(f"  {DIM}asks_about={response.get('ask_attribute')}  {elapsed_ms:.0f} ms{OFF}")
        for index, asin in enumerate(ranked[:top_n], 1):
            print(product_line(products, asin, index, False))
    print(f"\n  {DIM}Ten turn cap reached. The rules score a session zero past this point,{OFF}")
    print(f"  {DIM}so the agent enforces the cap itself rather than trusting the harness.{OFF}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", default="public_0012", help="sample_id to replay")
    parser.add_argument("--scenario", help="replay the first session of this scenario instead")
    parser.add_argument("--chat", action="store_true", help="free-form conversation, no ground truth")
    parser.add_argument("--list", action="store_true", help="list sessions by scenario and exit")
    parser.add_argument("--top", type=int, default=5, help="products to print per turn")
    args = parser.parse_args()

    samples = load_samples()

    if args.list:
        by_scenario: dict[str, list[str]] = {}
        for sample in samples.values():
            by_scenario.setdefault(sample["scenario_type"], []).append(sample["sample_id"])
        rule("Sessions available")
        for scenario in sorted(by_scenario):
            ids = by_scenario[scenario]
            print(f"  {BOLD}{scenario:<18}{OFF} {len(ids):>3} sessions   e.g. {', '.join(ids[:4])}")
        print(f"\n  {DIM}python3 demo/show.py --session <id>     python3 demo/show.py --scenario intent_override{OFF}")
        return

    rule("Conversational Shopping Agent")
    print("  Building the in-memory index over 50,000 products. No database, no network.")
    started = time.perf_counter()
    agent = Agent(CATALOG, Config())
    catalog_ids, categories, products = catalog_index(CATALOG)
    print(f"  Ready in {time.perf_counter() - started:.1f}s.")

    if args.chat:
        chat(agent, products, catalog_ids, args.top)
        return

    sample_id = args.session
    if args.scenario:
        match = next((s for s in samples.values() if s["scenario_type"] == args.scenario), None)
        if match is None:
            raise SystemExit(f"no session with scenario {args.scenario!r}. Try --list.")
        sample_id = match["sample_id"]
    if sample_id not in samples:
        raise SystemExit(f"unknown session {sample_id!r}. Try --list.")

    replay(agent, samples[sample_id], categories, products, catalog_ids, args.top)


if __name__ == "__main__":
    main()
