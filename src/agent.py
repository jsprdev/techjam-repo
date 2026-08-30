"""The entry class the official evaluator imports. OWNED BY ROLE 3.

Thin orchestration only. It sequences the four modules and owns the turn
counter, nothing else. Logic belongs in the modules.

Two guarantees this file must never lose, both worth a whole session each:

1. It never raises. The evaluator catches exceptions and substitutes an empty
   response, so a crash silently costs every remaining turn of that session.
   The broad except below is deliberate, not laziness.
2. It never exceeds ten turns. The rules make exceeding the cap a forced
   termination and a zero, and the instruction is to enforce it ourselves rather
   than trust the harness.

It also always returns recommendations. There is no penalty for guessing
alongside a question, the evaluator checks recommendations for a hit before it
reads `ask_attribute`, and a session can only end favourably by surfacing the
target. Never return an empty list.
"""

from __future__ import annotations

import time
from pathlib import Path

from src import catalog as catalog_module
from src import response as response_module
from src.config import Config
from src.interfaces import MAX_TURNS, TOP_K
from src.rank import PriorRanker
from src.retrieval import TfidfRetriever
from src.state import Slots
from src.trace import SINK, TurnTrace


class Agent:
    """Conversational shopping agent, deterministic and fully offline."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        config: Config | None = None,
    ) -> None:
        # Signature matches the starter agent, because the evaluator constructs
        # us as Agent(args.catalog) and nothing else.
        self.config = config or Config()
        self.catalog = catalog_module.load(catalog_path)
        self.retriever = TfidfRetriever(self.catalog, self.config)
        self.ranker = PriorRanker(self.catalog, self.config)
        self._sessions: dict[str, Slots] = {}

    def set_config(self, config: Config) -> None:
        """Adopt new tuning parameters without rebuilding the index.

        The sweep harness runs many variants against one built index, and
        reaching into `agent.ranker.config` from outside is fragile: the moment
        a module starts caching a value from config, or a new component appears,
        an external caller silently tunes nothing and the sweep reports a wrong
        number. Propagation belongs here, next to the components.

        Only safe for fields that do not change the index. Index-affecting
        fields are listed in `evaluation/sweep.py:INDEX_FIELDS` and force a
        rebuild instead.
        """
        self.config = config
        self.retriever.config = config
        self.ranker.config = config
        # Live sessions keep the config they were reset with, which is correct:
        # changing tuning mid-session would make a run uninterpretable.

    def reset(self, session_id: str, user_profile: dict) -> None:
        """Start a fresh session. MUST NEVER RAISE.

        The evaluator calls this outside any try/except, unlike respond(), so a
        single exception here aborts the whole 200 session run rather than
        costing one turn. Everything below is defensive for that reason.

        Note `average_prior_rating` is typed ["number", "null"] in the contract.
        It is never null across the 200 public sessions, but the private 800 may
        send null, so nothing here or downstream may assume a number.
        """
        try:
            profile = dict(user_profile) if isinstance(user_profile, dict) else {}
        except Exception:  # noqa: BLE001 - a raise here costs the entire run
            profile = {}
        slots = Slots(config=self.config)
        slots.profile = profile
        # Rebinding rather than mutating guarantees no state survives a reset,
        # since one Agent instance serves every session sequentially.
        self._sessions[str(session_id)] = slots

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int = TOP_K,
    ) -> dict:
        started = time.perf_counter()
        try:
            return self._respond(session_id, user_message, turn, top_k, started)
        except Exception:  # noqa: BLE001 - a crash costs the whole session
            return response_module.build(
                message="Let me keep looking. Could you tell me more about what you need?",
                ask_attribute="other",
                recommendations=self._fallback(top_k),
            )

    # -- internals -----------------------------------------------------------

    def _respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
        started: float,
    ) -> dict:
        # Enforce the cap ourselves rather than trusting the harness.
        if turn > MAX_TURNS:
            return response_module.build(
                message="Here are my best matches.",
                ask_attribute=None,
                recommendations=self._fallback(top_k),
            )

        slots = self._sessions.get(session_id)
        if slots is None:
            # reset was skipped. Recover rather than raise, unlike the starter.
            self.reset(session_id, {})
            slots = self._sessions[session_id]

        slots.observe(user_message, turn)
        query = slots.to_query()
        width = self._truncation_width(slots)
        candidates = self.retriever.retrieve(query, width)
        ranked = self.ranker.rank(candidates, slots, slots.profile)
        if not ranked:
            ranked = self._fallback(top_k)

        attribute = slots.pick_attribute()
        message = self._phrase(attribute)

        SINK.record(
            TurnTrace(
                session_id=session_id,
                turn=turn,
                user_message=user_message,
                query=query,
                ask_attribute=attribute,
                candidate_count=len(candidates),
                top_recommendations=ranked[:TOP_K],
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )
        )
        return response_module.build(
            message=message,
            ask_attribute=attribute,
            recommendations=ranked[: max(top_k, TOP_K)],
        )

    def _truncation_width(self, slots: Slots) -> int:
        """Dynamic truncation. ROLE 2 owns the routing that decides this.

        Placeholder heuristic: once a couple of constraints are in hand the
        session looks like Buying, so narrow. Before that keep the pool wide so
        open ended Browsing can still reach across categories.
        """
        if len(slots.constraints()) >= 2:
            return self.config.truncate_buying
        return self.config.truncate_browsing

    def _phrase(self, attribute: str | None) -> str:
        """PLACEHOLDER wording. ROLE 3 owns making this sound human."""
        if attribute is None:
            return "Here are the closest matches I found."
        readable = attribute.replace("_", " ")
        return f"Here are some options. Any preference on {readable}?"

    def _fallback(self, top_k: int) -> list[str]:
        """Most reviewed products, used when retrieval returns nothing.

        Popularity is a genuinely competitive baseline on leave-last-out Amazon
        splits, so this is a real guess rather than filler.
        """
        if not hasattr(self, "_popular"):
            ordered = sorted(
                self.catalog.products,
                key=lambda p: float(p.get("rating_number") or 0),
                reverse=True,
            )
            self._popular = [str(p["parent_asin"]) for p in ordered[:TOP_K]]
        return self._popular[: max(top_k, TOP_K)]
