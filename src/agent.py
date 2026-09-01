"""The entry class the official evaluator imports. OWNED BY ROLE 3.

Thin orchestration only. It sequences the modules and owns the turn counter,
nothing else. Logic belongs in the modules.

The per turn pipeline, and the pillar each stage answers:

    route     src/policy/intent.py    Buying or Browsing, chosen fresh    I, III
    retrieve  src/retrieval/          lexical candidates at track width   I
    believe   src/rank/               evidence blended into a belief      I, III
    decide    src/policy/commit.py    over-generality cutoff, commit      II
    ask       src/state/slots.py      which attribute buys the most       II
    phrase    src/language/phrase.py  the sentence, grounded              II

Nothing in that sequence is fixed at session start. The track, the truncation
width, the shortlist depth and the ask are all re-selected every turn from the
current state, which is the honest reading of Pillar III's Adaptive
Orchestration.

Two guarantees this file must never lose, both worth a whole session each:

1. It never raises. The evaluator catches exceptions and substitutes an empty
   response, so a crash silently costs every remaining turn of that session. The
   broad except below is deliberate, not laziness.
2. It never exceeds ten turns. The rules make exceeding the cap a forced
   termination and a zero, and the instruction is to enforce it ourselves rather
   than trust the harness.

It also always returns recommendations. There is no penalty for guessing
alongside a question, the evaluator checks recommendations for a hit before it
reads `ask_attribute`, and a session can only end favourably by surfacing the
target. Never return an empty list, and never shorten one to trade MRR against
MTTC.
"""

from __future__ import annotations

import time
from pathlib import Path

from src import catalog as catalog_module
from src import response as response_module
from src.config import Config
from src.interfaces import MAX_TURNS, TOP_K
from src.language import question as phrase_question
from src.policy.commit import decide
from src.policy.intent import route
from src.rank import PriorRanker
from src.retrieval import TfidfRetriever
from src.language import rerank as llm_rerank
from src.state import Slots
from src.state.session import SessionContext, TurnRecord
from src.trace import SINK, TurnTrace

# How many of the top candidates the question phrasing is grounded in. Wide
# enough that the options offered reflect the shortlist rather than one product,
# narrow enough that it does not describe items the customer will never see.
GROUNDING_WIDTH = 5


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
        self._sessions: dict[str, SessionContext] = {}

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
        self._sessions[str(session_id)] = SessionContext(
            session_id=str(session_id), slots=slots
        )

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
            if self.config.strict_errors:
                # Tests and the offline probe opt in to this. Without it, a
                # totally broken pipeline still returns a contract-legal list
                # and every assertion downstream passes on the fallback.
                raise
            # Record the fallback too. Traces that silently omit failed turns
            # make latency and failure rate look better than they are.
            SINK.record(
                TurnTrace(
                    session_id=session_id,
                    turn=turn,
                    user_message=user_message,
                    query="",
                    ask_attribute="other",
                    candidate_count=0,
                    top_recommendations=[],
                    elapsed_ms=(time.perf_counter() - started) * 1000.0,
                    extra={"fallback": True},
                )
            )
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

        session = self._sessions.get(session_id)
        if session is None:
            # reset was skipped. Recover rather than raise, unlike the starter.
            self.reset(session_id, {})
            session = self._sessions[session_id]
        slots = session.slots

        slots.observe(user_message, turn)
        session.raw_chars += len(user_message or "")
        query = slots.to_query()

        track = route(
            message=user_message or "",
            constraint_count=len(slots.constraints()),
            previous_entropy=session.last_entropy,
            config=self.config,
        )
        # Pass the disclosed phrases as well as the flattened query. The
        # flattening loses phrase boundaries, and those boundaries are exactly
        # what carries evidence here: the customer quotes the target product's
        # own text verbatim, so a whole-phrase hit before truncation is far
        # stronger than the sum of its tokens. The seam takes them optionally,
        # so this stays compatible with a retriever that ignores them.
        candidates = self.retriever.retrieve(query, track.width, slots.constraints())
        belief = self.ranker.believe(
            candidates, slots, slots.profile, depth=track.depth, sharpen=track.sharpen
        )
        decision = decide(belief, track.depth, self.config)
        if decision.depth != track.depth:
            # The over-generality cutoff. Rerank the narrower shortlist rather
            # than trimming the answer, so the wide low confidence pool is cut
            # off at source and the list the customer sees stays full length.
            belief = self.ranker.believe(
                candidates,
                slots,
                slots.profile,
                depth=decision.depth,
                sharpen=track.sharpen,
            )

        ranked = belief.ranking() or self._fallback(top_k)

        # Spec 6.5, the live semantic rerank. Off unless Config.use_llm is set,
        # because official scoring may run with networking disabled. It sees the
        # shortlist and what the customer said, and proposes an ordering; the
        # belief ordering above is what ships if it is unavailable, times out or
        # replies with nonsense. Per the layer boundary it never sets the belief
        # and never decides whether to commit, so a confident model cannot
        # override a confident belief, only reorder what the belief surfaced.
        rerank_usage = (0, 0)
        if self.config.use_llm and ranked:
            outcome = llm_rerank.rerank(
                slots.constraints(),
                [(asin, self.catalog.text(asin)) for asin in ranked[: TOP_K * 2]],
                self.config,
            )
            if outcome.used_llm:
                ranked = outcome.order + ranked[len(outcome.order) :]
                rerank_usage = (outcome.prompt_tokens, outcome.completion_tokens)

        attribute = slots.pick_attribute()
        message = phrase_question(
            attribute,
            [self.catalog.text(asin) for asin in ranked[:GROUNDING_WIDTH]],
            decision.overloaded,
        )

        session.record(
            TurnRecord(
                turn=turn,
                track=track.name,
                track_confidence=round(track.confidence, 4),
                width=track.width,
                depth=decision.depth,
                entropy=decision.entropy,
                peak_share=decision.peak_share,
                overloaded=decision.overloaded,
                committed=decision.commit,
                asked=attribute,
                constraints=len(slots.constraints()),
                retired=len(slots.retired_attributes),
                raw_chars=session.raw_chars,
                distilled_chars=len(query),
            )
        )
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
                extra={
                    "track": track.name,
                    "track_confidence": round(track.confidence, 4),
                    "width": track.width,
                    **decision.as_trace(),
                    "pivot": turn in slots.pivot_turns,
                    "retired": sorted(slots.retired_attributes),
                    "compression": round(session.compression(), 4),
                    # Diagnostic only, and empty unless the trace sink is on.
                    # `evaluation/rank_diagnostics.py` joins these against the
                    # known target to attribute a ranking loss to a term.
                    "retrieval_shortlist": [
                        {"parent_asin": asin, "score": round(float(score), 6)}
                        for asin, score in candidates
                    ],
                    "ranking": belief.components,
                },
            )
        )
        return response_module.build(
            message=message,
            ask_attribute=attribute,
            recommendations=ranked[: max(top_k, TOP_K)],
            # Token usage is a required disclosure. Zero when the rerank is off,
            # which is the default, so the reported total is honest either way.
            prompt_tokens=rerank_usage[0],
            completion_tokens=rerank_usage[1],
        )

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
