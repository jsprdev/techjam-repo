"""Frozen contracts between the four modules.

Four people edit this package for five days. Module ownership plus these
signatures, agreed before anyone writes logic, is what stops the last 48 hours
turning into merge conflicts. Anyone may propose a change to this file. Nobody
changes one silently, because every change here breaks somebody else's build.

Owners:
    Retriever   role 1, src/retrieval/
    SlotState   role 2, src/state/
    Ranker      role 3, src/rank/
    Agent       role 3, src/agent.py, shape fixed by the organiser
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

# A scored catalog item. The only currency passed between modules: keeping it a
# plain tuple means no module has to import another module's dataclass.
Candidate = tuple[str, float]

# The ten values the organiser's contract allows in `ask_attribute`, plus None.
# Sending anything else is a contract violation, so validate before returning.
ASK_ATTRIBUTES: frozenset[str] = frozenset(
    {
        "category",
        "material",
        "color",
        "size",
        "style",
        "brand",
        "budget",
        "feature",
        "use_case",
        "other",
    }
)

# Hard limits from docs/evaluation_config.json. Enforce these in our own code
# rather than trusting the evaluator, because exceeding the turn cap is a zero.
MAX_TURNS = 10
TOP_K = 10


@runtime_checkable
class Retriever(Protocol):
    """Role 1. Turns a query string into scored catalog candidates.

    Must run entirely in memory and must not touch the network, because the
    organiser may score us with networking disabled.
    """

    def retrieve(self, query: str, k: int) -> list[Candidate]:
        """Return up to `k` candidates, best first.

        `k` is the truncation width chosen by the caller and varies per turn:
        narrow on the Buying track, wide on Browsing. Returning fewer than `k`
        is fine. Returning an empty list is allowed but costs the turn.
        """
        ...


@runtime_checkable
class SlotState(Protocol):
    """Role 2. The accumulated picture of what the customer wants.

    Consumers only need to read it. All mutation lives behind role 2's own
    methods, so retrieval and ranking never write to the state they read.
    """

    turn: int

    def to_query(self) -> str:
        """Flatten current belief into one retrieval query string.

        This is the single seam where dialogue feeds retrieval. Everything the
        agent knows has to survive this call, so if information is not in this
        string, retrieval cannot use it.
        """
        ...

    def constraints(self) -> list[str]:
        """Every constraint phrase disclosed so far, most recent last.

        The simulated customer speaks in phrases lifted verbatim from the target
        product's own record, so ranking treats these as near-literal evidence
        rather than as loose natural language.
        """
        ...


@runtime_checkable
class Ranker(Protocol):
    """Role 3. Reorders a shortlist into the final answer."""

    def rank(
        self,
        candidates: list[Candidate],
        slots: SlotState,
        profile: dict,
    ) -> list[str]:
        """Return `parent_asin` values, best first.

        Only the first ten valid unique values are scored, but returning more is
        allowed by the contract. Never return an empty list when candidates were
        supplied: an unranked guess still has a chance of hitting, and a session
        can only end favourably by surfacing the target.
        """
        ...
