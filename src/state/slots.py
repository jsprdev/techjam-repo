"""v0 dialogue state. OWNED BY ROLE 2, this is your starting point.

Deliberately the simplest thing that satisfies the SlotState protocol: it
accumulates every phrase the customer says and concatenates them into a query.
It has no override handling, no decay, and a placeholder question policy.

Role 2's job, with what Phase 0 already established:

1. Override. `intent_override` sessions flip at turn 3 or 4 with a message
   starting "Actually, ignore my earlier preference." Hits before the flip are
   NOT counted by the evaluator, so those 30 sessions have a hard floor of three
   turns. Accumulating through the flip actively hurts: the old constraint is
   now wrong and needs erasing.
2. Decay. `Config.slot_decay` is unused. A turn one inference should not carry
   the same weight at turn eight as something just stated.
3. Question policy, the real work. The evaluator's `customer_reply` returns up
   to two undisclosed constraints whose `classify_constraint` bucket matches the
   attribute asked. Asking an attribute the target has no constraint for wastes
   the information, though not the turn, since asking is free. So model
   `classify_constraint`, which is a plain keyword matcher, and pick the
   attribute most likely to yield. `pick_attribute` below just cycles.

Read `phase0-findings.md` before starting. It carries the coverage table that
should weight your attribute choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.config import Config

# Ordered by how often the catalog can actually answer the attribute, measured
# in phase0-findings.md. Asking about size (1.8% populated) is close to wasted.
ATTRIBUTES_BY_COVERAGE = (
    "feature",
    "category",
    "brand",
    "budget",
    "color",
    "material",
    "style",
    "use_case",
    "size",
)


@dataclass
class Slots:
    """Accumulated dialogue state for one session."""

    config: Config = field(default_factory=Config)
    turn: int = 0
    profile: dict = field(default_factory=dict)
    _phrases: list[str] = field(default_factory=list)
    _asked: list[str] = field(default_factory=list)

    def observe(self, user_message: str, turn: int) -> None:
        """Record what the customer said this turn."""
        self.turn = turn
        cleaned = user_message.strip()
        if cleaned and cleaned not in self._phrases:
            self._phrases.append(cleaned)

    def constraints(self) -> list[str]:
        return list(self._phrases)

    def to_query(self) -> str:
        """Flatten everything known into one retrieval query.

        The profile's preference_tags are appended once as weak priors. They are
        vague ("fit", "comfort") and Phase 0 found the profile carries no brand,
        category or price history, so do not expect much from them.
        """
        parts = list(self._phrases)
        tags = self.profile.get("preference_tags") or []
        parts.extend(str(tag) for tag in tags)
        return " ".join(parts)

    def pick_attribute(self) -> str | None:
        """PLACEHOLDER. Cycles through attributes by catalog coverage.

        Role 2 replaces this with real expected information gain against the
        current candidate pool.
        """
        for attribute in ATTRIBUTES_BY_COVERAGE:
            if attribute not in self._asked:
                self._asked.append(attribute)
                return attribute
        if self.config.allow_other_fallback:
            return "other"
        return None

    @property
    def asked(self) -> list[str]:
        return list(self._asked)
