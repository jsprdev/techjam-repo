"""Dialogue state: what the customer has actually told us. ROLE 2.

The customer's utterances arrive as whole sentences wrapped in conversational
framing, and a large fraction of them carry no information at all. Feeding them
to retrieval verbatim, as the first version did, poisons the query: by turn six
a typical session had accumulated "I don't have an additional preference for
brand", "... for budget" and "... for color", contributing the tokens brand,
budget, colour, preference and additional to a search over a clothing catalog.

So this module's real job is separating signal from framing before anything
downstream sees it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.config import Config

# Ordered by how often the catalog can answer, measured in phase0-findings.md.
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

# Utterances that carry no information. Matching these is worth more than any
# other single change in this module: they are frequent, and every one of them
# adds noise tokens to the query that actively compete with real constraints.
NO_INFORMATION = (
    re.compile(r"^i don't have an additional preference for\b", re.I),
    re.compile(r"^i don't have a preference for\b", re.I),
    re.compile(r"^those options are not quite right yet\b", re.I),
)

# Conversational framing wrapped around the payload. Everything before the
# colon is scaffolding; the constraint itself follows it.
FRAMING = (
    re.compile(r"^for that,\s*what matters is:\s*", re.I),
    re.compile(r"^a key requirement is:\s*", re.I),
    re.compile(r"^actually,\s*ignore my earlier preference\.\s*what i need is:\s*", re.I),
    re.compile(r"^i'?m looking for\s+", re.I),
)

# The opening turn names the category and may append a constraint after a full
# stop. "but I'm still exploring" is filler on the Browsing track.
# The category must be GREEDY. A non-greedy [^.,]+? with an optional suffix
# matches a single character, silently turning "Accessories Belts" into "A" and
# destroying the strongest signal in the session.
OPENING = re.compile(
    r"^i'?m looking for\s+(?P<category>[^.,]+)"
    r"(?:,\s*but i'?m still exploring)?"
    r"[.,]?\s*(?P<rest>.*)$",
    re.I,
)


@dataclass
class Slots:
    """Accumulated dialogue state for one session."""

    config: Config = field(default_factory=Config)
    turn: int = 0
    profile: dict = field(default_factory=dict)
    category: str = ""
    _phrases: list[str] = field(default_factory=list)
    _asked: list[str] = field(default_factory=list)
    _informative_turns: int = 0

    # -- ingest --------------------------------------------------------------

    def observe(self, user_message: str, turn: int) -> None:
        self.turn = turn
        text = (user_message or "").strip()
        if not text:
            return

        if turn == 1:
            match = OPENING.match(text)
            if match:
                self.category = match.group("category").strip()
                text = match.group("rest").strip()
                if not text:
                    return

        if self.config.drop_no_information and any(p.match(text) for p in NO_INFORMATION):
            return

        for phrase in self._payloads(text):
            if not phrase:
                continue
            # Deliberately NOT deduplicated. A constraint the customer states
            # twice is one they care about, and repeating it in the query
            # raises its term frequency. Treating repetition as redundancy
            # measurably loses score. Config.dedupe_phrases exists to re-test
            # this rather than take it on faith.
            if self.config.dedupe_phrases and phrase in self._phrases:
                continue
            self._phrases.append(phrase)
            self._informative_turns += 1

    def _payloads(self, text: str) -> list[str]:
        """Strip framing and split a disclosure into its individual constraints.

        The simulator joins multiple constraints with "; ", and each one is
        lifted verbatim from the target product's own record, so splitting them
        apart gives ranking whole phrases it can match literally.
        """
        for pattern in FRAMING:
            stripped = pattern.sub("", text)
            if stripped != text:
                text = stripped
                break
        return [part.strip(" .;") for part in text.split(";") if part.strip(" .;")]

    # -- read ----------------------------------------------------------------

    def constraints(self) -> list[str]:
        """The disclosed constraint phrases, framing removed, most recent last."""
        return list(self._phrases)

    def to_query(self) -> str:
        parts: list[str] = []
        if self.category:
            # The opening category is the single most reliable signal we get:
            # it is the tail of the target's own category path.
            parts.append(self.category)
        parts.extend(self._phrases)
        if self.config.use_profile_tags:
            parts.extend(str(tag) for tag in (self.profile.get("preference_tags") or []))
        return " ".join(parts)

    @property
    def informative_turns(self) -> int:
        """How many turns actually told us something. Drives the ask policy."""
        return self._informative_turns

    # -- ask policy ----------------------------------------------------------

    def pick_attribute(self) -> str | None:
        for attribute in ATTRIBUTES_BY_COVERAGE:
            if attribute not in self._asked:
                self._asked.append(attribute)
                return attribute
        return "other" if self.config.allow_other_fallback else None

    @property
    def asked(self) -> list[str]:
        return list(self._asked)
