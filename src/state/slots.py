"""Dialogue state: what the customer has actually told us. ROLE 2.

The customer's utterances arrive as whole sentences wrapped in conversational
framing, and a large fraction of them carry no information at all. Feeding them
to retrieval verbatim, as the first version did, poisons the query: by turn six
a typical session had accumulated "I don't have an additional preference for
brand", "... for budget" and "... for color", contributing the tokens brand,
budget, colour, preference and additional to a search over a clothing catalog.

So this module's real job is separating signal from framing before anything
downstream sees it.

Two of the three operations the spec's state machine requires (5.5) live here:

- **Accumulation.** A new constraint arrives and joins the state.
- **Override.** The customer pivots, and the superseded constraints are demoted
  rather than deleted. See `OVERRIDE` below for why deletion is wrong here.
- **Decay.** Confidence in a constraint weakens each turn it is not reinforced,
  exposed through `constraint_weights` and consumed by the ranker.

`_exhausted` is the runtime reliability reweighting of spec 7.1: an attribute the
customer could not answer is retired for the rest of the session, so the policy
adapts its own question plan from evidence gathered inside the session. Which
attribute to ask is `src/policy/question.py`; this file only records the answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.config import Config
from src.policy import question as question_policy

# Utterances that carry no information. Matching these is worth more than any
# other single change in this module: they are frequent, and every one of them
# adds noise tokens to the query that actively compete with real constraints.
NO_INFORMATION = (
    re.compile(r"^i don't have an additional preference for\b", re.I),
    re.compile(r"^i don't have a preference for\b", re.I),
    re.compile(r"^those options are not quite right yet\b", re.I),
)

# A pivot: the customer redirects rather than adds. The first pattern is the
# simulator's exact wording, the rest are the ordinary English a real shopper
# uses, so the detector is not purely fitted to this harness.
#
# MEASURED, AND IT CHANGES THE DESIGN: across all 30 public intent_override
# sessions the preference the customer says to ignore is itself lifted from the
# target product's own record, and is literally present in that product's
# searchable text in 28 of the 30 (the other two differ only by "key: value"
# against "key value" formatting). The pivot is a change of emphasis, not a
# contradiction. Erasing the old slot therefore deletes true evidence about the
# target. So an override demotes, exactly as spec 5.4 requires everywhere else,
# and `Config.override_demote = 0.0` reproduces literal erasure for anyone who
# wants to check that claim rather than take it. Reproduce with
# evaluation/override_audit.py.
OVERRIDE = (
    re.compile(r"^actually,\s*ignore my earlier preference\.", re.I),
    re.compile(
        r"^(?:actually|instead|on second thought|scratch that|forget (?:that|it))\b",
        re.I,
    ),
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
class Phrase:
    """One constraint the customer disclosed, and when they disclosed it.

    The turn is what makes decay possible: without it every constraint weighs
    the same forever and a turn one aside outranks a turn eight statement.
    """

    text: str
    turn: int
    superseded: bool = False


@dataclass
class Slots:
    """Accumulated dialogue state for one session."""

    config: Config = field(default_factory=Config)
    turn: int = 0
    profile: dict = field(default_factory=dict)
    category: str = ""
    _phrases: list[Phrase] = field(default_factory=list)
    _asked: list[str] = field(default_factory=list)
    _exhausted: set[str] = field(default_factory=set)
    _informative_turns: int = 0
    _pivot_turns: list[int] = field(default_factory=list)

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

        if any(pattern.match(text) for pattern in OVERRIDE):
            self._pivot(turn)

        if any(pattern.match(text) for pattern in NO_INFORMATION):
            # The customer just told us this bucket is empty for their target.
            # Record it so the policy never spends another turn on it. This is
            # the runtime reliability reweighting of spec 7.1: the question plan
            # adapts from evidence gathered inside the session.
            if self._asked:
                self._exhausted.add(self._asked[-1])
            if self.config.drop_no_information:
                return

        known = {phrase.text for phrase in self._phrases}
        for text_part in self._payloads(text):
            if not text_part:
                continue
            # Deliberately NOT deduplicated. A constraint the customer states
            # twice is one they care about, and repeating it in the query
            # raises its term frequency. Treating repetition as redundancy
            # measurably loses score. Config.dedupe_phrases exists to re-test
            # this rather than take it on faith.
            if self.config.dedupe_phrases and text_part in known:
                continue
            self._phrases.append(Phrase(text=text_part, turn=turn))
            known.add(text_part)
            self._informative_turns += 1

    def _pivot(self, turn: int) -> None:
        """Handle an Intent Override: demote what came before, keep it all.

        Everything already accumulated is marked superseded, and what arrives on
        this turn lands at full weight. Nothing is removed, because the audit
        above shows the superseded constraint is still true of the target in 28
        of 30 public override sessions. `Config.override_demote` sets how far it
        falls; 0.0 is literal erasure and is there to be measured, not used.
        """
        self._pivot_turns.append(turn)
        for phrase in self._phrases:
            phrase.superseded = True

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
        return [phrase.text for phrase in self._phrases]

    def constraint_weights(self) -> list[float]:
        """Confidence in each constraint, aligned with `constraints()`.

        Two effects multiply. Decay weakens a constraint by `Config.slot_decay`
        for every turn since the customer last stated it, so recent evidence
        wins ties against old evidence. Supersession applies
        `Config.override_demote` to anything the customer pivoted away from.

        The ranker consumes this. `to_query` deliberately does not: retrieval
        takes a plain string, and the only way to express a weight in a bag of
        words is repetition, which collides with the measured finding that
        repeated phrases are already signal rather than noise. Decay therefore
        reweights the evidence, not the query, and that limit is stated in the
        README rather than hidden.
        """
        decay = self.config.slot_decay
        demote = self.config.override_demote
        weights: list[float] = []
        for phrase in self._phrases:
            age = max(0, self.turn - phrase.turn)
            weight = decay ** age if decay != 1.0 else 1.0
            if phrase.superseded:
                weight *= demote
            weights.append(weight)
        return weights

    def to_query(self) -> str:
        parts: list[str] = []
        if self.category:
            # The opening category is the single most reliable signal we get:
            # it is the tail of the target's own category path.
            parts.append(self.category)
        parts.extend(phrase.text for phrase in self._phrases)
        if self.config.use_profile_tags:
            parts.extend(str(tag) for tag in (self.profile.get("preference_tags") or []))
        return " ".join(parts)

    @property
    def informative_turns(self) -> int:
        """How many turns actually told us something. Drives the ask policy."""
        return self._informative_turns

    @property
    def pivot_turns(self) -> list[int]:
        """Turns on which the customer overrode their earlier preference."""
        return list(self._pivot_turns)

    @property
    def retired_attributes(self) -> set[str]:
        """Attributes the customer could not answer, retired for this session."""
        return set(self._exhausted)

    # -- ask policy ----------------------------------------------------------

    def pick_attribute(self) -> str | None:
        """Choose the next attribute to ask about, and remember having asked it.

        The choice itself is `src/policy/question.py`. This method stays because
        `SlotState` in `src/interfaces.py` is frozen and every consumer calls it,
        and because recording the ask is state, which belongs here rather than in
        the policy.
        """
        attribute = question_policy.choose(self._asked, self._exhausted, self.config)
        if attribute is not None:
            self._asked.append(attribute)
        return attribute

    @property
    def asked(self) -> list[str]:
        return list(self._asked)
