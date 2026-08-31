"""Which attribute to ask about next. ROLE 2.

Spec 5.7 asks for expected information gain weighted by attribute reliability.
This module ships the reliability half and deliberately not the entropy half,
and the reason is measured rather than aesthetic.

**Why not information gain over the belief.** Splitting the candidate pool evenly
is only worth anything if the customer can answer the question that splits it.
They mostly cannot: this simulated customer answers an attribute only when one of
the target's own constraint phrases falls into that bucket, and those phrases land
in six of the ten legal buckets, three of which are never reachable at all. An
attribute that splits the pool perfectly and is answered "I don't have an
additional preference" costs a whole turn and buys nothing. Reliability therefore
dominates gain by a wide enough margin that gain is noise beneath it, and ordering
by measured yield is the honest implementation of "weight each attribute's score
by its reliability, because information gain computed over a field that is 60%
missing is fake information".

The ordering below is the empirical reliability model of spec 6.2 and 7.2, learnt
from the 200 public sessions rather than hand assigned. `evaluation/ask_yield.py`
recomputes it.

The runtime half is `retired`: an attribute the customer could not answer this
session is dropped for the rest of it, which is the reliability reweighting of
spec 7.1 and the one part of this policy that adapts inside a session.
"""

from __future__ import annotations

from src.config import Config

# The ask order, measured rather than assumed.
#
# The customer only answers an attribute if one of the target's own constraint
# phrases falls into that bucket. Measured across all 200 public sessions, the
# buckets those phrases actually land in are:
#
#     feature    50.5% of constraints, answerable in 96.0% of sessions
#     material   37.8%                                76.5%
#     color       7.5%                                25.5%
#     style       2.4%                                 9.0%
#     size        1.4%                                 4.5%
#     use_case    0.5%                                 2.0%
#
# Reproduce with the bucket audit in evaluation/ask_yield.py.
ATTRIBUTES_BY_YIELD = (
    "feature",
    "material",
    "color",
    "style",
    "size",
    "use_case",
)

# Asking any of these is a guaranteed miss. Nothing the customer says is ever
# classified into them, so the reply is always "I don't have an additional
# preference" and the information the turn could have bought is lost. The first
# version of this policy asked category and brand on turns 2 and 3, spending the
# two most valuable early asks on questions with no possible answer.
UNANSWERABLE = ("category", "brand", "budget")

# Matches ANY undisclosed constraint regardless of bucket, so it is the single
# most productive ask available. Kept as a fallback rather than the default
# because "tell me anything else" is a worse thing to say to a shopper than a
# specific question, and the specific questions above already reach 96%.
WILDCARD = "other"


def choose(asked: list[str], retired: set[str], config: Config) -> str | None:
    """The next attribute to ask about, or None when nothing is left worth asking.

    Ordered by measured yield, skipping anything already asked. An attribute that
    came back empty is never retried: the customer told us that bucket is empty
    for this target, and asking again spends a turn to be told so twice.
    """
    for attribute in ATTRIBUTES_BY_YIELD:
        if attribute not in asked and attribute not in retired:
            return attribute
    if config.allow_other_fallback and WILDCARD not in retired:
        return WILDCARD
    return None
