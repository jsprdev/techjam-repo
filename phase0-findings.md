# Phase 0 findings

Resolves the UNVERIFIED open questions in `techjam-detailed-agent-spec.md` section 10, against the
actual participant kit (`TechJam2026/techjam-conversational-search`, tag `participant-kit`) and the
frozen 50,000 row catalog. Reproduce the catalog numbers with `python evaluation/diagnostics.py`.

## 10.1 Does the simulator answer `ask_attribute`? RESOLVED, yes

`evaluator/local_evaluator.py:customer_reply` consumes `ask_attribute` programmatically. It returns
up to two of the target's own constraint strings that are not yet disclosed and that
`classify_constraint` maps to the attribute asked. Question selection therefore has real information
gain and spec section 5.7 stands.

Three qualifications:

- Asking an attribute the target has no constraint for returns "I don't have an additional preference
  for X" and burns the turn. Coverage-weighted attribute selection is not optional.
- `classify_constraint` is a crude keyword matcher, not a semantic one. It decides attribute buckets
  by scanning for literal words such as `cotton`, `black`, `sleeve`, `hiking`. Model the classifier,
  not the concept.
- Boundary sessions answer the first question with "I don't have a preference, use your judgment".

## The larger finding: the simulator is fully deterministic and computable

`intent_card(product)` builds the customer's constraints from the target's own `features` and
`details`, plus a regex sweep for a material and a colour, plus a price line. Every sentence the
simulated customer speaks is a verbatim substring of the target product's catalog record. Observed:

    target B09PYB7B6Z, features contain "Triple Moon Pentagram Symbol"
    ask 'feature' -> "For that, what matters is: Triple Moon Pentagram Symbol; ..."

Turn 1 also leaks the last two components of the target's own `categories` path via
`coarse_category`, in every scenario.

Consequence: this is not a fuzzy natural language matching problem. Exact phrase matching of the
customer's utterances against catalog text is close to an oracle. Retrieval effort should go there
first, before any semantic modelling.

## 10.2 MRR timing. RESOLVED

The session breaks on the first turn where the target appears in the normalised top 10, and
`reciprocal_rank` is `1 / rank` at that moment. A miss scores 0 and is charged turn 11 for MTTC.

Note the gate: `if override_applied and target in ranked`. For `intent_override` sessions
`override_applied` starts False and flips at turn 3 or 4 (seeded per sample). Hits before that turn
are **not counted**. Those sessions have a floor of 3 turns no matter how good retrieval is.

## 10.3 Profile contents. RESOLVED, and weaker than the spec assumed

`user_profile` carries only `average_prior_rating`, `preference_tags`, `purchase_frequency`,
`rating_style`, and a one line `summary`. There is no brand, category, or price history.

Spec section 5.3 proposed brand, category, and price band continuity as a collaborative filtering
prior. That signal does not exist in this data. The only usable session-start priors are
`rating_number` popularity and the soft `preference_tags`.

## 10.4 Catalog fields. RESOLVED

Ten fields survived: `parent_asin`, `title`, `features`, `description`, `price`, `categories`,
`details`, `average_rating`, `rating_number`, `store`. Dropped entirely: `main_category`,
`bought_together`, `images`, `videos`. Any design referencing `main_category` needs rewriting to use
`categories`.

## Attribute coverage audit, before any LLM extraction

    100.0%  category      categories
     99.4%  brand         store
     89.6%  feature       features
     21.1%  budget        price
      4.9%  color         details.Color
      4.3%  material      details.Material
      3.5%  style         details.Style
      1.8%  size          details.Size
      0.0%  use_case      no source field

`details` holds 287 distinct keys across the catalog and is dominated by logistics fields
(`Date First Available` 93.8%, `Department` 87.2%, `Package Dimensions` 54.1%). The four attributes
the spec expected to read straight out of `details` are populated in under 5% of rows.

This cuts both ways. Entropy based question selection over the raw `details` dict is dead on arrival,
which is the case for the Layer 0 extraction pass in spec 6.1. But the evaluator does not read
`details` for material or colour either: it regexes the whole product text. Recovering those two
attributes is a regex job over `features` and `description`, not an LLM job.

## Offline extraction cost

Reading title, features, description, details, and categories for all 50,000 products is roughly
12.4M input tokens, or 9.7M trimmed to 1200 characters per product. At an assumed 100 output tokens
per product (5M output), the standard-rate order of magnitude is tens of dollars on a small model and
low hundreds on a frontier model, halved on the Batch API.

## 10.5 Config and submission rules

Not yet read. `docs/evaluation_config.json` and `docs/submission_rules.md` are in the kit.

## Recommended revision to build order

1. Exact and near-exact phrase retrieval over `title`, `features`, `description`, `categories`, since
   the customer speaks in catalog substrings.
2. Recall ceiling measurement on the 200 public sessions.
3. Regex recovery of material and colour, mirroring the evaluator's own patterns.
4. Attribute selection weighted by the coverage table above.
5. LLM derivation only for `use_case` and soft attributes, and only if step 2 shows headroom that
   retrieval alone cannot reach.
