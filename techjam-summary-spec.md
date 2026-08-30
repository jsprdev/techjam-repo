## What it does, in layers

**Layer 0, before anyone talks.** The catalog gets cleaned once, offline. Amazon's `details` field is a free-form mess where the same concept appears under different keys and half the entries are missing. An LLM normalises it into a consistent attribute table and derives what isn't there at all, like use case and formality. Nothing above works without this.

**Layer 1, the deterministic engine.** Reads the intent, routes to Buying or Browsing, pulls candidates through three routes, and holds a probability over the whole catalog. It decides what to ask and when to stop asking. Zero LLM calls, so it still runs if the API dies.

**Layer 2, the language layer.** Handles what maths can't: understanding "my sister's beach wedding," noticing when someone reverses direction, reranking the shortlist, phrasing the question like a person.

**Layer 3, adaptation.** Re-picks its own pipeline shape every turn, and learns from its failures between runs.

**The boundary between 1 and 2 is the whole design.** Language proposes, probability decides. The LLM never holds the belief and never chooses when to commit, because models are badly calibrated about their own confidence and calibration is the entire point.

---

## Where the pillars land

**Pillar I, Core Architecture.** Intent router splits Buying from Browsing and changes the retrieval weights and truncation width accordingly. Three routes merge in memory. LLM reranks the shortlist. All four named requirements sit in one flow.

**Pillar II, Dialog Strategy.** A slot tracker running alongside the belief, handling accumulation, override erasure, and decay. When the pool is overloaded and the belief goes flat, it cuts retrieval and asks instead of dumping a weak list.

**Pillar III, Self-Evolution.** This is the one to say carefully, because the brief asks for _runtime_, not just offline. At runtime, the track, the truncation width, and the ask-versus-commit call are re-decided every turn from the current belief, so the pipeline shape is not fixed at session start. Attributes that fail to move the distribution get downweighted mid-session. Offline, the failure loop rewrites its own extraction schema.

**Pillar IV, Evaluation.** Coverage is the retrieval ceiling, which we measure before building anything. Precision is the reranker. Efficiency is the commit rule, which asks only when asking pays.

---

**The one-liner:** every pillar is a consequence of one idea, that the system tracks its own uncertainty and re-plans around it each turn.

Two things to keep honest in the pitch. The Buying track uses steep reweighting rather than literal filtering, and that is a deviation worth naming. And long-term cross-session profiling has no eval surface here, since the rules define sessions as isolated, so say that rather than faking it.