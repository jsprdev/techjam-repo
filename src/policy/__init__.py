"""Decision policy: routing, question selection, and when to commit.

Deliberately empty of re-exports. `src/state/slots.py` imports
`src.policy.question`, and a package init that eagerly imported `commit` (which
reads `src.state.belief`) would make that a cycle. Import the submodule you
need.
"""
