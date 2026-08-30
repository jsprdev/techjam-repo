"""Prove the agent runs with no network access.

`docs/submission_rules.md` warns that organiser policy may disable network
access for official scoring, and requires us to document whether we depend on
it. This is the check that turns that claim into evidence rather than a hope.

It runs a real multi turn session in a subprocess with every socket entry point
poisoned, so any attempt to open a connection raises immediately and loudly.
Run it before every submission, and after anyone adds anything to
`src/language/`.

    python evaluation/verify_offline.py

Exits non-zero on the first sign of a network dependency, so it drops straight
into CI.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CATALOG = REPO_ROOT / "techjam-conversational-search/data/catalog.jsonl"

# Runs in a fresh interpreter so nothing already imported can hold a live
# socket. Poisoning happens before src is imported, which also catches a module
# that opens a connection at import time.
PROBE = r'''
import socket, sys, ssl

class Blocked(Exception):
    pass

def _blocked(*args, **kwargs):
    raise Blocked("network access attempted")

socket.socket = _blocked
socket.create_connection = _blocked
socket.getaddrinfo = _blocked
socket.gethostbyname = _blocked
ssl.SSLContext.wrap_socket = _blocked

sys.path.insert(0, {repo!r})
from src.agent import Agent
from src.response import violations

agent = Agent({catalog!r})
agent.reset("offline-probe", {{
    "purchase_frequency": "3-4 prior purchases",
    "average_prior_rating": None,
    "rating_style": "usually positive",
    "preference_tags": ["fit", "comfort"],
    "summary": "Prior purchases emphasize fit and comfort.",
}})

messages = [
    "I'm looking for Watches Wrist Watches. Stainless Steel Band",
    "For that, what matters is: Water Resistant; 3 Year Battery.",
    "Actually, ignore my earlier preference. What I need is: leather.",
    "I don't have an additional preference for color.",
]
for turn in range(1, 11):
    result = agent.respond("offline-probe", messages[(turn - 1) % len(messages)], turn, 10)
    problems = violations(result)
    if problems:
        print("CONTRACT VIOLATION on turn %d: %s" % (turn, problems))
        raise SystemExit(2)
    if not result["recommendations"]:
        print("EMPTY recommendations on turn %d" % turn)
        raise SystemExit(2)
print("OK 10 turns completed with every socket blocked")
'''


def main() -> None:
    catalog = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CATALOG
    if not catalog.exists():
        print(f"catalog not found at {catalog}")
        raise SystemExit(1)

    print(f"[offline] probing with sockets blocked, catalog {catalog}")
    completed = subprocess.run(
        [sys.executable, "-c", PROBE.format(repo=str(REPO_ROOT), catalog=str(catalog))],
        capture_output=True,
        text=True,
    )
    sys.stdout.write(completed.stdout)
    if completed.returncode != 0:
        sys.stderr.write(completed.stderr)
        print("\n[offline] FAILED. The agent depends on network access.")
        raise SystemExit(completed.returncode)
    print("[offline] PASS. Safe to submit as running fully offline.")


if __name__ == "__main__":
    main()
