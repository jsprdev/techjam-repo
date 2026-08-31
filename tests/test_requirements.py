"""The pinned dependencies must actually install on the Python we support.

An exact pin is the right call for reproducibility, but it introduces a failure
mode a floor does not have: a version can declare a Requires-Python that
excludes the interpreter a judge is using, and `pip install -r
requirements.txt` then fails outright with "No matching distribution found".
That is worse than the score drift the pin exists to prevent, and it is
invisible until someone on a different interpreter tries it.

This nearly shipped. numpy 2.5.2 declares Requires-Python >=3.12, so pinning it
would have made the project uninstallable on Python 3.11.
"""

from __future__ import annotations

import re
from pathlib import Path

REQUIREMENTS = Path(__file__).resolve().parent.parent / "requirements.txt"

# The floor the project claims to support. Raising this is a real decision:
# it narrows who can run the submission, so it should be deliberate.
MINIMUM_PYTHON = (3, 11)

# Requires-Python floors for the versions pinned below, read from each
# distribution's own metadata. A pin change without a check here is the bug
# this file exists to catch.
KNOWN_PYTHON_FLOORS = {
    ("numpy", "2.4.6"): (3, 11),
    ("numpy", "2.5.2"): (3, 12),
    ("scikit-learn", "1.9.0"): (3, 10),
}


def parse_pins() -> dict[str, str]:
    """Package name to exact version, for the `==` lines only."""
    pins = {}
    for line in REQUIREMENTS.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        match = re.fullmatch(r"([A-Za-z0-9._-]+)==([0-9][^\s]*)", line)
        if match:
            pins[match.group(1).lower()] = match.group(2)
    return pins


def test_the_runtime_dependencies_are_pinned_not_floored():
    """A floor does not reproduce a published number, which the rules require."""
    pins = parse_pins()
    assert "numpy" in pins, "numpy must be pinned exactly"
    assert "scikit-learn" in pins, "scikit-learn must be pinned exactly"


def test_no_pin_excludes_the_minimum_supported_python():
    """A pin the judge's interpreter cannot install is worse than no pin."""
    for package, version in parse_pins().items():
        floor = KNOWN_PYTHON_FLOORS.get((package, version))
        assert floor is not None, (
            f"{package}=={version} is pinned but its Requires-Python floor is not "
            f"recorded in KNOWN_PYTHON_FLOORS. Look it up before changing a pin: "
            f"a version that needs a newer Python than {MINIMUM_PYTHON} makes "
            f"`pip install -r requirements.txt` fail outright."
        )
        assert floor <= MINIMUM_PYTHON, (
            f"{package}=={version} requires Python >={floor[0]}.{floor[1]}, which "
            f"excludes the supported {MINIMUM_PYTHON[0]}.{MINIMUM_PYTHON[1]}."
        )
