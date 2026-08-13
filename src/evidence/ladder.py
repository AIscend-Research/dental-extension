"""Standards of proof: the asymmetric burden ladder.

Law does not use one evidentiary bar. It uses a ladder, and it deliberately
puts different rungs on the two sides of a case, because the two ways of being
wrong do not cost the same. This package borrows the structure literally.

For a screening camera the asymmetry runs the way it does in medicine, not the
way it does in criminal law. Wrongly *convicting* a sound tooth costs an
unnecessary referral and some anxiety. Wrongly *discharging* a carious one
sends someone home with untreated disease, and in a setting where the health
worker will not be back for six months, that is the expensive error. So the
default configuration demands MORE evidence to discharge than to convict -- an
inversion of the courtroom, and one worth stating out loud in the paper
because the metaphor otherwise invites the opposite assumption.

The thresholds are 1/alpha, straight from Ville's inequality: crossing 1/alpha
at any stopping time happens with probability at most alpha when the null is
true. So the ladder is not a heuristic scale, it is a calibrated one -- each
rung names the error rate it is buying.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Standard:
    """One rung of the ladder.

    Attributes:
        name: the legal name, used in tables and figures.
        alpha: the error probability the rung guarantees under the null.
        description: what it means operationally.
    """

    name: str
    alpha: float
    description: str

    @property
    def threshold(self) -> float:
        """Wealth a verdict requires: 1/alpha, by Ville's inequality."""
        return 1.0 / self.alpha


PREPONDERANCE = Standard(
    "preponderance",
    0.5,
    "More likely than not. The weakest admissible bar; wealth must merely double.",
)
CLEAR_AND_CONVINCING = Standard(
    "clear_and_convincing",
    0.2,
    "Substantially more probable than not. Wealth must reach 5x.",
)
BEYOND_REASONABLE_DOUBT = Standard(
    "beyond_reasonable_doubt",
    0.05,
    "The criminal standard. Wealth must reach 20x.",
)
NEAR_CERTAINTY = Standard(
    "near_certainty",
    0.01,
    "Stricter than any legal standard; included to probe where the budget breaks.",
)

STANDARDS: list[Standard] = [
    PREPONDERANCE,
    CLEAR_AND_CONVINCING,
    BEYOND_REASONABLE_DOUBT,
    NEAR_CERTAINTY,
]

_BY_NAME = {s.name: s for s in STANDARDS}


def standard_by_name(name: str) -> Standard:
    if name not in _BY_NAME:
        raise KeyError(f"unknown standard {name!r}; have {sorted(_BY_NAME)}")
    return _BY_NAME[name]


@dataclass(frozen=True)
class BurdenSpec:
    """The two-sided burden a session must meet.

    `convict` is the bar for declaring caries; `discharge` is the bar for
    declaring the tooth sound. Defaults encode the medical asymmetry described
    in the module docstring: discharging is the more consequential call in a
    screening setting, so it carries the heavier burden.
    """

    convict: Standard = CLEAR_AND_CONVINCING
    discharge: Standard = BEYOND_REASONABLE_DOUBT

    @property
    def name(self) -> str:
        return f"convict={self.convict.name}/discharge={self.discharge.name}"

    @classmethod
    def symmetric(cls, standard: Standard) -> "BurdenSpec":
        """Same bar both ways -- the control for the asymmetry ablation."""
        return cls(convict=standard, discharge=standard)
