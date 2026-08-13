"""The measurability firewall, expressed as a type.

The validity theorem needs the betting fraction, the stopping decision, and the
choice of instruction to all be measurable with respect to

    G_t = sigma( D_1..D_t, Y_1..Y_{t-1} )

-- everything about the *degradation* channel up to and including the current
shot, but only the *diagnosis* channel up to the previous shot. Betting on
Y_t using Y_t is betting on a horse after the race.

Rather than assert that a policy respects G_t and hope, the two views below
make it structural. `EvidenceView` physically does not carry the current
diagnosis score, so a bettor handed one cannot condition on it even by
accident. A policy that wants to peek must accept a `PeekingView`, which is
both a louder name and a different type -- and every experiment that reports a
validity guarantee asserts the arm used `EvidenceView`.

This is why the unsound baselines in `src/bench/policies.py` are honest
baselines rather than straw men: they are not "the same policy with a bug",
they are policies that genuinely need information the sound ones are denied.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvidenceView:
    """G_t-measurable information: what a sound bettor is allowed to see.

    Attributes:
        shot_index: which capture this is, 0-based.
        degradation: predicted per-degradation severities for the CURRENT
            shot, from the confidence head. Allowed: this is the D_t channel.
        predicted_usability: the head's scalar usability for the current shot.
        stratum: the calibration stratum the current shot falls in, derived
            from `degradation`/`predicted_usability` alone.
        wealth_convict: wealth of the "there is caries" e-process, BEFORE this
            shot's update.
        wealth_discharge: wealth of the "the tooth is sound" e-process, before
            this shot's update.
        past_scores: diagnosis scores from *previous* shots only (Y_1..Y_{t-1}).
            Present because G_t genuinely includes them; the current shot's
            score is what is absent.
        captures_remaining: budget left, for policies that ration.
    """

    shot_index: int
    degradation: dict[str, float]
    predicted_usability: float
    stratum: int
    wealth_convict: float
    wealth_discharge: float
    past_scores: tuple[float, ...] = ()
    captures_remaining: int = 0

    @property
    def dominant_degradation(self) -> str:
        """Worst predicted artifact -- what a retake instruction should target."""
        if not self.degradation:
            return "blur"
        return max(self.degradation.items(), key=lambda kv: kv[1])[0]


@dataclass(frozen=True)
class PeekingView(EvidenceView):
    """EvidenceView plus the current shot's diagnosis score.

    Using this forfeits the anytime-validity guarantee. It exists so the
    benchmark can quantify exactly how much a policy gains, and how much
    type-I error it buys, by looking at the answer before betting on it.

    `current_score` defaults to NaN so that constructing one without actually
    supplying the forbidden quantity is a loud failure downstream rather than
    a silent zero.
    """

    current_score: float = field(default=float("nan"))
