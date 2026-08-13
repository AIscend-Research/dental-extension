"""The verdict machine: two e-processes, an asymmetric burden, and a subpoena power.

This is the object that turns "a model produced a number" into "the system has
met a stated burden of proof, or has said why it cannot". Per shot it:

  1. reads the degradation channel and picks the calibration stratum,
  2. fixes its stakes from that channel alone (never from the diagnosis),
  3. updates both wealth processes with the shot's conformal p-values,
  4. renders a verdict if either burden is met at *any* point so far,
  5. otherwise issues a subpoena naming what to fix, if budget remains,
  6. otherwise escalates to a clinician.

Step 4 checks the running maximum, not the current wealth. That is what
anytime-validity means: a verdict reached at shot 2 stays legitimate even if
shot 3's evidence would have dragged the wealth back below the bar. The
alternative -- requiring the wealth to be above threshold at the moment you
happen to stop -- is both weaker and, counterintuitively, no more conservative.

Step 5's `REFER` outcome is not a failure mode bolted on for safety. It is the
system's third legal option, and the benchmark scores it as such: a session
that escalates has behaved correctly, it just has not produced a verdict, and
so it costs captures without earning a verdict in the headline metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from src.evidence.calibration import Calibrator
from src.evidence.ewealth import BettingStrategy, DegradationAwareBet, EWealth
from src.evidence.ladder import BurdenSpec
from src.evidence.view import EvidenceView
from src.sim.instructions import Instruction, instruction_for_factor

# The degradation channel speaks in degradation.py's vocabulary ("blur"); the
# instruction space speaks in latent-factor terms ("tremor"). One mapping, here.
DEGRADATION_TO_FACTOR: dict[str, str] = {
    "glare": "glare",
    "blur": "tremor",
    "low_light": "darkness",
    "angle": "tilt",
    "jpeg": None,  # not fixable by retaking, on purpose
}


class VerdictOutcome(Enum):
    """What a session can conclude."""

    CARIES = "caries"  # burden to convict met
    SOUND = "sound"  # burden to discharge met
    REFER = "refer"  # neither met, budget exhausted -> a human decides
    PENDING = "pending"  # neither met, budget remains -> subpoena more evidence


@dataclass
class Verdict:
    """The machine's decision after one shot.

    Attributes:
        outcome: see VerdictOutcome.
        instruction: what to fix, set only when outcome is PENDING.
        wealth_convict / wealth_discharge: running maxima after this shot.
        stratum: which calibration stratum this shot used.
        e_convict / e_discharge: the shot's e-values, whichever construction
            the calibrator supplies (conformal-calibrated or likelihood ratio).
        stake_convict / stake_discharge: the fractions actually bet.
    """

    outcome: VerdictOutcome
    instruction: Instruction | None = None
    wealth_convict: float = 1.0
    wealth_discharge: float = 1.0
    stratum: int = 0
    e_convict: float = 1.0
    e_discharge: float = 1.0
    stake_convict: float = 0.0
    stake_discharge: float = 0.0

    @property
    def is_terminal(self) -> bool:
        return self.outcome is not VerdictOutcome.PENDING

    @property
    def rendered(self) -> bool:
        """Did the system actually decide the case, as opposed to escalating?"""
        return self.outcome in (VerdictOutcome.CARIES, VerdictOutcome.SOUND)


@dataclass
class VerdictMachine:
    """Runs one case to a verdict.

    Args:
        calibrator: fitted Calibrator supplying conformal p-values.
        burden: the two-sided standard of proof to meet.
        strategy: how much to stake per shot. Must read only an EvidenceView.
        n_strata: must match what the calibrator was fitted with.
    """

    calibrator: Calibrator
    burden: BurdenSpec = field(default_factory=BurdenSpec)
    strategy: BettingStrategy = field(default_factory=DegradationAwareBet)
    n_strata: int = 4

    def __post_init__(self) -> None:
        self.convict = EWealth()
        self.discharge = EWealth()
        self.past_scores: list[float] = []
        self.shot_index = 0

    def observe(
        self,
        score: float,
        degradation: dict[str, float],
        predicted_usability: float,
        captures_remaining: int,
    ) -> Verdict:
        """Process one capture and decide what to do next.

        Args:
            score: the diagnosis channel's caries score for THIS shot.
            degradation: the confidence head's predicted severities for this shot.
            predicted_usability: the head's scalar usability for this shot.
            captures_remaining: shots left in the budget AFTER this one.

        Note the ordering inside: the EvidenceView -- and therefore both
        stakes -- is constructed before `score` is used for anything. That is
        the measurability condition, made structural rather than promised.
        """
        stratum = self.calibrator.stratum_of(predicted_usability)

        view = EvidenceView(
            shot_index=self.shot_index,
            degradation=dict(degradation),
            predicted_usability=float(predicted_usability),
            stratum=stratum,
            wealth_convict=self.convict.wealth,
            wealth_discharge=self.discharge.wealth,
            past_scores=tuple(self.past_scores),
            captures_remaining=captures_remaining,
        )
        lam_c = self.strategy.stake(view, "convict")
        lam_d = self.strategy.stake(view, "discharge")

        # only now is the diagnosis score allowed to matter
        e_c, e_d = self.calibrator.e_values(score, stratum)
        self.convict.update(e_c, lam_c)
        self.discharge.update(e_d, lam_d)
        self.past_scores.append(float(score))
        self.shot_index += 1

        verdict = Verdict(
            outcome=VerdictOutcome.PENDING,
            wealth_convict=self.convict.running_max,
            wealth_discharge=self.discharge.running_max,
            stratum=stratum,
            e_convict=e_c,
            e_discharge=e_d,
            stake_convict=lam_c,
            stake_discharge=lam_d,
        )

        met_convict = self.convict.has_crossed(self.burden.convict.threshold)
        met_discharge = self.discharge.has_crossed(self.burden.discharge.threshold)

        if met_convict and met_discharge:
            # Both burdens met: the evidence is internally contradictory, which
            # is exactly the case a human should see. Rendering either verdict
            # here would be picking a winner by tiebreak, and a tiebreak is not
            # proof. Escalate instead.
            verdict.outcome = VerdictOutcome.REFER
        elif met_convict:
            verdict.outcome = VerdictOutcome.CARIES
        elif met_discharge:
            verdict.outcome = VerdictOutcome.SOUND
        elif captures_remaining > 0:
            verdict.instruction = self._subpoena(view)
        else:
            verdict.outcome = VerdictOutcome.REFER

        return verdict

    def _subpoena(self, view: EvidenceView) -> Instruction:
        """Name what to fix, using only the degradation channel."""
        worst = view.dominant_degradation
        return instruction_for_factor(DEGRADATION_TO_FACTOR.get(worst))
