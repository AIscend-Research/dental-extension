"""Evidence as wealth: anytime-valid burdens of proof for a retake loop.

The problem this package solves. A capture policy that may keep asking for
another photo until it is satisfied is doing *optional stopping*. Reporting the
final shot's p-value as though it were a single pre-planned test is invalid,
and the inflation is not small -- it is roughly the multiplicity of the number
of shots you allowed yourself. Standard split-conformal does not rescue this
either: it assumes exchangeability, and captures in a retake loop are
deliberately non-exchangeable (shot 2 exists *because* shot 1 had glare).

The construction here is testing-by-betting (Shafer; Ramdas et al.): evidence
against a hypothesis is the wealth of a gambler betting against it, the wealth
process is a nonnegative supermartingale under the null, and Ville's inequality
then bounds the probability that it *ever* crosses 1/alpha -- at any stopping
time, however adaptively chosen. Optional stopping stops being a sin and
becomes the design.

The one thing you must not do is bet using information you are testing. That is
the restriction the whole design is organised around, and it is enforced by the
interface rather than by discipline: a bettor is handed an `EvidenceView`
(src/evidence/view.py) that contains the degradation channel and the wealth
history and *cannot* contain the current diagnosis score. Policies that want to
peek must ask for a `PeekingView` explicitly, which marks them as an
unsound-baseline arm.

Modules:
    view         -- the measurability firewall (G_t) as a type
    calibration  -- degradation-stratified conformal p-values
    ewealth      -- p-to-e calibrators, betting strategies, the wealth process
    ladder       -- standards of proof (preponderance -> beyond reasonable doubt)
    verdict      -- the two-sided verdict machine that drives a session

The formal statement and proof sketch live in docs/theory_anytime_validity.md.
"""

from src.evidence.calibration import (
    Calibrator,
    LikelihoodRatioCalibrator,
    MarginalCalibrator,
    StratifiedCalibrator,
    usability_stratum,
)
from src.evidence.ewealth import BettingStrategy, EWealth, ConstantBet, DegradationAwareBet, p_to_e
from src.evidence.ladder import STANDARDS, Standard, standard_by_name
from src.evidence.verdict import Verdict, VerdictMachine, VerdictOutcome
from src.evidence.view import EvidenceView, PeekingView

__all__ = [
    "BettingStrategy",
    "Calibrator",
    "ConstantBet",
    "DegradationAwareBet",
    "EWealth",
    "EvidenceView",
    "LikelihoodRatioCalibrator",
    "MarginalCalibrator",
    "PeekingView",
    "STANDARDS",
    "Standard",
    "StratifiedCalibrator",
    "Verdict",
    "VerdictMachine",
    "VerdictOutcome",
    "p_to_e",
    "standard_by_name",
    "usability_stratum",
]
