"""Evidence as wealth: p-to-e calibration and the betting (supermartingale) process.

A gambler starts with one unit and bets against the null. If the null is true
the game is fair-or-worse, so the wealth is a nonnegative supermartingale and
Ville's inequality gives, for any stopping time tau,

    P_H0( exists t <= tau : W_t >= 1/alpha ) <= alpha.

"Exists t" is what buys the retake loop its licence: the bound holds no matter
how adaptively the decision to take another photo was made, so long as each
bet is fixed *before* seeing the outcome it is betting on. That measurability
condition is enforced upstream by `src.evidence.view.EvidenceView`.

Two design choices worth stating rather than burying:

1. The bet is a *fraction* lambda in [0, 1] of current wealth, giving the
   update W_t = W_{t-1} * (1 + lambda_t (e_t - 1)). With e_t >= 0 and
   E[e_t | past] <= 1 this is nonnegative (worst case 1 - lambda) and has
   conditional expectation <= W_{t-1}. Both properties are needed and both
   are checked in tests/test_evidence.py.

2. lambda_t may depend on the degradation channel, which is the whole point:
   the system stakes little on a shot it can already tell is bad, and stakes
   heavily on a clean one -- without that costing any validity, because the
   degradation reading is available before the diagnosis is bet on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.evidence.view import EvidenceView

# Default exponent for the p-to-e calibrator. 0.5 is the common choice: it is
# the most powerful member of the family against moderate alternatives, and it
# is the one Vovk & Wang single out.
DEFAULT_KAPPA = 0.5


def p_to_e(p: float | np.ndarray, kappa: float = DEFAULT_KAPPA) -> float | np.ndarray:
    """Vovk-Wang calibrator: e = kappa * p^(kappa - 1), for kappa in (0, 1).

    This is an admissible p-to-e calibrator: if p is (super-)uniform under the
    null then E[e] <= 1, since integral over [0,1] of kappa p^(kappa-1) dp = 1.
    It is decreasing in p, so small p (surprising evidence) means large e
    (the gambler gets rich).

    p = 0 would give infinite wealth from a single shot, so p is floored at a
    small positive value. In practice the conformal p-values from
    src/evidence/calibration.py are bounded below by 1/(n+1) anyway, so this
    floor only ever fires on degenerate input.
    """
    if not 0.0 < kappa < 1.0:
        raise ValueError(f"kappa must be in (0, 1), got {kappa}")
    p_arr = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0)
    e = kappa * np.power(p_arr, kappa - 1.0)
    return float(e) if np.isscalar(p) or np.ndim(p) == 0 else e


class BettingStrategy:
    """Chooses the stake lambda in [0, 1] for the next shot.

    `stake` receives only an `EvidenceView`, so any strategy written against
    this interface is G_t-measurable by construction and inherits the validity
    guarantee. Subclasses must not reach outside the view for information.
    """

    name = "base"

    def stake(self, view: EvidenceView, side: str) -> float:
        """Fraction of wealth to bet on this shot.

        Args:
            view: G_t-measurable information.
            side: "convict" or "discharge" -- the two e-processes can stake
                differently, e.g. requiring more evidence to convict.
        """
        raise NotImplementedError


@dataclass
class ConstantBet(BettingStrategy):
    """Bet the same fraction every shot, ignoring image quality.

    Intended as the naive baseline; it turned out to be the better rule. E4
    (ablation 2) finds constant staking dominates `DegradationAwareBet` on
    verdicts per capture, and increasingly so as the stake rises.

    The explanation is that the information is already spent. Stratified
    e-values are computed *within* a degradation stratum, so a bad photograph
    has already been compared against the right null and its e-value already
    sits near 1. Shrinking the stake on top of that discounts the same
    evidence twice, and since predicted usability is rarely near 1, it mostly
    just slows accumulation on the good shots too.

    Note the risk that the headline number hides: at lam = 1.0 a single
    e-value near zero destroys the wealth permanently, with no way back. The
    accuracy column drops slightly at all-in staking for exactly that reason,
    so `lam` around 0.8 is the better default -- most of the power, and the
    process keeps a fifth of its wealth through a bad shot.
    """

    lam: float = 0.5
    name: str = "constant"

    def stake(self, view: EvidenceView, side: str) -> float:
        return float(np.clip(self.lam, 0.0, 1.0))


@dataclass
class DegradationAwareBet(BettingStrategy):
    """Stake in proportion to how usable the current shot looks.

    lambda_t = lam_max * predicted_usability_t ** sharpness

    It is legal precisely because `predicted_usability` comes from the
    degradation channel D_t, which G_t contains -- the system is allowed to
    know its photo is bad before deciding how much to stake on what the photo
    says. So this rule demonstrates that the firewall permits quality-adaptive
    betting at no cost to validity.

    It does not, however, pay off: E4 finds it loses to a constant stake, for
    the double-counting reason given in `ConstantBet`. Kept as a first-class
    strategy because the negative result is worth reporting and because the
    conclusion is specific to stratified e-values -- against an *unstratified*
    construction, where the e-value has not already been conditioned on image
    quality, quality-adaptive staking has something left to contribute.
    """

    lam_max: float = 0.8
    sharpness: float = 1.5
    floor: float = 0.02
    name: str = "degradation_aware"

    def stake(self, view: EvidenceView, side: str) -> float:
        u = float(np.clip(view.predicted_usability, 0.0, 1.0))
        lam = self.lam_max * (u**self.sharpness)
        return float(np.clip(lam, self.floor, 1.0))


class EWealth:
    """One nonnegative supermartingale: the wealth of a gambler against one null.

    Attributes:
        wealth: current W_t, starting at 1.0.
        history: every W_t including the initial 1.0, for plotting and for the
            running-maximum check that Ville's inequality is about.
    """

    def __init__(self, kappa: float = DEFAULT_KAPPA):
        self.kappa = kappa
        self.wealth = 1.0
        self.history: list[float] = [1.0]

    def update(self, e_value: float, lam: float) -> float:
        """Bet fraction `lam` on an e-value; return the new wealth.

        Raises if `lam` leaves [0, 1] or if the e-value is negative -- outside
        those ranges the process can go negative and stops being a
        supermartingale, which would silently void every guarantee downstream.
        Better to fail here than to report a coverage number that does not mean
        what it says.
        """
        if not 0.0 <= lam <= 1.0:
            raise ValueError(f"betting fraction must be in [0, 1], got {lam}")
        e = float(e_value)
        if e < 0.0 or not np.isfinite(e):
            raise ValueError(f"e-value must be finite and nonnegative, got {e_value}")
        self.wealth = float(self.wealth * (1.0 + lam * (e - 1.0)))
        self.history.append(self.wealth)
        return self.wealth

    def update_from_p(self, p_value: float, lam: float) -> float:
        """Convenience: calibrate a p-value to an e-value, then bet on it."""
        return self.update(float(p_to_e(p_value, self.kappa)), lam)

    @property
    def running_max(self) -> float:
        """Largest wealth ever reached. This is the quantity Ville bounds."""
        return float(max(self.history))

    def has_crossed(self, threshold: float) -> bool:
        """Whether the process EVER crossed -- not just where it sits now.

        Using the running maximum rather than the current value is the whole
        content of anytime-validity: a verdict rendered the moment the
        threshold was hit is legitimate even if further evidence would have
        pulled the wealth back down.
        """
        return self.running_max >= threshold
