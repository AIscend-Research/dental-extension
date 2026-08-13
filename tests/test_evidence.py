"""Tests for the evidence machinery (src/evidence/).

The two that matter most are `test_wealth_is_a_supermartingale_under_the_null`
and `test_ville_inequality_holds_under_optional_stopping`: between them they
are the property the whole framework rests on, checked numerically rather than
taken on trust from the algebra.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evidence.calibration import (
    LikelihoodRatioCalibrator,
    MarginalCalibrator,
    StratifiedCalibrator,
    quantile_edges,
    usability_stratum,
)
from src.evidence.ewealth import ConstantBet, DegradationAwareBet, EWealth, p_to_e
from src.evidence.ladder import BurdenSpec, PREPONDERANCE, standard_by_name
from src.evidence.verdict import VerdictMachine, VerdictOutcome
from src.evidence.view import EvidenceView, PeekingView


# ---------------------------------------------------------------------------
# p-to-e calibration
# ---------------------------------------------------------------------------


def test_p_to_e_has_unit_expectation_under_a_uniform_null():
    """The defining property: E[e] <= 1 when p is uniform."""
    rng = np.random.default_rng(0)
    p = rng.random(400000)
    for kappa in (0.2, 0.5, 0.8):
        e = np.asarray(p_to_e(p, kappa))
        assert e.mean() == pytest.approx(1.0, abs=0.02), kappa


def test_p_to_e_is_decreasing_and_rejects_bad_kappa():
    assert p_to_e(0.001) > p_to_e(0.1) > p_to_e(0.9)
    for bad in (0.0, 1.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="kappa"):
            p_to_e(0.5, bad)


# ---------------------------------------------------------------------------
# the wealth process
# ---------------------------------------------------------------------------


def test_wealth_rejects_invalid_stakes_and_evidence():
    w = EWealth()
    for bad_lam in (-0.01, 1.01):
        with pytest.raises(ValueError, match="betting fraction"):
            w.update(1.0, bad_lam)
    for bad_e in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="e-value"):
            w.update(bad_e, 0.5)


def test_wealth_stays_nonnegative_for_any_admissible_stake():
    rng = np.random.default_rng(1)
    for _ in range(200):
        w = EWealth()
        for _ in range(20):
            w.update(float(rng.exponential(1.0)), float(rng.random()))
            assert w.wealth >= 0.0


def test_wealth_is_a_supermartingale_under_the_null():
    """E[W_t | W_{t-1}] <= W_{t-1} when the p-values are uniform."""
    rng = np.random.default_rng(2)
    n = 200000
    for lam in (0.2, 0.5, 1.0):
        e = np.asarray(p_to_e(rng.random(n)))
        multiplier = 1.0 + lam * (e - 1.0)
        assert multiplier.mean() <= 1.0 + 0.02, lam
        assert (multiplier >= 0.0).all()


def test_ville_inequality_holds_under_optional_stopping():
    """P(sup_t W_t >= 1/alpha) <= alpha, however adaptively we stop.

    The stopping rule here deliberately peeks at the wealth itself -- which is
    allowed, since wealth is past-measurable -- and stops the moment it likes
    what it sees. That is exactly the abuse the guarantee is supposed to
    tolerate.
    """
    rng = np.random.default_rng(3)
    n_paths, horizon = 40000, 12
    crossed = {a: 0 for a in (0.5, 0.2, 0.05)}
    for _ in range(n_paths):
        w = EWealth()
        for _ in range(horizon):
            w.update(float(p_to_e(rng.random())), 0.6)
            if w.wealth > 3.0:  # an adaptive, wealth-dependent stopping rule
                break
        for a in crossed:
            crossed[a] += int(w.has_crossed(1.0 / a))
    for a, k in crossed.items():
        rate = k / n_paths
        assert rate <= a, f"alpha={a}: empirical crossing rate {rate:.4f} exceeds it"


def test_running_max_not_current_wealth_decides_a_verdict():
    w = EWealth()
    w.update(20.0, 1.0)  # wealth shoots up
    high = w.wealth
    w.update(0.0, 1.0)  # and is then wiped out
    assert w.wealth == pytest.approx(0.0)
    assert w.running_max == pytest.approx(high)
    assert w.has_crossed(5.0), "an earlier crossing must remain a valid verdict"


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------


def _calibration_sample(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < 0.4).astype(int)
    usab = rng.beta(4, 3, n)
    # score separates the classes better when the image is more usable
    scores = 1.0 / (1.0 + np.exp(-(rng.normal(0, 1, n) + 2.5 * usab * (2 * labels - 1))))
    return scores, labels, usab


def test_conformal_p_values_are_superuniform_under_the_null():
    scores, labels, usab = _calibration_sample(6000, seed=1)
    cal = StratifiedCalibrator(n_strata=4).fit(scores, labels, usab)
    # fresh null draws from the same law
    s2, l2, u2 = _calibration_sample(6000, seed=2)
    p = np.array([
        cal.p_values(s, cal.stratum_of(u))[0] for s, l, u in zip(s2, l2, u2) if l == 0
    ])
    for alpha in (0.05, 0.1, 0.2, 0.5):
        assert (p <= alpha).mean() <= alpha + 0.03, alpha


def test_p_values_are_bounded_and_never_zero():
    scores, labels, usab = _calibration_sample(2000)
    for cls in (StratifiedCalibrator, MarginalCalibrator):
        cal = cls(n_strata=4).fit(scores, labels, usab)
        for s in (-5.0, 0.0, 0.5, 1.0, 5.0):
            pc, pd = cal.p_values(s, cal.stratum_of(0.5))
            assert 0.0 < pc <= 1.0 and 0.0 < pd <= 1.0


def test_likelihood_ratio_e_values_have_unit_expectation_on_held_out_nulls():
    """The LR route's validity is estimated, not proved -- so check it."""
    scores, labels, usab = _calibration_sample(8000, seed=4)
    cal = LikelihoodRatioCalibrator(n_strata=4).fit(scores, labels, usab)
    s2, l2, u2 = _calibration_sample(8000, seed=5)
    e = np.array([
        cal.e_values(s, cal.stratum_of(u))[0] for s, l, u in zip(s2, l2, u2) if l == 0
    ])
    assert e.mean() < 1.15, f"E[e] under the null is {e.mean():.3f}, should be about 1"


def test_stratum_edges_are_frozen_at_fit_time():
    scores, labels, usab = _calibration_sample(3000)
    cal = StratifiedCalibrator(n_strata=4).fit(scores, labels, usab)
    assert cal.edges is not None and len(cal.edges) == 3
    assert cal.stratum_of(0.0) == 0
    assert cal.stratum_of(1.0) == 3
    assert all(cal.stratum_of(e - 1e-9) <= cal.stratum_of(e + 1e-9) for e in cal.edges)


def test_quantile_edges_split_the_sample_evenly():
    x = np.random.default_rng(0).random(10000)
    edges = quantile_edges(x, 4)
    strata = np.array([usability_stratum(v, 4, edges) for v in x])
    counts = np.bincount(strata, minlength=4) / len(x)
    assert np.allclose(counts, 0.25, atol=0.02)


def test_sparse_strata_fall_back_and_say_so():
    """A stratum with too few points must borrow the marginal pool, visibly."""
    rng = np.random.default_rng(0)
    n = 300
    usab = np.concatenate([rng.uniform(0.0, 0.05, 5), rng.uniform(0.5, 1.0, n - 5)])
    labels = (rng.random(n) < 0.5).astype(int)
    scores = rng.random(n)
    cal = StratifiedCalibrator(n_strata=8).fit(scores, labels, usab)
    assert cal.fallback_strata, "sparse strata should be recorded, not silently used"


# ---------------------------------------------------------------------------
# the measurability firewall -- the paper's central structural claim
# ---------------------------------------------------------------------------


def test_evidence_view_cannot_carry_the_current_diagnosis_score():
    view = EvidenceView(
        shot_index=0, degradation={"blur": 0.3}, predicted_usability=0.7,
        stratum=2, wealth_convict=1.0, wealth_discharge=1.0,
    )
    assert not hasattr(view, "current_score")
    peeking = PeekingView(
        shot_index=0, degradation={"blur": 0.3}, predicted_usability=0.7,
        stratum=2, wealth_convict=1.0, wealth_discharge=1.0,
    )
    assert np.isnan(peeking.current_score), "peeking must default to a loud NaN, not 0.0"


def test_stake_is_independent_of_the_diagnosis_score():
    """Two shots identical except for the score must be staked identically.

    This is the measurability condition made executable: if a future change
    let the stake see the score, this test fails and the validity guarantee
    would have been quietly voided.
    """
    scores, labels, usab = _calibration_sample(4000)
    cal = StratifiedCalibrator(n_strata=4).fit(scores, labels, usab)
    degradation = {"blur": 0.4, "glare": 0.2, "angle": 0.1, "low_light": 0.3, "jpeg": 0.05}

    stakes = []
    for score in (0.01, 0.5, 0.99):
        machine = VerdictMachine(calibrator=cal, burden=BurdenSpec())
        verdict = machine.observe(score, degradation, 0.62, captures_remaining=2)
        stakes.append((verdict.stake_convict, verdict.stake_discharge))
    assert len(set(stakes)) == 1, f"stake varied with the diagnosis score: {stakes}"


def test_subpoena_names_the_dominant_predicted_artifact():
    scores, labels, usab = _calibration_sample(4000)
    cal = StratifiedCalibrator(n_strata=4).fit(scores, labels, usab)
    machine = VerdictMachine(calibrator=cal, burden=BurdenSpec())
    v = machine.observe(
        0.5, {"blur": 0.05, "glare": 0.9, "angle": 0.1, "low_light": 0.1, "jpeg": 0.0},
        0.3, captures_remaining=2,
    )
    assert v.outcome is VerdictOutcome.PENDING
    assert v.instruction is not None and v.instruction.target == "glare"


def test_compression_produces_no_actionable_instruction():
    """jpeg is a channel property; no retake can fix it, so no targeted ask."""
    scores, labels, usab = _calibration_sample(4000)
    cal = StratifiedCalibrator(n_strata=4).fit(scores, labels, usab)
    machine = VerdictMachine(calibrator=cal, burden=BurdenSpec())
    v = machine.observe(
        0.5, {"blur": 0.0, "glare": 0.0, "angle": 0.0, "low_light": 0.0, "jpeg": 0.9},
        0.3, captures_remaining=2,
    )
    assert v.instruction is not None and v.instruction.target is None


def test_contradictory_evidence_escalates_rather_than_picking_a_side():
    class BothWaysCalibrator(StratifiedCalibrator):
        def e_values(self, score, stratum):
            return 1000.0, 1000.0  # both burdens smashed at once

    scores, labels, usab = _calibration_sample(2000)
    cal = BothWaysCalibrator(n_strata=4).fit(scores, labels, usab)
    machine = VerdictMachine(calibrator=cal, burden=BurdenSpec())
    v = machine.observe(0.5, {"blur": 0.1}, 0.8, captures_remaining=3)
    assert v.outcome is VerdictOutcome.REFER


def test_budget_exhaustion_escalates_rather_than_guessing():
    scores, labels, usab = _calibration_sample(4000)
    cal = StratifiedCalibrator(n_strata=4).fit(scores, labels, usab)
    machine = VerdictMachine(calibrator=cal, burden=BurdenSpec())
    v = machine.observe(0.5, {"blur": 0.5}, 0.5, captures_remaining=0)
    assert v.outcome is VerdictOutcome.REFER


# ---------------------------------------------------------------------------
# the ladder
# ---------------------------------------------------------------------------


def test_thresholds_are_one_over_alpha():
    for s in (PREPONDERANCE,):
        assert s.threshold == pytest.approx(1.0 / s.alpha)
    assert standard_by_name("beyond_reasonable_doubt").threshold == pytest.approx(20.0)
    with pytest.raises(KeyError):
        standard_by_name("vibes")


def test_default_burden_demands_more_to_discharge_than_to_convict():
    """The medical asymmetry: sending someone home untreated is the costly error."""
    b = BurdenSpec()
    assert b.discharge.alpha < b.convict.alpha
    assert b.discharge.threshold > b.convict.threshold


def test_degradation_aware_stake_tracks_image_quality():
    strat = DegradationAwareBet()
    def view(u):
        return EvidenceView(0, {"blur": 1 - u}, u, 0, 1.0, 1.0)
    assert strat.stake(view(0.9), "convict") > strat.stake(view(0.2), "convict")
    assert 0.0 <= strat.stake(view(0.0), "convict") <= 1.0
    assert ConstantBet(0.7).stake(view(0.1), "convict") == pytest.approx(0.7)
