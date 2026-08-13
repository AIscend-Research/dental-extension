"""Tests for the benchmark (src/bench/) and the surrogate reader."""

from __future__ import annotations

import numpy as np
import pytest

from src.bench.docket import make_docket, make_image_docket
from src.bench.metrics import score_results, wilson_interval
from src.bench.policies import POLICIES, PolicyContext, SessionResult, policy_by_name
from src.bench.runner import fit_calibrator, run_docket
from src.evidence.ladder import BurdenSpec
from src.evidence.verdict import VerdictOutcome
from src.models.diagnostic import (
    Case,
    SurrogateChannel,
    _auc,
    calibrate_loss_scale,
    calibrate_separation,
    predicted_usability,
)


@pytest.fixture(scope="module")
def world():
    channel = SurrogateChannel()
    calibrate_separation(channel, 0.88, n=3000, seed=0)
    calibrate_loss_scale(channel, 0.78, n=1500, seed=1)
    calibrator, data = fit_calibrator(channel, n=4000, seed=7)
    return channel, calibrator


# ---------------------------------------------------------------------------
# the surrogate reader
# ---------------------------------------------------------------------------


def test_calibrate_separation_hits_its_clean_auc_target():
    channel = SurrogateChannel()
    calibrate_separation(channel, target_auc=0.88, n=6000, seed=0)
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, 8000)
    scores = [channel.read(Case(int(y), 0.3), {}, rng).score for y in labels]
    assert _auc(labels, np.array(scores)) == pytest.approx(0.88, abs=0.02)


def test_quality_is_monotone_decreasing_in_severity():
    channel = SurrogateChannel()
    prev = channel.quality({})
    assert prev == pytest.approx(1.0)
    for sev in (0.2, 0.5, 0.8, 1.0):
        q = channel.quality({"blur": sev})
        assert 0.0 <= q <= prev
        prev = q


def test_degradation_hurts_the_reader():
    channel = SurrogateChannel()
    calibrate_separation(channel, 0.88, n=3000, seed=0)
    rng = np.random.default_rng(2)
    labels = rng.integers(0, 2, 6000)
    clean = np.array([channel.read(Case(int(y), 0.3), {}, rng).score for y in labels])
    dirty = np.array([
        channel.read(Case(int(y), 0.3), {"blur": 0.8, "glare": 0.6}, rng).score for y in labels
    ])
    assert _auc(labels, clean) > _auc(labels, dirty) + 0.05


def test_predicted_usability_is_dominated_by_the_worst_artifact():
    one_bad = predicted_usability({"blur": 0.9, "glare": 0.0, "angle": 0.0})
    all_mild = predicted_usability({"blur": 0.3, "glare": 0.3, "angle": 0.3})
    assert one_bad < all_mild
    assert 0.0 <= one_bad <= 1.0


def test_oracle_head_reports_the_true_severities():
    channel = SurrogateChannel(head_noise=0.0)
    rng = np.random.default_rng(0)
    sev = {"blur": 0.4, "glare": 0.7}
    reading = channel.read(Case(1, 0.2), sev, rng)
    assert reading.degradation["blur"] == pytest.approx(0.4)
    assert reading.degradation["glare"] == pytest.approx(0.7)
    assert reading.degradation["jpeg"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# dockets
# ---------------------------------------------------------------------------


def test_docket_is_reproducible_from_its_seed():
    a = make_docket("x", n_cases=200, seed=4)
    b = make_docket("x", n_cases=200, seed=4)
    assert [c.label for c in a.cases] == [c.label for c in b.cases]
    assert [c.seed for c in a.cases] == [c.seed for c in b.cases]
    c = make_docket("x", n_cases=200, seed=5)
    assert [x.seed for x in a.cases] != [x.seed for x in c.cases]


def test_docket_prevalence_is_roughly_as_requested():
    d = make_docket("x", n_cases=8000, prevalence=0.35, seed=0)
    assert d.prevalence == pytest.approx(0.35, abs=0.02)


def test_image_docket_labels_follow_the_pooled_image():
    pool_labels = [0, 1, 1, 0, 1]
    d = make_image_docket("img", pool_labels, n_cases=500, seed=1)
    for case in d.cases:
        assert case.label == pool_labels[case.image_index]
        assert case.difficulty == 0.0


# ---------------------------------------------------------------------------
# policies and scoring
# ---------------------------------------------------------------------------


def test_every_registered_policy_runs_and_respects_its_budget(world):
    channel, calibrator = world
    d = make_docket("smoke", n_cases=120, budget=3, seed=9)
    for name in POLICIES:
        results = run_docket(d, policy_by_name(name), channel, calibrator)
        assert len(results) == len(d)
        for r in results:
            assert 1 <= r.n_captures <= d.budget
            assert r.outcome in set(VerdictOutcome)
            if r.outcome is VerdictOutcome.REFER:
                assert r.correct is None
            else:
                assert isinstance(r.correct, bool)


def test_single_shot_takes_exactly_one_photograph(world):
    channel, calibrator = world
    d = make_docket("smoke", n_cases=100, budget=5, seed=9)
    results = run_docket(d, policy_by_name("single_shot"), channel, calibrator)
    assert {r.n_captures for r in results} == {1}


def test_fixed_retake_always_spends_the_whole_budget(world):
    channel, calibrator = world
    d = make_docket("smoke", n_cases=100, budget=4, seed=9)
    results = run_docket(d, policy_by_name("fixed_retake"), channel, calibrator)
    assert {r.n_captures for r in results} == {4}


def test_unsound_arms_are_flagged_as_having_peeked(world):
    channel, calibrator = world
    d = make_docket("smoke", n_cases=100, budget=3, seed=9)
    for name in ("greedy_diagnostic", "naive_best_shot"):
        results = run_docket(d, policy_by_name(name), channel, calibrator)
        assert all(r.peeked for r in results)
        assert not score_results(name, results, d.burden).guaranteed
    for name in ("single_shot", "evidential_capture", "fixed_retake"):
        results = run_docket(d, policy_by_name(name), channel, calibrator)
        assert not any(r.peeked for r in results)
        assert score_results(name, results, d.burden).guaranteed


def test_two_policies_see_the_same_opening_shot(world):
    """Arms must diverge only once they issue different instructions."""
    channel, calibrator = world
    d = make_docket("smoke", n_cases=60, budget=3, seed=9)
    a = run_docket(d, policy_by_name("single_shot"), channel, calibrator)
    b = run_docket(d, policy_by_name("evidential_capture"), channel, calibrator)
    for ra, rb in zip(a, b):
        assert ra.score_trace[0] == pytest.approx(rb.score_trace[0])


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def _result(label, outcome, n_captures=1, peeked=False):
    correct = None
    if outcome is VerdictOutcome.CARIES:
        correct = label == 1
    elif outcome is VerdictOutcome.SOUND:
        correct = label == 0
    return SessionResult(
        case_id=0, label=label, outcome=outcome, n_captures=n_captures,
        correct=correct, peeked=peeked,
    )


def test_verdicts_per_capture_counts_only_rendered_verdicts():
    results = [
        _result(1, VerdictOutcome.CARIES, 2),
        _result(0, VerdictOutcome.SOUND, 2),
        _result(1, VerdictOutcome.REFER, 4),
    ]
    row = score_results("t", results, BurdenSpec())
    assert row.n_captures == 8
    assert row.verdicts_per_capture == pytest.approx(2 / 8)
    assert row.verdict_rate == pytest.approx(2 / 3)
    assert row.verdict_accuracy == pytest.approx(1.0)
    assert row.refer_rate == pytest.approx(1 / 3)


def test_violation_needs_the_interval_lower_bound_above_alpha():
    """Noise must not be able to manufacture a violation."""
    burden = BurdenSpec()
    # 3 false convictions out of 10 sound cases: high rate, tiny sample
    few = [_result(0, VerdictOutcome.CARIES) for _ in range(3)]
    few += [_result(0, VerdictOutcome.SOUND) for _ in range(7)]
    assert not score_results("t", few, burden).convict_violation

    # the same rate on a large sample is a real violation
    many = [_result(0, VerdictOutcome.CARIES) for _ in range(300)]
    many += [_result(0, VerdictOutcome.SOUND) for _ in range(700)]
    assert score_results("t", many, burden).convict_violation


def test_wilson_interval_brackets_the_estimate_and_handles_zero():
    lo, hi = wilson_interval(0, 100)
    assert lo == 0.0 and 0.0 < hi < 0.06
    lo, hi = wilson_interval(50, 100)
    assert lo < 0.5 < hi
    assert all(np.isnan(v) for v in wilson_interval(0, 0))


def test_scoring_empty_results_is_an_error_not_a_nan_row():
    with pytest.raises(ValueError, match="no results"):
        score_results("t", [], BurdenSpec())
