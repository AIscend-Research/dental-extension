"""Runnable checks for the selective-prediction metrics."""

import numpy as np

from src.eval.metrics import (
    risk_coverage_curve,
    area_under_rc,
    safe_deferral_rate,
    accuracy_at_coverage,
)


def test_perfect_confidence_beats_random():
    rng = np.random.default_rng(0)
    n = 1000
    correct = rng.random(n) < 0.8
    good = correct * rng.uniform(0.5, 1, n) + (~correct) * rng.uniform(0, 0.5, n)
    rand = rng.random(n)
    assert area_under_rc(correct, good) > area_under_rc(correct, rand)


def test_full_coverage_equals_base_accuracy():
    correct = np.array([1, 1, 0, 1, 0], dtype=float)
    conf = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    coverage, accuracy = risk_coverage_curve(correct, conf)
    assert coverage[-1] == 1.0
    assert np.isclose(accuracy[-1], correct.mean())


def test_safe_deferral_bounds():
    correct = np.array([1, 1, 1, 0], dtype=float)
    conf = np.array([0.9, 0.8, 0.7, 0.1])
    # deferring the single wrong (least confident) case hits 100% accuracy
    assert safe_deferral_rate(correct, conf, 1.0) == 0.25
    # target met at full coverage -> zero deferral
    assert safe_deferral_rate(correct, conf, 0.5) == 0.0
    # unreachable target -> defer everything
    assert safe_deferral_rate(correct, conf, 1.01) == 1.0


def test_accuracy_at_coverage_monotone_region():
    correct = np.array([1, 1, 0, 0], dtype=float)
    conf = np.array([0.9, 0.8, 0.2, 0.1])
    assert accuracy_at_coverage(correct, conf, 0.5) == 1.0


if __name__ == "__main__":
    for fn in [
        test_perfect_confidence_beats_random,
        test_full_coverage_equals_base_accuracy,
        test_safe_deferral_bounds,
        test_accuracy_at_coverage_monotone_region,
    ]:
        fn()
        print("PASS", fn.__name__)
