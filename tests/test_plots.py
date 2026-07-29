"""Runnable checks for the risk-coverage plotting utility."""

import os
import tempfile

import numpy as np

from src.eval.plots import plot_risk_coverage, plot_risk_coverage_comparison


def test_plot_risk_coverage_writes_a_file():
    rng = np.random.default_rng(0)
    correct = rng.random(200) < 0.8
    conf = rng.random(200)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "rc.png")
        result = plot_risk_coverage(correct, conf, out)
        assert result == out
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0


def test_plot_risk_coverage_comparison_handles_multiple_series():
    rng = np.random.default_rng(1)
    correct = rng.random(200) < 0.8
    series = {
        "arm_a": (correct, rng.random(200)),
        "arm_b": (correct, rng.random(200)),
    }
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "rc_cmp.png")
        plot_risk_coverage_comparison(series, out)
        assert os.path.exists(out)


if __name__ == "__main__":
    for fn in [
        test_plot_risk_coverage_writes_a_file,
        test_plot_risk_coverage_comparison_handles_multiple_series,
    ]:
        fn()
        print("PASS", fn.__name__)
