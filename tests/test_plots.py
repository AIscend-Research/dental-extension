"""Runnable checks for the risk-coverage plotting utility."""

import os
import tempfile

import numpy as np

from src.eval.plots import (
    plot_risk_coverage,
    plot_risk_coverage_comparison,
    plot_reliability_diagram,
)


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


def test_plot_reliability_diagram_writes_a_file():
    rng = np.random.default_rng(2)
    confidence = rng.uniform(0, 1, 500)
    correct = (rng.random(500) < confidence).astype(float)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "reliability.png")
        result = plot_reliability_diagram(correct, confidence, out)
        assert result == out
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0


if __name__ == "__main__":
    for fn in [
        test_plot_risk_coverage_writes_a_file,
        test_plot_risk_coverage_comparison_handles_multiple_series,
        test_plot_reliability_diagram_writes_a_file,
    ]:
        fn()
        print("PASS", fn.__name__)
