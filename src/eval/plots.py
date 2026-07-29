"""
Plotting for the selective-prediction metrics (Phase 4).

TASKS.md calls for "plotting the risk-coverage curve" as part of Stream 3's
analysis work, buildable before the detector exists by validating on
simulated (correct, confidence) arrays -- see src/eval/metrics.py's own
__main__ block for exactly that kind of synthetic check.

Uses matplotlib's non-interactive Agg backend so this is safe to call from a
script or test with no display attached.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.eval.metrics import risk_coverage_curve


def plot_risk_coverage(
    correct: np.ndarray,
    confidence: np.ndarray,
    out_path: str,
    label: str = "model",
    title: str | None = None,
) -> str:
    """Save a risk-coverage (accuracy vs. coverage) plot to out_path.

    A single curve. For comparing several models/ablations on one axis, call
    plot_risk_coverage_comparison instead.
    """
    return plot_risk_coverage_comparison({label: (correct, confidence)}, out_path, title=title)


def plot_risk_coverage_comparison(
    series: dict[str, tuple[np.ndarray, np.ndarray]],
    out_path: str,
    title: str | None = None,
) -> str:
    """Overlay risk-coverage curves for several (correct, confidence) arrays.

    Args:
        series: {label: (correct, confidence)} -- one entry per model/ablation
            arm, e.g. {"baseline": (...), "robustness-trained": (...),
            "full system": (...)}.
        out_path: where to write the PNG.
        title: plot title; defaults to a generic one.

    Returns:
        out_path, for convenience chaining.
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for label, (correct, confidence) in series.items():
        coverage, accuracy = risk_coverage_curve(correct, confidence)
        if len(coverage) == 0:
            continue
        ax.plot(coverage, accuracy, label=label)

    ax.set_xlabel("coverage (fraction of images kept)")
    ax.set_ylabel("accuracy on kept images")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 1.0)
    ax.set_title(title or "Risk-coverage curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    # Simulated sanity check -- NOT real model output. Mirrors
    # src/eval/metrics.py's __main__ block so the plot can be eyeballed
    # before any detector predictions exist.
    rng = np.random.default_rng(0)
    n = 2000
    correct = rng.random(n) < 0.8
    good_conf = correct * rng.uniform(0.5, 1.0, n) + (~correct) * rng.uniform(0, 0.5, n)
    rand_conf = rng.random(n)

    out = plot_risk_coverage_comparison(
        {"good confidence (simulated)": (correct, good_conf),
         "random confidence (simulated)": (correct, rand_conf)},
        "figures/example_risk_coverage.png",
        title="Risk-coverage curve -- SIMULATED data, not a trained model",
    )
    print("wrote", out)
