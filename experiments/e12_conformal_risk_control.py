"""E12 -- a conformal-risk-control-style baseline on the leaderboard.

The roadmap calls for adding "SelectiveNet and conformal risk control as
leaderboard entries... a benchmark carrying only the authors' own methods
isn't yet a benchmark." `confidence_threshold_selective` (E3) covers the
first. This covers the second, with an honest scoping note: conformal risk
control proper [Angelopoulos, Bates, Fisch, Lei, Schuster 2022] is a specific
algorithm for calibrating a threshold to control an arbitrary bounded loss
via a finite-sample monotonicity argument, generalizing split conformal
prediction beyond miscoverage. Reimplementing that algorithm faithfully is
out of scope here; what this experiment adds instead is the closest existing
building block in this codebase to what that literature represents as a
*family*: a single-look, distribution-free-calibrated decision with no
retake and no per-artifact stratification -- i.e. `single_shot` under
`MarginalCalibrator` rather than `StratifiedCalibrator`. This is the
"plain split-conformal, one look, ignore image quality" baseline the theory
note's own §6 already discusses as the default anyone would build; putting
it on the leaderboard as its own labeled row, rather than only inside E4's
ablation table, is what was actually missing.

Run: .venv/bin/python -m experiments.e12_conformal_risk_control
"""

from __future__ import annotations

from experiments.common import (
    CLINIC_DIFFICULTY,
    HEADLINE_BURDEN,
    PREVALENCE,
    banner,
    build_world,
    figure_path,
    save_results,
    save_table,
)
from src.bench.docket import make_docket
from src.bench.metrics import format_leaderboard, score_results
from src.bench.policies import EvidentialCapture, SingleShot
from src.bench.runner import fit_calibrator, run_docket
from src.evidence.calibration import MarginalCalibrator, StratifiedCalibrator

N_CASES = 4000
BUDGET = 4


def main() -> None:
    world = build_world()
    banner("E12 -- a conformal-risk-control-style baseline", world)

    docket = make_docket(
        "conformal_risk_control", n_cases=N_CASES, prevalence=PREVALENCE, budget=BUDGET,
        burden=HEADLINE_BURDEN, clinic_difficulty=CLINIC_DIFFICULTY, seed=61,
    )

    marginal, _ = fit_calibrator(
        world.channel, n_strata=world.calibrator.n_strata, calibrator_cls=MarginalCalibrator,
        n=8000, prevalence=PREVALENCE, clinic_difficulty=CLINIC_DIFFICULTY, seed=62,
    )

    rows = [
        score_results(
            "conformal_risk_control_style (single-shot, marginal)",
            run_docket(docket, SingleShot(), world.channel, marginal),
            docket.burden,
        ),
        score_results(
            "single_shot (stratified)",
            run_docket(docket, SingleShot(), world.channel, world.calibrator),
            docket.burden,
        ),
        score_results(
            "evidential_capture (stratified, retake)",
            run_docket(docket, EvidentialCapture(), world.channel, world.calibrator),
            docket.burden,
        ),
    ]
    print(format_leaderboard(rows))

    by_name = {r.policy: r for r in rows}
    stratification_gain = (
        by_name["single_shot (stratified)"].verdicts_per_capture
        - by_name["conformal_risk_control_style (single-shot, marginal)"].verdicts_per_capture
    )
    retake_gain = (
        by_name["evidential_capture (stratified, retake)"].verdicts_per_capture
        - by_name["single_shot (stratified)"].verdicts_per_capture
    )
    print(
        f"\nstratification alone (single-shot, marginal -> stratified): {stratification_gain:+.4f} VPC"
        f"\nadding the retake loop on top (single-shot -> evidential_capture): {retake_gain:+.4f} VPC"
    )

    fig = make_figure(rows, stratification_gain, retake_gain)
    print(f"\nfigure -> {fig}")

    save_table("e12_conformal_risk_control", [r.as_dict() for r in rows])
    save_results("e12_conformal_risk_control", {
        "reader": {"clean_auc": world.clean_auc, "clinic_auc": world.clinic_auc},
        "docket": {"n_cases": N_CASES, "budget": BUDGET, "burden": HEADLINE_BURDEN.name},
        "rows": [r.as_dict() for r in rows],
        "stratification_gain_vpc": stratification_gain,
        "retake_gain_vpc": retake_gain,
        "scoping_note": (
            "Not a reimplementation of Angelopoulos et al. 2022's conformal risk "
            "control algorithm -- see module docstring. This is the single-shot, "
            "unstratified conformal baseline that family of methods represents here."
        ),
        "figure": fig,
    })
    print("results -> results/e12_conformal_risk_control.json")


def make_figure(rows, stratification_gain, retake_gain) -> str:
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.gridspec import GridSpec

    from experiments.common import sample_tooth_crops
    from src.sim.render import render_severities

    fig = plt.figure(figsize=(11.5, 6.6))
    gs = GridSpec(2, 5, figure=fig, height_ratios=[1.0, 2.1], hspace=0.6, wspace=0.15)

    crop = sample_tooth_crops(1, label=0, seed=33)[0]
    rng = np.random.default_rng(8)
    single_out = render_severities(crop.image, {"blur": 0.4, "glare": 0.3}, rng=rng)
    ax_single = fig.add_subplot(gs[0, 0])
    ax_single.imshow(cv2.cvtColor(single_out.image, cv2.COLOR_BGR2RGB))
    ax_single.set_xticks([]); ax_single.set_yticks([])
    for spine in ax_single.spines.values():
        spine.set_edgecolor("tab:grey"); spine.set_linewidth(2.0)
    ax_single.set_title("single_shot:\none look, decide now", fontsize=8.5, color="tab:grey")

    for i in range(4):
        rng = np.random.default_rng(40 + i)
        severity = max(0.55 - 0.15 * i, 0.1)
        out = render_severities(crop.image, {"blur": severity, "glare": max(severity - 0.1, 0.0)}, rng=rng)
        ax = fig.add_subplot(gs[0, i + 1])
        ax.imshow(cv2.cvtColor(out.image, cv2.COLOR_BGR2RGB))
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("tab:blue"); spine.set_linewidth(1.6)
        ax.set_title(f"retake\nshot {i + 1}", fontsize=8.5, color="tab:blue")
    fig.text(0.5, 0.945, "the same real tooth: one look vs evidential_capture's retake loop, quality improving shot by shot until the burden is met",
              ha="center", fontsize=8.5, color="#6b6b6b")

    ax = fig.add_subplot(gs[1, 0:2])
    labels = ["conformal-risk-control\nstyle (marginal)", "single_shot\n(stratified)", "evidential_capture\n(stratified, retake)"]
    vpc = [r.verdicts_per_capture for r in rows]
    bars = ax.bar(labels, vpc, color=["tab:grey", "tab:orange", "tab:blue"])
    for bar, v in zip(bars, vpc):
        ax.annotate(f"{v:.3f}", (bar.get_x() + bar.get_width() / 2, v), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("verdicts per capture")
    ax.set_title("(a) the three arms")
    ax.grid(alpha=0.3, axis="y")

    ax = fig.add_subplot(gs[1, 2:5])
    base = rows[0].verdicts_per_capture
    strat = base + stratification_gain
    final = strat + retake_gain
    ax.bar(["marginal\n(baseline)"], [base], color="tab:grey")
    ax.bar(["+ stratification"], [stratification_gain], bottom=[base], color="tab:orange")
    ax.bar(["+ retake loop"], [retake_gain], bottom=[strat], color="tab:blue")
    ax.annotate(f"{stratification_gain:+.4f}", (1, base + stratification_gain / 2), ha="center", fontsize=8)
    ax.annotate(f"{retake_gain:+.4f}", (2, strat + retake_gain / 2), ha="center", fontsize=8)
    ax.set_ylabel("verdicts per capture")
    ax.set_title("(b) where the gain comes from:\nstratification alone buys ~nothing, retaking does")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("E12: a conformal-risk-control-style baseline on the leaderboard", fontsize=12, y=1.0)
    path = figure_path("e12_conformal_risk_control.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return str(path)


if __name__ == "__main__":
    main()
