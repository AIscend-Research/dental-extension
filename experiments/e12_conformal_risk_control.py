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
    })
    print("results -> results/e12_conformal_risk_control.json")


if __name__ == "__main__":
    main()
