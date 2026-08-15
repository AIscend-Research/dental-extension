"""E11 -- per-case capture budgets under one shared cost, instead of a fixed K.

Every other experiment gives each case the same budget K. This one asks
whether that is wasteful: a case whose first photo is already excellent
spends the same allowance as one that needs every retake. `src.bench.
allocation.run_docket_global_budget` spends the identical TOTAL number of
photographs (N x K, so the comparison is at equal cost, not a bigger
budget) but lets a greedy allocator hand the next shot to whichever pending
case's most recent photo looks most usable -- closest, on that signal, to
being rescued into a verdict.

Two questions:

  1. Does reallocating help -- more verdicts per capture at the same total
     spend, or does the uniform K already spend about as well as a simple
     greedy scheme can?
  2. Does the guarantee survive interleaving cases through a shared
     scheduler? The allocator's priority signal is predicted usability
     (degradation-channel-only), never a diagnosis score, and cases are
     independent processes, so `src/bench/allocation.py`'s docstring argues
     validity should be unaffected -- checked here the way E2 checks the
     single-case guarantee, on an all-sound null docket.

Run: .venv/bin/python -m experiments.e11_capture_budgets
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
from src.bench.allocation import run_docket_global_budget
from src.bench.docket import make_docket
from src.bench.metrics import format_leaderboard, score_results
from src.bench.policies import EvidentialCapture
from src.bench.runner import run_docket
from src.evidence.calibration import LikelihoodRatioCalibrator
from src.evidence.ladder import BurdenSpec, PREPONDERANCE

N_CASES = 4000
BUDGET = 4
NULL_N_CASES = 20000


def main() -> None:
    world = build_world()
    banner("E11 -- per-case capture budgets under a shared cost", world)

    lr = LikelihoodRatioCalibrator(n_strata=world.calibrator.n_strata).fit(
        world.calibration_data.scores, world.calibration_data.labels, world.calibration_data.usabilities
    )

    docket = make_docket(
        "capture_budgets", n_cases=N_CASES, prevalence=PREVALENCE, budget=BUDGET,
        burden=HEADLINE_BURDEN, clinic_difficulty=CLINIC_DIFFICULTY, seed=51,
    )
    total_budget = len(docket) * docket.budget

    print(f"\n[1] equal total cost: {total_budget} photographs across {N_CASES} cases either way")
    uniform = score_results(
        "evidential_capture (uniform K=4)",
        run_docket(docket, EvidentialCapture(), world.channel, lr),
        docket.burden,
    )
    allocated = score_results(
        "evidential_capture (shared budget)",
        run_docket_global_budget(docket, world.channel, lr, total_budget=total_budget),
        docket.burden,
    )
    rows = [uniform, allocated]
    print(format_leaderboard(rows))
    gain = allocated.verdicts_per_capture - uniform.verdicts_per_capture
    print(
        f"\nreallocation gain: {gain:+.4f} VPC "
        f"({'reallocating helps' if gain > 0 else 'uniform K already spends about as well'})"
    )

    # -- 2. validity check on an all-sound null docket, mirroring E2 --------
    print(f"\n[2] validity under a shared scheduler: {NULL_N_CASES} genuinely sound cases, K={BUDGET}")
    symmetric_burden = BurdenSpec.symmetric(PREPONDERANCE)
    null_docket = make_docket(
        "null_shared_budget", n_cases=NULL_N_CASES, prevalence=0.0, budget=BUDGET,
        burden=symmetric_burden, clinic_difficulty=CLINIC_DIFFICULTY, seed=52,
    )
    null_total_budget = len(null_docket) * null_docket.budget
    null_results = run_docket_global_budget(null_docket, world.channel, lr, total_budget=null_total_budget)
    null_row = score_results("evidential_capture (shared budget, null)", null_results, symmetric_burden)
    print(format_leaderboard([null_row]))
    alpha = symmetric_burden.convict.alpha
    verdict = (
        "GUARANTEE HOLDS under the shared scheduler"
        if not null_row.convict_violation
        else "VIOLATED -- cross-case scheduling broke the guarantee, contradicting the independence argument"
    )
    print(f"\nfalse-conviction rate {null_row.false_conviction_rate:.4f} vs alpha {alpha:.2f}: {verdict}")

    save_table("e11_capture_budgets", [r.as_dict() for r in rows] + [null_row.as_dict()])
    save_results("e11_capture_budgets", {
        "reader": {"clean_auc": world.clean_auc, "clinic_auc": world.clinic_auc},
        "equal_cost_comparison": {
            "n_cases": N_CASES, "total_budget": total_budget,
            "uniform": uniform.as_dict(), "shared_budget": allocated.as_dict(),
            "reallocation_gain_vpc": gain,
        },
        "validity_check": {
            "n_cases": NULL_N_CASES, "alpha": alpha,
            "false_conviction_rate": null_row.false_conviction_rate,
            "false_conviction_ci": null_row.false_conviction_ci,
            "violated": null_row.convict_violation,
            "verdict": verdict,
        },
    })
    print("results -> results/e11_capture_budgets.json")


if __name__ == "__main__":
    main()
