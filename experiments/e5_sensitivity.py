"""E5 -- sensitivity: does the conclusion survive a different world?

Every number in E3 was produced inside one parameterisation of a simulator
whose parameters are, in the end, chosen. The claim that matters is therefore
not "evidential capture scores 0.22" -- it is "evidential capture beats the
alternatives, and the guarantee holds, across the range of worlds this
simulator can plausibly represent".

So this experiment re-runs the sound arms across four axes and reports, for
each world, whether the policy ORDERING is preserved and whether any arm
violated its stated burden. A conclusion that flips somewhere is more useful
than one asserted to be universal, and where the ordering does break, that
boundary is itself the finding.

Axes:
  clinic difficulty  -- district hospital through torch-lit outpost
  prevalence         -- how much disease there is to find
  reader quality     -- the clinic-AUC anchor the surrogate is fitted to
  case difficulty    -- how many lesions are undecidable from any photograph

Run: .venv/bin/python -m experiments.e5_sensitivity
"""

from __future__ import annotations

import numpy as np

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
from src.bench.metrics import score_results
from src.bench.policies import (
    EvidentialCapture,
    FixedRetake,
    SingleShot,
    UntargetedEvidential,
)
from src.bench.runner import run_docket
from src.evidence.calibration import LikelihoodRatioCalibrator

N_CASES = 2500
BUDGET = 4

# The ordering the headline result claims, best first. "Preserved" means each
# arm scores at least as well as the one after it.
EXPECTED_ORDER = [
    "evidential_capture",
    "untargeted_evidential",
    "fixed_retake",
    "single_shot",
]
ARMS = {
    "evidential_capture": EvidentialCapture,
    "untargeted_evidential": UntargetedEvidential,
    "fixed_retake": FixedRetake,
    "single_shot": SingleShot,
}


def evaluate_world(
    clinic_difficulty=CLINIC_DIFFICULTY,
    prevalence=PREVALENCE,
    clinic_auc_target=0.78,
    difficulty_beta=(2.0, 4.0),
    seed=31,
) -> dict:
    """Build one world, run every sound arm, and check the ordering holds."""
    world = build_world(
        clinic_difficulty=clinic_difficulty, clinic_auc_target=clinic_auc_target
    )
    cal = LikelihoodRatioCalibrator(n_strata=world.calibrator.n_strata).fit(
        world.calibration_data.scores,
        world.calibration_data.labels,
        world.calibration_data.usabilities,
    )
    docket = make_docket(
        "sensitivity", n_cases=N_CASES, prevalence=prevalence, budget=BUDGET,
        burden=HEADLINE_BURDEN, clinic_difficulty=clinic_difficulty, seed=seed,
        difficulty_alpha=difficulty_beta[0], difficulty_beta=difficulty_beta[1],
    )
    scores = {}
    violations = []
    for name, cls in ARMS.items():
        r = score_results(name, run_docket(docket, cls(), world.channel, cal), docket.burden)
        scores[name] = r
        if r.convict_violation or r.discharge_violation:
            violations.append(name)

    vpc = {n: scores[n].verdicts_per_capture for n in ARMS}
    ordered = [vpc[n] for n in EXPECTED_ORDER]
    preserved = all(ordered[i] >= ordered[i + 1] for i in range(len(ordered) - 1))
    return {
        "clinic_auc": world.clinic_auc,
        "vpc": vpc,
        "accuracy": {n: scores[n].verdict_accuracy for n in ARMS},
        "refer_rate": {n: scores[n].refer_rate for n in ARMS},
        "order_preserved": preserved,
        "violations": violations,
        "margin_over_untargeted": vpc["evidential_capture"] - vpc["untargeted_evidential"],
    }


def sweep(name: str, values, key: str, fmt="{}") -> list[dict]:
    rows = []
    print(f"\n[{name}]")
    print(
        f"  {'setting':<18} {'evidential':>11} {'untargeted':>11} {'fixed':>9} "
        f"{'single':>9} {'margin':>9}  status"
    )
    for v in values:
        out = evaluate_world(**{key: v})
        status = "ok" if out["order_preserved"] else "ORDER FLIPPED"
        if out["violations"]:
            status += " + VIOLATION:" + ",".join(out["violations"])
        rows.append({
            "axis": name, "setting": fmt.format(v),
            "value": v if not isinstance(v, tuple) else str(v),
            "clinic_auc": out["clinic_auc"],
            **{f"vpc_{k}": val for k, val in out["vpc"].items()},
            **{f"acc_{k}": val for k, val in out["accuracy"].items()},
            "margin_over_untargeted": out["margin_over_untargeted"],
            "order_preserved": out["order_preserved"],
            "violations": ",".join(out["violations"]),
        })
        print(
            f"  {fmt.format(v):<18} {out['vpc']['evidential_capture']:>11.3f} "
            f"{out['vpc']['untargeted_evidential']:>11.3f} {out['vpc']['fixed_retake']:>9.3f} "
            f"{out['vpc']['single_shot']:>9.3f} {out['margin_over_untargeted']:>+9.4f}  {status}"
        )
    return rows


def main() -> None:
    banner("E5 -- sensitivity of the conclusion to the world")
    rows = []
    rows += sweep("clinic difficulty", [0.2, 0.35, 0.5, 0.65, 0.8], "clinic_difficulty", "difficulty {}")
    rows += sweep("prevalence", [0.10, 0.20, 0.35, 0.50], "prevalence", "prevalence {}")
    rows += sweep("reader quality", [0.70, 0.74, 0.78, 0.84], "clinic_auc_target", "clinic AUC {}")
    rows += sweep(
        "case difficulty",
        [(1.5, 6.0), (2.0, 4.0), (3.0, 3.0), (5.0, 2.5)],
        "difficulty_beta",
        "Beta{}",
    )

    n_flipped = sum(1 for r in rows if not r["order_preserved"])
    n_violated = sum(1 for r in rows if r["violations"])
    # The full-ordering check is strict: it fails if ANY adjacent pair swaps,
    # including the two weakest arms trading places, which says nothing about
    # the claim being made. Track the headline claim separately.
    n_top = sum(
        1 for r in rows
        if r["vpc_evidential_capture"] >= max(r[f"vpc_{n}"] for n in EXPECTED_ORDER)
    )
    print(
        f"\nsummary: {len(rows)} worlds | full ordering preserved in {len(rows) - n_flipped} "
        f"| evidential_capture ranked first in {n_top} | guarantee violations in {n_violated}"
    )
    if n_flipped:
        print("  worlds where the full ordering broke (and which pair swapped):")
        for r in rows:
            if r["order_preserved"]:
                continue
            ranked = sorted(EXPECTED_ORDER, key=lambda n: -r[f"vpc_{n}"])
            print(f"    {r['axis']}, {r['setting']}: {' > '.join(ranked)}")
    margins = [r["margin_over_untargeted"] for r in rows]
    print(
        f"margin of targeted over untargeted: min {min(margins):+.4f}, "
        f"median {float(np.median(margins)):+.4f}, max {max(margins):+.4f}"
    )

    fig = make_figure(rows)
    print(f"figure -> {fig}")
    save_table("e5_sensitivity", rows)
    save_results("e5_sensitivity", {
        "n_worlds": len(rows),
        "n_order_flipped": n_flipped,
        "n_evidential_ranked_first": n_top,
        "n_with_violations": n_violated,
        "margin_min": min(margins),
        "margin_median": float(np.median(margins)),
        "margin_max": max(margins),
        "rows": rows,
        "figure": fig,
    })
    print("results -> results/e5_sensitivity.json")


def make_figure(rows) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axes_names = ["clinic difficulty", "prevalence", "reader quality", "case difficulty"]
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.2), sharey=True)
    for ax, axis in zip(axs, axes_names):
        sel = [r for r in rows if r["axis"] == axis]
        x = np.arange(len(sel))
        for name in EXPECTED_ORDER:
            ax.plot(x, [r[f"vpc_{name}"] for r in sel], "o-", label=name, ms=4)
        ax.set_xticks(x, [r["setting"].split()[-1] for r in sel], fontsize=8)
        ax.set_title(axis)
        ax.grid(alpha=0.3)
        for i, r in enumerate(sel):
            if not r["order_preserved"]:
                ax.axvspan(i - 0.4, i + 0.4, color="tab:red", alpha=0.12)
    axs[0].set_ylabel("verdicts per capture")
    axs[0].legend(fontsize=7)
    fig.suptitle(
        "E5: policy ordering across worlds (red band = ordering flipped)", fontsize=12
    )
    fig.tight_layout()
    path = figure_path("e5_sensitivity.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


if __name__ == "__main__":
    main()
