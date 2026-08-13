"""E3 -- The Docket leaderboard: justified verdicts per photograph.

The benchmark proper. Every policy is run over the same frozen docket, with
the same starting scene and operator per case, so the arms diverge only once
they start issuing different instructions.

Three sweeps:

  A. the headline table at one budget and one burden, under both evidence
     constructions (conformal-calibrated, provably valid; and likelihood-ratio,
     empirically valid and more powerful);
  B. capture budget K -- does a bigger budget buy verdicts, and for whom;
  C. the standards ladder -- where the burden becomes unmeetable from a phone.

Sweep C is the one that should be read first by anyone deciding whether to
deploy this. It locates the point at which a photographic screening system
*cannot* meet a clinical burden of proof no matter how many photographs it
takes, and that boundary is a property of the reader and the capture process,
not of the policy. A benchmark that only reported the regime where methods
differ would hide it.

Run: .venv/bin/python -m experiments.e3_leaderboard
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
from src.bench.metrics import format_leaderboard, score_results
from src.bench.policies import policy_by_name
from src.bench.runner import run_docket
from src.evidence.calibration import LikelihoodRatioCalibrator
from src.evidence.ladder import (
    BEYOND_REASONABLE_DOUBT,
    BurdenSpec,
    CLEAR_AND_CONVINCING,
    NEAR_CERTAINTY,
    PREPONDERANCE,
)

N_CASES = 4000
HEADLINE_BUDGET = 4
BUDGET_GRID = [1, 2, 3, 4, 6, 8]
ARMS = [
    "single_shot",
    "fixed_retake",
    "untargeted_evidential",
    "evidential_capture",
    "oracle_instruction",
    "greedy_diagnostic",
    "naive_best_shot",
]
LADDER = [
    ("preponderance", BurdenSpec.symmetric(PREPONDERANCE)),
    ("headline (prep / clear+conv)", HEADLINE_BURDEN),
    ("clear and convincing", BurdenSpec.symmetric(CLEAR_AND_CONVINCING)),
    ("clear+conv / beyond doubt", BurdenSpec(CLEAR_AND_CONVINCING, BEYOND_REASONABLE_DOUBT)),
    ("beyond reasonable doubt", BurdenSpec.symmetric(BEYOND_REASONABLE_DOUBT)),
    ("near certainty", BurdenSpec.symmetric(NEAR_CERTAINTY)),
]


def run_arms(world, calibrator, docket, arms=ARMS) -> list:
    return [
        score_results(name, run_docket(docket, policy_by_name(name), world.channel, calibrator), docket.burden)
        for name in arms
    ]


def main() -> None:
    world = build_world()
    banner("E3 -- The Docket leaderboard", world)

    lr_calibrator = LikelihoodRatioCalibrator(n_strata=world.calibrator.n_strata).fit(
        world.calibration_data.scores,
        world.calibration_data.labels,
        world.calibration_data.usabilities,
    )
    constructions = {
        "conformal (provable)": world.calibrator,
        "likelihood-ratio (estimated)": lr_calibrator,
    }

    # -- A: headline table --------------------------------------------------
    docket = make_docket(
        "headline",
        n_cases=N_CASES,
        prevalence=PREVALENCE,
        budget=HEADLINE_BUDGET,
        burden=HEADLINE_BURDEN,
        clinic_difficulty=CLINIC_DIFFICULTY,
        seed=11,
    )
    print(
        f"\n[A] headline docket: n={len(docket)}, K={docket.budget}, "
        f"prevalence={docket.prevalence:.3f}, burden={docket.burden.name}"
    )
    headline = {}
    for label, cal in constructions.items():
        rows = run_arms(world, cal, docket)
        headline[label] = rows
        print(f"\n  evidence construction: {label}")
        print(format_leaderboard(rows))
        save_table(f"e3_headline_{label.split()[0]}", [r.as_dict() for r in rows])

    # -- B: budget sweep ----------------------------------------------------
    print(f"\n[B] verdicts per capture vs budget K (burden={HEADLINE_BURDEN.name})")
    budget_rows = []
    for K in BUDGET_GRID:
        d = make_docket(
            "budget", n_cases=N_CASES // 2, prevalence=PREVALENCE, budget=K,
            burden=HEADLINE_BURDEN, clinic_difficulty=CLINIC_DIFFICULTY, seed=12,
        )
        for name in ARMS:
            r = score_results(name, run_docket(d, policy_by_name(name), world.channel, lr_calibrator), d.burden)
            budget_rows.append({
                "budget": K, "policy": name,
                "verdicts_per_capture": r.verdicts_per_capture,
                "verdict_rate": r.verdict_rate,
                "verdict_accuracy": r.verdict_accuracy,
                "mean_captures": r.mean_captures,
                "guaranteed": r.guaranteed,
                "violated": r.convict_violation or r.discharge_violation,
            })
        cells = " ".join(
            f"{row['policy'].split('_')[0]}={row['verdicts_per_capture']:.3f}"
            for row in budget_rows if row["budget"] == K
        )
        print(f"  K={K}: {cells}")

    # -- C: the standards ladder -------------------------------------------
    print("\n[C] where the burden becomes unmeetable (evidential_capture, LR evidence)")
    ladder_rows = []
    for label, burden in LADDER:
        d = make_docket(
            "ladder", n_cases=N_CASES // 2, prevalence=PREVALENCE, budget=HEADLINE_BUDGET,
            burden=burden, clinic_difficulty=CLINIC_DIFFICULTY, seed=13,
        )
        r = score_results(
            "evidential_capture",
            run_docket(d, policy_by_name("evidential_capture"), world.channel, lr_calibrator),
            burden,
        )
        ladder_rows.append({
            "standard": label,
            "alpha_convict": burden.convict.alpha,
            "alpha_discharge": burden.discharge.alpha,
            "verdicts_per_capture": r.verdicts_per_capture,
            "verdict_rate": r.verdict_rate,
            "verdict_accuracy": r.verdict_accuracy,
            "refer_rate": r.refer_rate,
            "false_conviction_rate": r.false_conviction_rate,
            "false_discharge_rate": r.false_discharge_rate,
            "violated": r.convict_violation or r.discharge_violation,
        })
        print(
            f"  {label:<30} VPC {r.verdicts_per_capture:.3f} | decided {r.verdict_rate*100:5.1f}% "
            f"| acc {r.verdict_accuracy:.3f} | escalated {r.refer_rate*100:5.1f}%"
        )

    fig = make_figure(headline, budget_rows, ladder_rows)
    print(f"\nfigure -> {fig}")

    save_table("e3_budget_sweep", budget_rows)
    save_table("e3_ladder", ladder_rows)
    save_results(
        "e3_leaderboard",
        {
            "reader": {"clean_auc": world.clean_auc, "clinic_auc": world.clinic_auc},
            "docket": {
                "n_cases": len(docket), "budget": docket.budget,
                "prevalence": docket.prevalence, "burden": docket.burden.name,
                "clinic_difficulty": CLINIC_DIFFICULTY,
            },
            "headline": {k: [r.as_dict() for r in v] for k, v in headline.items()},
            "budget_sweep": budget_rows,
            "ladder": ladder_rows,
            "figure": fig,
        },
    )
    print("results -> results/e3_leaderboard.json")


def make_figure(headline, budget_rows, ladder_rows) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # (a) headline: VPC vs accuracy, sound arms filled, unsound hollow
    ax = axes[0]
    rows = headline["likelihood-ratio (estimated)"]
    for r in rows:
        ok = r.guaranteed and not (r.convict_violation or r.discharge_violation)
        ax.scatter(
            r.verdicts_per_capture, r.verdict_accuracy, s=90,
            facecolors="tab:blue" if ok else "none",
            edgecolors="tab:blue" if ok else "tab:red", linewidths=1.8,
            marker="o" if ok else "X",
        )
        ax.annotate(r.policy.replace("_", "\n"), (r.verdicts_per_capture, r.verdict_accuracy),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("verdicts per capture")
    ax.set_ylabel("accuracy on rendered verdicts")
    ax.set_title("(a) the headline trade-off\n(hollow red X = guarantee forfeited)")
    ax.grid(alpha=0.3)

    # (b) budget sweep
    ax = axes[1]
    policies = sorted({r["policy"] for r in budget_rows})
    for name in policies:
        pts = [r for r in budget_rows if r["policy"] == name]
        pts.sort(key=lambda r: r["budget"])
        unsound = any(not p["guaranteed"] for p in pts)
        ax.plot(
            [p["budget"] for p in pts], [p["verdicts_per_capture"] for p in pts],
            "--" if unsound else "o-", label=name, alpha=0.9, ms=4,
        )
    ax.set_xlabel("capture budget K")
    ax.set_ylabel("verdicts per capture")
    ax.set_title("(b) more photographs, diminishing returns")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # (c) the ladder
    ax = axes[2]
    labels = [r["standard"] for r in ladder_rows]
    x = np.arange(len(labels))
    ax.bar(x, [r["verdict_rate"] for r in ladder_rows], color="tab:blue", label="decided")
    ax.bar(x, [r["refer_rate"] for r in ladder_rows],
           bottom=[r["verdict_rate"] for r in ladder_rows], color="tab:grey", label="escalated")
    ax.set_xticks(x, [l.replace(" ", "\n") for l in labels], fontsize=6.5)
    ax.set_ylabel("fraction of cases")
    ax.set_title("(c) where a phone stops being admissible")
    ax.legend(fontsize=8)

    fig.suptitle("E3: The Docket -- justified verdicts per photograph", fontsize=12)
    fig.tight_layout()
    path = figure_path("e3_leaderboard.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


if __name__ == "__main__":
    main()
