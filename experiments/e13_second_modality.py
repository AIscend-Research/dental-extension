"""E13 -- a second real modality: chest radiographs, not dental ones.

The roadmap asked for a head-to-head on CheXphoto chest radiographs: "if the
guarantee and the ordering reproduce in a second modality, the claim stops
being about teeth." Real CheXphoto data -- natural photos of CheXpert x-rays,
plus CheXphoto's own synthetic photographic transforms -- is gated behind a
CheXpert data-use agreement not obtainable in this environment. Rather than
skip this or fake it, this experiment does the honest available version:
apply this project's OWN capture simulator and evidence machinery (not
CheXphoto's transform code, never obtained) to a genuinely different real
medical-imaging classification task on a freely licensed dataset -- Kermany
et al.'s pediatric pneumonia chest X-ray dataset (CC BY 4.0, no DUA,
`src/data/chest_xray_crops.py`).

What this can claim: whether the framework's qualitative story (targeted
retaking beats untargeted, the guarantee holds, verdicts-per-capture ordering
is stable) survives a second real classification task with a different
prevalence, a different reader, and different real images, using the exact
same unmodified evidence/bench code that runs on DENTEX.

What this cannot claim: that it reproduces CheXphoto's specific photographic
corruption model (glare-matte masks, moire, monitor-photography artifacts)
or their natural-photo capture pipeline (Nokia/iPhone rigs). It reuses this
project's own five-artifact taxonomy (blur/glare/low_light/angle/jpeg),
which was designed and grounded for photographing PRINTED FILMS or PRINTED
RADIOGRAPHS on a lightbox (see docs/simulator_grounding.md), not monitor
photography -- CheXphoto's validation-only "film" subset (200 photos of
physical x-ray films) is the closer analogue of the two capture scenarios
CheXphoto studies, and that is the one this taxonomy was built for.

Run: .venv/bin/python -m experiments.e13_second_modality
"""

from __future__ import annotations

import numpy as np

from experiments.common import (
    CLINIC_DIFFICULTY,
    HEADLINE_BURDEN,
    annotate_no_overlap,
    banner,
    figure_path,
    save_results,
    save_table,
)
from experiments.e6_real_images import collect_real_calibration, evaluate_auc, MIN_INTERPRETABLE_AUC
from src.bench.docket import make_image_docket
from src.bench.metrics import format_leaderboard, score_results
from src.bench.policies import policy_by_name
from src.bench.runner import run_docket
from src.data.chest_xray_crops import describe, load_chest_crops, split_test
from src.data.degradation import DEGRADATION_NAMES
from src.evidence.calibration import LikelihoodRatioCalibrator
from src.models.diagnostic import _auc
from src.models.real_channel import train_real_channel
from src.sim.session import CaptureSession

N_CASES = 3000
BUDGET = 4
N_RENDERS_TRAIN = 12
SEVERITY_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
ARMS = [
    "single_shot",
    "fixed_retake",
    "untargeted_evidential",
    "evidential_capture",
    "greedy_diagnostic",
    "naive_best_shot",
]


def main() -> None:
    banner("E13 -- second real modality: pneumonia vs normal chest radiographs")
    rng = np.random.default_rng(0)

    data = load_chest_crops()
    train, test_all = data["train"], data["test_all"]
    cal_crops, test = split_test(test_all, fraction=0.5, seed=3)
    for name, part in [("train", train), ("calibration", cal_crops), ("test", test)]:
        print(describe(part, name))

    print(f"\ntraining real_channel on {len(train)} chest radiographs x {N_RENDERS_TRAIN} renders ...", flush=True)
    channel = train_real_channel(train, rng, n_renders=N_RENDERS_TRAIN, clinic_difficulty=CLINIC_DIFFICULTY)

    clean_auc = evaluate_auc(channel, test, {}, rng)
    print(f"\n[1] clean AUC on held-out chest radiographs: {clean_auc:.3f}")
    interpretable = clean_auc >= MIN_INTERPRETABLE_AUC
    if not interpretable:
        print(f"    below {MIN_INTERPRETABLE_AUC} -- damage curve below is not read as a finding.")

    print("\n[2] per-artifact damage curve (AUC on held-out chest radiographs)")
    header = f"  {'severity':>9} " + " ".join(f"{n:>10}" for n in DEGRADATION_NAMES)
    print(header)
    damage_rows = []
    curves = {n: [] for n in DEGRADATION_NAMES}
    for sev in SEVERITY_GRID:
        row = {"severity": sev}
        for name in DEGRADATION_NAMES:
            auc = evaluate_auc(channel, test, {name: sev}, rng, n_repeats=6)
            row[name] = auc
            curves[name].append(auc)
        damage_rows.append(row)
        print(f"  {sev:>9.1f} " + " ".join(f"{row[n]:>10.3f}" for n in DEGRADATION_NAMES))

    measured_damage = {n: (clean_auc - curves[n][-1]) / max(clean_auc - 0.5, 1e-6) for n in DEGRADATION_NAMES}
    measured_rank = sorted(DEGRADATION_NAMES, key=lambda n: -measured_damage[n])
    print("\n  measured damage ordering (worst first): " + " > ".join(measured_rank))
    if not interpretable:
        print("  ^ INCONCLUSIVE: computed from a below-threshold reader; not evidence either way.")

    print("\n--- The Docket on real chest radiographs ---")
    cal_data = collect_real_calibration(channel, cal_crops, rng)
    print(f"  calibration: {len(cal_data)} first shots from {len(cal_crops)} held-out radiographs")
    calibrator = LikelihoodRatioCalibrator(n_strata=4).fit(cal_data.scores, cal_data.labels, cal_data.usabilities)

    pool = [c.image for c in test]
    docket = make_image_docket(
        "chest", [c.label for c in test], n_cases=N_CASES, budget=BUDGET,
        burden=HEADLINE_BURDEN, clinic_difficulty=CLINIC_DIFFICULTY, seed=5,
    )
    print(f"  docket: {len(docket)} cases over {len(pool)} distinct radiographs, "
          f"prevalence {docket.prevalence:.3f}, K={BUDGET}")

    rows = [
        score_results(name, run_docket(docket, policy_by_name(name), channel, calibrator, image_pool=pool), docket.burden)
        for name in ARMS
    ]
    print(format_leaderboard(rows))

    by_name = {r.policy: r for r in rows}
    sound_arms = ["single_shot", "fixed_retake", "untargeted_evidential", "evidential_capture"]
    ordering_reproduces = (
        by_name["evidential_capture"].verdicts_per_capture > by_name["untargeted_evidential"].verdicts_per_capture
        > by_name["fixed_retake"].verdicts_per_capture
    )
    no_violations = not any(r.convict_violation or r.discharge_violation for r in rows if r.guaranteed)
    print(
        f"\nordering (evidential_capture > untargeted > fixed) reproduces: {ordering_reproduces}"
        f"\nzero violations among guaranteed arms: {no_violations}"
    )

    fig = make_figure(damage_rows, clean_auc, rows)
    print(f"\nfigure -> {fig}")

    save_table("e13_damage_curves", damage_rows)
    save_table("e13_leaderboard", [r.as_dict() for r in rows])
    save_results("e13_second_modality", {
        "dataset": "Kermany et al. pediatric pneumonia chest X-ray (CC BY 4.0), NOT CheXphoto",
        "scoping_note": (
            "Real CheXphoto photographs are gated behind a CheXpert data-use agreement, "
            "not obtained. This experiment tests generalization of this project's OWN "
            "simulator/evidence machinery to a second real modality, not a reproduction "
            "of CheXphoto's specific corruption pipeline."
        ),
        "n_train": len(train), "n_calibration": len(cal_crops), "n_test": len(test),
        "clean_auc": clean_auc,
        "damage_analysis_interpretable": bool(interpretable),
        "damage_curves": damage_rows,
        "measured_damage": measured_damage,
        "measured_rank": measured_rank,
        "docket": {"n_cases": N_CASES, "budget": BUDGET, "prevalence": docket.prevalence},
        "leaderboard": [r.as_dict() for r in rows],
        "ordering_reproduces": bool(ordering_reproduces),
        "zero_violations": bool(no_violations),
        "figure": fig,
    })
    print("results -> results/e13_second_modality.json")


def make_figure(damage_rows, clean_auc, rows) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    ax = axes[0]
    sev = [r["severity"] for r in damage_rows]
    for name in DEGRADATION_NAMES:
        ax.plot(sev, [r[name] for r in damage_rows], "o-", label=name, ms=4)
    ax.axhline(0.5, color="k", ls=":", lw=1.2)
    ax.set_xlabel("rendered severity")
    ax.set_ylabel("AUC on held-out chest radiographs")
    ax.set_title(f"(a) pneumonia reader under real artifacts\n(clean AUC {clean_auc:.3f})")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for r in rows:
        ok = r.guaranteed and not (r.convict_violation or r.discharge_violation)
        ax.scatter(
            r.verdicts_per_capture, r.verdict_accuracy, s=90,
            facecolors="tab:blue" if ok else "none",
            edgecolors="tab:blue" if ok else "tab:red", linewidths=1.8,
            marker="o" if ok else "X",
        )
    ax.margins(0.18)
    ax.set_xlabel("verdicts per capture")
    ax.set_ylabel("accuracy on rendered verdicts")
    ax.set_title("(b) The Docket, second modality\n(chest radiographs, pneumonia vs normal)")
    ax.grid(alpha=0.3)
    ax_b = ax  # label placement deferred past tight_layout() -- see below

    fig.suptitle("E13: second-modality generalization check (NOT CheXphoto)", fontsize=12)
    fig.tight_layout()
    # Placed AFTER tight_layout(), not before -- see e3_leaderboard.py's
    # make_figure for why. naive_best_shot sits high enough in this panel
    # that its label used to collide with the two-line title above; the
    # title is now always in annotate_no_overlap's collision set for exactly
    # this reason, not just here.
    annotate_no_overlap(
        ax_b, [r.verdicts_per_capture for r in rows], [r.verdict_accuracy for r in rows],
        [r.policy.replace("_", "\n") for r in rows],
    )
    path = figure_path("e13_second_modality.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


if __name__ == "__main__":
    main()
