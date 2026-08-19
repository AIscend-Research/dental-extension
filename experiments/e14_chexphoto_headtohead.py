"""E14 -- head-to-head against CheXphoto's own corruption model.

The roadmap item is "head-to-head on CheXphoto chest radiographs: if the
guarantee and the ordering reproduce in a second modality, the claim stops
being about teeth." E13 did the modality half -- real chest radiographs
instead of teeth -- but ran them through *this project's* capture simulator,
which leaves the sharpest version of the objection standing: the simulator and
the method were designed together, so a guarantee that holds under it might be
a fact about the simulator rather than about the method.

This experiment closes that half. CheXphoto's synthetic transformations were
released as code under the MIT License (the *dataset* is gated; see
`docs/chexphoto_access.md`), so the corruption model itself is obtainable even
though the photographs are not. `src/data/chexphoto_transforms.py` is a
verified port of that code (`tests/test_chexphoto_transforms.py` checks it
against the vendored original), and `src/models/chexphoto_channel.py` swaps it
in as the renderer behind an otherwise identical reader.

So the head-to-head is: same real chest radiographs, same fitted reader, same
calibrator protocol, same policies, same docket, same seeds -- and two capture
processes, ours and CheXphoto's. Three questions, in order of how much they
matter:

  1. Does the **guarantee** survive a third party's corruption model? (If
     validity only holds under the corruption process we wrote, it is not a
     property of the method.)
  2. Does the **ordering** -- targeted retaking beats untargeted beats fixed --
     survive it?
  3. Does a calibrator fitted under *our* corruption model stay valid when
     deployed under *theirs*? This is the deployment question: nobody
     calibrates on the corruption process they will actually meet.

What this still does not do: use CheXphoto's natural photographs (Nokia10k,
iPhone1k, the 250-image film subset). Those are the gated half, and no result
here should be described as running on CheXphoto data. What runs here is
CheXphoto's corruption model on freely licensed chest radiographs (Kermany et
al., CC BY 4.0), which is exactly the same image pool E13 used -- deliberately,
so that the only thing that changes between E13 and E14 is the capture process.

Run: .venv/bin/python -m experiments.e14_chexphoto_headtohead
"""

from __future__ import annotations

import numpy as np

from experiments.common import CLINIC_DIFFICULTY, HEADLINE_BURDEN, banner, figure_path, save_results, save_table
from experiments.e6_real_images import collect_real_calibration, evaluate_auc, MIN_INTERPRETABLE_AUC
from src.bench.docket import make_image_docket
from src.bench.metrics import format_leaderboard, score_results
from src.bench.policies import policy_by_name
from src.bench.runner import run_docket
from src.data.chest_xray_crops import describe, load_chest_crops, split_test
from src.data.chexphoto_transforms import LEVELS
from src.data.degradation import DEGRADATION_NAMES
from src.evidence.calibration import LikelihoodRatioCalibrator
from src.models.chexphoto_channel import ARTIFACT_TO_PERTURBATION, CheXphotoChannel
from src.models.real_channel import train_real_channel

# Smaller than E13's 3000 because a CheXphoto capture costs ~3x a simulated one
# (the moire stage upsamples and composites two warped masks). Both arms here
# run at the same size on the same docket seed, so the head-to-head comparison
# is exact; only the E13 cross-reference is approximate.
N_CASES = 1500
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
# Mid-band severities that bin onto CheXphoto levels 1..4 exactly (L/4).
LEVEL_SEVERITY = {level: level / len(LEVELS) for level in LEVELS}


def damage_ordering(clean_auc: float, final_auc: dict[str, float]) -> tuple[dict[str, float], list[str]]:
    """Normalised damage per artifact and the worst-first ordering it implies."""
    damage = {n: (clean_auc - final_auc[n]) / max(clean_auc - 0.5, 1e-6) for n in final_auc}
    return damage, sorted(final_auc, key=lambda n: -damage[n])


def kendall_tau(a: list[str], b: list[str]) -> float:
    """Rank correlation between two orderings of the same items, -1 to 1."""
    rank_b = {name: i for i, name in enumerate(b)}
    items = [rank_b[name] for name in a]
    n = len(items)
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if items[i] < items[j]:
                concordant += 1
            else:
                discordant += 1
    total = n * (n - 1) / 2
    return float((concordant - discordant) / total) if total else float("nan")


def main() -> None:
    banner("E14 -- head-to-head: this project's capture model vs CheXphoto's")
    rng = np.random.default_rng(0)

    data = load_chest_crops()
    train, test_all = data["train"], data["test_all"]
    cal_crops, test = split_test(test_all, fraction=0.5, seed=3)
    for name, part in [("train", train), ("calibration", cal_crops), ("test", test)]:
        print(describe(part, name))

    # Same seed and same renders as E13, so this is E13's reader, not a new one.
    print(f"\ntraining real_channel on {len(train)} chest radiographs x {N_RENDERS_TRAIN} renders ...", flush=True)
    sim_channel = train_real_channel(train, rng, n_renders=N_RENDERS_TRAIN, clinic_difficulty=CLINIC_DIFFICULTY)
    cxp_channel = CheXphotoChannel(reader=sim_channel)
    print("  the CheXphoto arm reuses this exact fitted reader -- only the renderer differs")

    clean_auc = evaluate_auc(sim_channel, test, {}, rng)
    interpretable = clean_auc >= MIN_INTERPRETABLE_AUC
    print(f"\n[1] clean AUC on held-out chest radiographs: {clean_auc:.3f}")
    if not interpretable:
        print(f"    below {MIN_INTERPRETABLE_AUC} -- the damage comparison below is not read as a finding.")

    # ---------------------------------------------------------------- damage
    print("\n[2] damage head-to-head: same reader, same images, two corruption models")
    print("\n  (a) this project's simulator, severity 0 -> 1")
    print(f"  {'severity':>9} " + " ".join(f"{n:>10}" for n in DEGRADATION_NAMES))
    sim_rows = []
    sim_curves = {n: [] for n in DEGRADATION_NAMES}
    for sev in SEVERITY_GRID:
        row = {"severity": sev}
        for name in DEGRADATION_NAMES:
            auc = evaluate_auc(sim_channel, test, {name: sev}, rng, n_repeats=6)
            row[name] = auc
            sim_curves[name].append(auc)
        sim_rows.append(row)
        print(f"  {sev:>9.1f} " + " ".join(f"{row[n]:>10.3f}" for n in DEGRADATION_NAMES))

    print("\n  (b) CheXphoto's transformations, level 1 -> 4")
    header_names = [f"{ARTIFACT_TO_PERTURBATION[n]}" for n in DEGRADATION_NAMES]
    print(f"  {'level':>9} " + " ".join(f"{n:>15}" for n in header_names))
    cxp_rows = []
    cxp_curves = {n: [] for n in DEGRADATION_NAMES}
    for level in LEVELS:
        row = {"level": level}
        for name in DEGRADATION_NAMES:
            auc = evaluate_auc(cxp_channel, test, {name: LEVEL_SEVERITY[level]}, rng, n_repeats=6)
            row[ARTIFACT_TO_PERTURBATION[name]] = auc
            cxp_curves[name].append(auc)
        cxp_rows.append(row)
        print(f"  {level:>9d} " + " ".join(f"{row[ARTIFACT_TO_PERTURBATION[n]]:>15.3f}" for n in DEGRADATION_NAMES))

    sim_damage, sim_rank = damage_ordering(clean_auc, {n: sim_curves[n][-1] for n in DEGRADATION_NAMES})
    cxp_damage, cxp_rank = damage_ordering(clean_auc, {n: cxp_curves[n][-1] for n in DEGRADATION_NAMES})
    tau = kendall_tau(sim_rank, cxp_rank)
    print("\n  damage ordering at full strength (worst first)")
    print("    ours:      " + " > ".join(sim_rank))
    print("    CheXphoto: " + " > ".join(f"{ARTIFACT_TO_PERTURBATION[n]}" for n in cxp_rank))
    print(f"    Kendall tau between the two orderings: {tau:+.2f}")
    print("    (jpeg <-> moire is a substitution, not a correspondence -- see chexphoto_channel.py)")

    # Where on our severity axis does each CheXphoto level land? Answered per
    # artifact by finding the severity whose AUC is closest to that level's.
    alignment = {}
    for name in DEGRADATION_NAMES:
        per_level = {}
        for i, level in enumerate(LEVELS):
            target = cxp_curves[name][i]
            best = min(range(len(SEVERITY_GRID)), key=lambda k: abs(sim_curves[name][k] - target))
            per_level[level] = SEVERITY_GRID[best]
        alignment[name] = per_level
    print("\n  severity alignment: our severity whose damage matches each CheXphoto level")
    print(f"  {'artifact':>10} " + " ".join(f"{'L' + str(l):>6}" for l in LEVELS))
    for name in DEGRADATION_NAMES:
        print(f"  {name:>10} " + " ".join(f"{alignment[name][l]:>6.1f}" for l in LEVELS))

    # ---------------------------------------------------------------- docket
    print("\n[3] The Docket, run twice: our capture model and CheXphoto's")
    pool = [c.image for c in test]
    docket = make_image_docket(
        "chest", [c.label for c in test], n_cases=N_CASES, budget=BUDGET,
        burden=HEADLINE_BURDEN, clinic_difficulty=CLINIC_DIFFICULTY, seed=5,
    )
    print(f"  docket: {len(docket)} cases over {len(pool)} distinct radiographs, "
          f"prevalence {docket.prevalence:.3f}, K={BUDGET}")

    arms_out = {}
    calibrators = {}
    for arm, channel in [("simulator", sim_channel), ("chexphoto", cxp_channel)]:
        print(f"\n  --- {arm} capture model ---", flush=True)
        cal_data = collect_real_calibration(channel, cal_crops, rng)
        calibrator = LikelihoodRatioCalibrator(n_strata=4).fit(cal_data.scores, cal_data.labels, cal_data.usabilities)
        calibrators[arm] = calibrator
        rows = [
            score_results(name, run_docket(docket, policy_by_name(name), channel, calibrator, image_pool=pool),
                          docket.burden)
            for name in ARMS
        ]
        print(format_leaderboard(rows))
        arms_out[arm] = rows

    def verdict(rows) -> dict:
        by_name = {r.policy: r for r in rows}
        return {
            "ordering_reproduces": bool(
                by_name["evidential_capture"].verdicts_per_capture
                > by_name["untargeted_evidential"].verdicts_per_capture
                > by_name["fixed_retake"].verdicts_per_capture
            ),
            "zero_violations": bool(
                not any(r.convict_violation or r.discharge_violation for r in rows if r.guaranteed)
            ),
            "verdicts_per_capture": {r.policy: r.verdicts_per_capture for r in rows},
        }

    checks = {arm: verdict(rows) for arm, rows in arms_out.items()}
    print("\n  head-to-head verdict")
    for arm, c in checks.items():
        print(f"    {arm:<10} ordering reproduces: {str(c['ordering_reproduces']):<5} "
              f"zero violations: {c['zero_violations']}")

    # ------------------------------------------------- calibration transfer
    print("\n[4] calibration transfer: calibrator fitted on OUR captures, deployed on CheXphoto's")
    print("    (the deployment case -- nobody calibrates on the corruption process they will meet)")
    transfer_rows = [
        score_results(
            name,
            run_docket(docket, policy_by_name(name), cxp_channel, calibrators["simulator"], image_pool=pool),
            docket.burden,
        )
        for name in ("single_shot", "fixed_retake", "untargeted_evidential", "evidential_capture")
    ]
    print(format_leaderboard(transfer_rows))
    transfer_violations = [
        r.policy for r in transfer_rows if r.guaranteed and (r.convict_violation or r.discharge_violation)
    ]
    print(f"    arms whose guarantee broke under the mismatched calibrator: {transfer_violations or 'none'}")

    fig = make_figure(sim_rows, cxp_rows, clean_auc, arms_out)
    print(f"\nfigure -> {fig}")

    save_table("e14_damage_simulator", sim_rows)
    save_table("e14_damage_chexphoto", cxp_rows)
    for arm, rows in arms_out.items():
        save_table(f"e14_leaderboard_{arm}", [r.as_dict() for r in rows])
    save_table("e14_leaderboard_transfer", [r.as_dict() for r in transfer_rows])
    save_results("e14_chexphoto_headtohead", {
        "corruption_models": {
            "simulator": "src/data/degradation.py (this project's five artifacts)",
            "chexphoto": "src/data/chexphoto_transforms.py (port of the MIT-licensed CheXphoto release)",
        },
        "scoping_note": (
            "CheXphoto's synthetic transformation CODE is used here, applied to freely "
            "licensed chest radiographs (Kermany et al., CC BY 4.0). CheXphoto's natural "
            "photographs are gated and were NOT obtained; see docs/chexphoto_access.md. "
            "The jpeg->moire slot is a substitution, not a correspondence."
        ),
        "artifact_mapping": ARTIFACT_TO_PERTURBATION,
        "n_train": len(train), "n_calibration": len(cal_crops), "n_test": len(test),
        "clean_auc": clean_auc,
        "damage_analysis_interpretable": bool(interpretable),
        "damage_simulator": sim_rows,
        "damage_chexphoto": cxp_rows,
        "damage_ordering": {"simulator": sim_rank, "chexphoto": cxp_rank, "kendall_tau": tau},
        "normalised_damage": {"simulator": sim_damage, "chexphoto": cxp_damage},
        "severity_alignment": alignment,
        "docket": {"n_cases": N_CASES, "budget": BUDGET, "prevalence": docket.prevalence},
        "leaderboards": {arm: [r.as_dict() for r in rows] for arm, rows in arms_out.items()},
        "checks": checks,
        "calibration_transfer": {
            "leaderboard": [r.as_dict() for r in transfer_rows],
            "arms_with_violations": transfer_violations,
        },
        "figure": fig,
    })
    print("results -> results/e14_chexphoto_headtohead.json")


def make_figure(sim_rows, cxp_rows, clean_auc, arms_out) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    ax = axes[0]
    sev = [r["severity"] for r in sim_rows]
    for name in DEGRADATION_NAMES:
        ax.plot(sev, [r[name] for r in sim_rows], "o-", label=name, ms=4)
    ax.axhline(0.5, color="k", ls=":", lw=1.2)
    ax.set_xlabel("severity (this project's simulator)")
    ax.set_ylabel("AUC on held-out chest radiographs")
    ax.set_title(f"(a) our capture model\n(clean AUC {clean_auc:.3f})")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    levels = [r["level"] for r in cxp_rows]
    for name in DEGRADATION_NAMES:
        key = ARTIFACT_TO_PERTURBATION[name]
        ax.plot(levels, [r[key] for r in cxp_rows], "s-", label=key, ms=4)
    ax.axhline(0.5, color="k", ls=":", lw=1.2)
    ax.set_xticks(list(LEVELS))
    ax.set_xlabel("level (CheXphoto)")
    ax.set_title("(b) CheXphoto's capture model\n(same reader, same images)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[2]
    markers = {"simulator": "o", "chexphoto": "s"}
    colours = {"simulator": "tab:blue", "chexphoto": "tab:orange"}
    for arm, rows in arms_out.items():
        for r in rows:
            ok = r.guaranteed and not (r.convict_violation or r.discharge_violation)
            ax.scatter(
                r.verdicts_per_capture, r.verdict_accuracy, s=80,
                facecolors=colours[arm] if ok else "none",
                edgecolors=colours[arm], linewidths=1.7,
                marker=markers[arm] if ok else "X",
                label=arm if r.policy == "single_shot" else None,
            )
    ax.set_xlabel("verdicts per capture")
    ax.set_ylabel("accuracy on rendered verdicts")
    ax.set_title("(c) The Docket under both capture models\n(filled = guarantee held)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("E14: head-to-head against CheXphoto's corruption model (not CheXphoto data)", fontsize=12)
    fig.tight_layout()
    path = figure_path("e14_chexphoto_headtohead.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


if __name__ == "__main__":
    main()
