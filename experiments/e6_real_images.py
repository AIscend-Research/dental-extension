"""E6 -- the real-image arm: does any of this survive contact with radiographs?

Everything up to here ran on an analytic reader whose behaviour under
degradation was specified by hand. That is the right instrument for measuring a
theorem's coverage, and the wrong one for asking whether the assumptions
resemble reality. This experiment swaps in real DENTEX tooth crops and two
genuinely fitted models -- a diagnosis head and a degradation head -- and asks
four questions:

  1. Does the real reader degrade under rendered capture artifacts at all, and
     by how much?
  2. Does the *ordering* of per-artifact damage match what the surrogate
     assumes in `SIGNAL_LOSS_WEIGHTS`? This is the sim-to-real check that is
     actually available without new photography: the surrogate asserts blur
     and glare hurt most and off-axis geometry least, and a real classifier on
     real radiographs can confirm or refute that ranking.
  3. Can a degradation head trained by weak supervision on the simulator's own
     labels recover which artifact is present in a real image?
  4. Does the benchmark's policy ordering, and the guarantee, reproduce?

Two tasks are used, each where it is actually valid, because neither supports
both jobs:

  caries_vs_deep (Caries vs Deep Caries, 133 teeth) carries the artifact-damage
    analysis. Both classes are ordinary erupted teeth, so the decision rests on
    fine-grained density rather than shape -- the kind of evidence real caries
    reading depends on. It is far too thin on positives (32 in total) to
    support a conformal calibration set, so it does not run the Docket.

  caries_vs_other (Caries+Deep vs Impacted+Periapical, 182 teeth) carries the
    end-to-end Docket run. It has enough teeth per class to calibrate on, and
    it is close to saturated -- impacted teeth are separable on coarse
    morphology alone -- so it demonstrates the machinery runs on real
    radiographs and must NOT be read as evidence about caries detection
    difficulty.

Scope, stated plainly. Neither task is "caries vs healthy": DENTEX contains no
sound-tooth annotations, and manufacturing negatives from unannotated regions
would label undiagnosed disease as healthy (see src/data/dentex_crops.py).
Sessions resample capture conditions, not images, so every interval below is
narrow in the capture dimension and says nothing about generalisation across
patients.

Run: .venv/bin/python -m experiments.e6_real_images
"""

from __future__ import annotations

import numpy as np

from experiments.common import (
    CLINIC_DIFFICULTY,
    HEADLINE_BURDEN,
    banner,
    figure_path,
    save_results,
    save_table,
)
from src.bench.docket import make_image_docket
from src.bench.metrics import format_leaderboard, score_results
from src.bench.policies import policy_by_name
from src.bench.runner import CalibrationData, run_docket
from src.data.degradation import DEGRADATION_NAMES
from src.data.dentex_crops import describe, load_tooth_crops, split_by_source_image
from src.evidence.calibration import LikelihoodRatioCalibrator, StratifiedCalibrator
from src.models.diagnostic import SIGNAL_LOSS_WEIGHTS, Case, _auc
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


#: Below this AUC the reader carries no signal, so an artifact-damage curve is
#: measuring noise. Reported as inconclusive rather than as a finding.
MIN_INTERPRETABLE_AUC = 0.60


def grouped_cv_channels(crops, rng, n_folds=5, n_renders=N_RENDERS_TRAIN):
    """Leave-group-out CV over source radiographs, using every tooth once.

    With 133 crops and 32 positives, a single train/test split leaves ~21
    positives to fit on and ~5 to test on -- too few to fit anything or to
    measure it. Grouped CV spends the whole pool: every tooth is scored by a
    model that never saw its radiograph, and the folds are formed on
    `source_image_id` so no patient straddles a fold.

    Returns (folds, out_of_fold_auc) where folds is a list of
    (channel, held_out_crops) usable for the damage curves.
    """
    image_ids = sorted({c.source_image_id for c in crops})
    order = rng.permutation(len(image_ids))
    assignment = {image_ids[order[i]]: i % n_folds for i in range(len(image_ids))}

    folds, labels, scores = [], [], []
    for k in range(n_folds):
        train = [c for c in crops if assignment[c.source_image_id] != k]
        held = [c for c in crops if assignment[c.source_image_id] == k]
        if not held or len({c.label for c in train}) < 2:
            continue
        ch = train_real_channel(train, rng, n_renders=n_renders, clinic_difficulty=CLINIC_DIFFICULTY)
        folds.append((ch, held))
        for crop in held:
            case = Case(label=crop.label, difficulty=0.0, payload=crop.image)
            labels.append(crop.label)
            scores.append(ch.read(case, {}, rng).score)
    return folds, float(_auc(np.array(labels), np.array(scores)))


def evaluate_auc_folds(folds, severities, rng, n_repeats=4) -> float:
    """Pooled out-of-fold AUC at fixed severities, across all CV folds."""
    labels, scores = [], []
    for ch, held in folds:
        for crop in held:
            case = Case(label=crop.label, difficulty=0.0, payload=crop.image)
            for _ in range(n_repeats):
                labels.append(crop.label)
                scores.append(ch.read(case, severities, rng).score)
    return float(_auc(np.array(labels), np.array(scores)))


def evaluate_auc(channel, crops, severities, rng, n_repeats=6) -> float:
    """AUC of the real reader on `crops` rendered at fixed `severities`."""
    labels, scores = [], []
    for crop in crops:
        case = Case(label=crop.label, difficulty=0.0, payload=crop.image)
        for _ in range(n_repeats):
            labels.append(crop.label)
            scores.append(channel.read(case, severities, rng).score)
    return float(_auc(np.array(labels), np.array(scores)))


def collect_real_calibration(channel, crops, rng, n_per_crop=40) -> CalibrationData:
    """Calibration set from FIRST shots of held-out crops, matching the protocol."""
    scores, labels, usabilities = [], [], []
    for crop in crops:
        case = Case(label=crop.label, difficulty=0.0, payload=crop.image)
        for _ in range(n_per_crop):
            session = CaptureSession(rng=rng, difficulty=CLINIC_DIFFICULTY)
            capture = session.capture()
            reading = channel.read(case, capture.severities, rng)
            scores.append(reading.score)
            labels.append(crop.label)
            usabilities.append(reading.usability)
    return CalibrationData(
        scores=np.array(scores), labels=np.array(labels), usabilities=np.array(usabilities)
    )


def main() -> None:
    banner("E6 -- real DENTEX tooth crops")
    rng = np.random.default_rng(0)

    print("\n--- task A: caries_vs_deep (fine-grained; carries the damage analysis) ---")
    crops = load_tooth_crops(task="caries_vs_deep")
    print(describe(crops, "all"))
    print(f"\nfitting {5}-fold grouped CV over source radiographs ...", flush=True)
    folds, clean_auc = grouped_cv_channels(crops, rng)

    # -- 1 + 2. how much damage, and in what order --------------------------
    print(f"\n[1] real reader, out-of-fold on all {len(crops)} teeth: clean AUC {clean_auc:.3f}")
    interpretable = clean_auc >= MIN_INTERPRETABLE_AUC
    if not interpretable:
        print(
            f"    AUC is below {MIN_INTERPRETABLE_AUC} -- this reader carries essentially no\n"
            "    signal on this task, so the damage curves below measure noise and NO\n"
            "    ordering conclusion is drawn from them. See the summary at the end."
        )

    print("\n[2] per-artifact damage curve (AUC on held-out teeth)")
    header = f"  {'severity':>9} " + " ".join(f"{n:>10}" for n in DEGRADATION_NAMES)
    print(header)
    damage_rows = []
    curves = {n: [] for n in DEGRADATION_NAMES}
    for sev in SEVERITY_GRID:
        row = {"severity": sev}
        for name in DEGRADATION_NAMES:
            auc = evaluate_auc_folds(folds, {name: sev}, rng)
            row[name] = auc
            curves[name].append(auc)
        damage_rows.append(row)
        print(f"  {sev:>9.1f} " + " ".join(f"{row[n]:>10.3f}" for n in DEGRADATION_NAMES))

    # damage = how much AUC above chance is lost by severity 1.0
    measured_damage = {
        n: (clean_auc - curves[n][-1]) / max(clean_auc - 0.5, 1e-6) for n in DEGRADATION_NAMES
    }
    measured_rank = sorted(DEGRADATION_NAMES, key=lambda n: -measured_damage[n])
    assumed_rank = sorted(DEGRADATION_NAMES, key=lambda n: -SIGNAL_LOSS_WEIGHTS[n])
    rho = _spearman(
        [measured_damage[n] for n in DEGRADATION_NAMES],
        [SIGNAL_LOSS_WEIGHTS[n] for n in DEGRADATION_NAMES],
    )
    print("\n  measured damage ordering (worst first): " + " > ".join(measured_rank))
    print("  surrogate's assumed ordering           : " + " > ".join(assumed_rank))
    print(f"  Spearman correlation of the two rankings: {rho:+.3f}")
    if not interpretable:
        print("  ^ INCONCLUSIVE: computed from a chance-level reader; not evidence either way.")

    # -- 3. can the degradation head read a real image? ---------------------
    print("\n[3] degradation head on real held-out crops")
    truth, pred = [], []
    for ch, held in folds:
        for crop in held:
            case = Case(label=crop.label, difficulty=0.0, payload=crop.image)
            for _ in range(20):
                session = CaptureSession(rng=rng, difficulty=CLINIC_DIFFICULTY)
                capture = session.capture()
                reading = ch.read(case, capture.severities, rng)
                truth.append([capture.severities.get(n, 0.0) for n in DEGRADATION_NAMES])
                pred.append([reading.degradation[n] for n in DEGRADATION_NAMES])
    truth, pred = np.array(truth), np.array(pred)
    head_rows = []
    for i, name in enumerate(DEGRADATION_NAMES):
        r = float(np.corrcoef(truth[:, i], pred[:, i])[0, 1])
        mae = float(np.abs(truth[:, i] - pred[:, i]).mean())
        head_rows.append({"degradation": name, "pearson_r": r, "mae": mae})
        print(f"  {name:<10} r = {r:+.3f} | MAE {mae:.3f}")
    dominant_acc = float(np.mean(truth.argmax(axis=1) == pred.argmax(axis=1)))
    chance = 1.0 / len(DEGRADATION_NAMES)
    print(f"  dominant-artifact accuracy: {dominant_acc:.3f} (chance {chance:.2f})")

    # -- 4. the benchmark, on real images -----------------------------------
    # Switched to the coarse task here: caries_vs_deep has only ~6 positive
    # teeth in the calibration split, which cannot support a conformal null.
    print("\n--- task B: caries_vs_other (coarse; carries the end-to-end Docket) ---")
    crops_b = load_tooth_crops(task="caries_vs_other")
    train_b, cal_crops, test = split_by_source_image(crops_b, seed=3)
    for name, part in [("train", train_b), ("calibration", cal_crops), ("test", test)]:
        print(describe(part, name))
    channel_b = train_real_channel(train_b, rng, n_renders=N_RENDERS_TRAIN, clinic_difficulty=CLINIC_DIFFICULTY)
    clean_auc_b = evaluate_auc(channel_b, test, {}, rng)
    print(f"  clean AUC on this task: {clean_auc_b:.3f} (near-saturated by design -- see docstring)")
    channel = channel_b

    print("\n[4] The Docket on real crops")
    cal_data = collect_real_calibration(channel, cal_crops, rng)
    print(f"  calibration: {len(cal_data)} first shots from {len(cal_crops)} held-out teeth")
    calibrator = LikelihoodRatioCalibrator(n_strata=4).fit(
        cal_data.scores, cal_data.labels, cal_data.usabilities
    )
    conformal = StratifiedCalibrator(n_strata=4).fit(
        cal_data.scores, cal_data.labels, cal_data.usabilities
    )

    pool = [c.image for c in test]
    docket = make_image_docket(
        "real", [c.label for c in test], n_cases=N_CASES, budget=BUDGET,
        burden=HEADLINE_BURDEN, clinic_difficulty=CLINIC_DIFFICULTY, seed=5,
    )
    print(f"  docket: {len(docket)} cases over {len(pool)} distinct teeth, "
          f"prevalence {docket.prevalence:.3f}, K={BUDGET}")

    leaderboards = {}
    for label, cal in [("likelihood-ratio", calibrator), ("conformal", conformal)]:
        rows = [
            score_results(
                name, run_docket(docket, policy_by_name(name), channel, cal, image_pool=pool),
                docket.burden,
            )
            for name in ARMS
        ]
        leaderboards[label] = rows
        print(f"\n  evidence construction: {label}")
        print(format_leaderboard(rows))
        save_table(f"e6_real_leaderboard_{label}", [r.as_dict() for r in rows])

    fig = make_figure(damage_rows, clean_auc, head_rows, leaderboards["likelihood-ratio"])
    print(f"\nfigure -> {fig}")

    save_table("e6_damage_curves", damage_rows)
    save_table("e6_degradation_head", head_rows)
    save_results("e6_real_images", {
        "task_a_damage_analysis": {
            "task": "caries_vs_deep (Deep Caries vs Caries)",
            "n_crops": len(crops),
            "n_source_images": len({c.source_image_id for c in crops}),
            "protocol": "5-fold grouped CV over source radiographs, all teeth scored out-of-fold",
        },
        "task_b_docket": {
            "task": "caries_vs_other (Caries+Deep vs Impacted+Periapical)",
            "n_crops": len(crops_b), "n_train": len(train_b),
            "n_calibration_crops": len(cal_crops), "n_test": len(test),
            "clean_auc": clean_auc_b,
            "note": "near-saturated; demonstrates the pipeline, not caries difficulty",
        },
        "clean_auc": clean_auc,
        "damage_curves": damage_rows,
        "measured_damage": measured_damage,
        "assumed_weights": SIGNAL_LOSS_WEIGHTS,
        "measured_rank": measured_rank,
        "assumed_rank": assumed_rank,
        "rank_spearman": rho,
        "damage_analysis_interpretable": bool(interpretable),
        "min_interpretable_auc": MIN_INTERPRETABLE_AUC,
        "degradation_head": head_rows,
        "dominant_artifact_accuracy": dominant_acc,
        "leaderboards": {k: [r.as_dict() for r in v] for k, v in leaderboards.items()},
        "figure": fig,
    })
    print("results -> results/e6_real_images.json")


def _spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def make_figure(damage_rows, clean_auc, head_rows, leaderboard) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    ax = axes[0]
    sev = [r["severity"] for r in damage_rows]
    for name in DEGRADATION_NAMES:
        ax.plot(sev, [r[name] for r in damage_rows], "o-", label=name, ms=4)
    ax.axhline(0.5, color="k", ls=":", lw=1.2)
    ax.set_xlabel("rendered severity")
    ax.set_ylabel("AUC on held-out real teeth")
    ax.set_title(f"(a) real reader under real artifacts\n(clean AUC {clean_auc:.3f})")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    names = [r["degradation"] for r in head_rows]
    x = np.arange(len(names))
    ax.bar(x, [r["pearson_r"] for r in head_rows], color="tab:green")
    ax.set_xticks(x, names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("correlation, predicted vs true severity")
    ax.set_title("(b) degradation head, weak supervision\non real crops")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    for r in leaderboard:
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
    ax.set_title("(c) The Docket, on real radiographs")
    ax.grid(alpha=0.3)

    fig.suptitle("E6: the real-image arm (DENTEX, 182 teeth / 50 radiographs)", fontsize=12)
    fig.tight_layout()
    path = figure_path("e6_real_images.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


if __name__ == "__main__":
    main()
