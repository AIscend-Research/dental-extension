"""E9 -- burst fusion vs sequential accumulation, calibrated fairly.

`src/models/fusion.py` (a real nn.Module, attention-weighted burst merge over
detector features) is built and tested but unused end to end, because the
project's actual answer to "how do you combine K looks" is the wealth
process: bet on each shot, multiply. `docs/phase4_adjacent_fields.md` flags
that computational photography and astronomical stacking both treat simple
averaging as the baseline any learned fusion has to beat -- this experiment
runs that comparison on the Docket itself, at the level the framework
actually operates (fuse the two channel readings, not raw pixels; a trained
BurstFusion needs Kaggle features and is a separate, deferred comparison, not
this one).

`fixed_retake` already spends K untargeted shots and accumulates their
evidence *sequentially*, one wealth update per shot. `burst_fusion_analytic`
spends the identical K untargeted shots but averages the diagnosis score and
every predicted severity across the burst first, then tests the fused
reading once. Both are sound (untargeted, degradation-channel-only stakes).

The one thing that makes this comparison easy to get wrong, and worth
surfacing explicitly rather than by accident: averaging K scores shrinks
their variance toward 0.5 (regression toward the mean), so testing the fused
statistic against a calibration pool fit on *single*-shot scores is a
distribution mismatch that cripples the fused arm for reasons that have
nothing to do with whether fusion is a good idea -- it looks unusable purely
because its null was never fit to it. Part 1 below reproduces that mistake
on purpose, to show how large the effect is. Part 2 fixes it by calibrating
a dedicated null on the same K-shot-averaged statistic burst_fusion_analytic
actually produces, which is the fair comparison.

Run: .venv/bin/python -m experiments.e9_burst_vs_sequential
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
from src.bench.policies import BurstFusionAnalytic, FixedRetake
from src.bench.runner import CalibrationData, run_docket
from src.evidence.calibration import LikelihoodRatioCalibrator
from src.models.diagnostic import Case, predicted_usability
from src.sim.instructions import RETAKE_ANY
from src.sim.session import CaptureSession

N_CASES = 4000
K = 4
CALIBRATION_N = 8000


def collect_burst_calibration(
    channel,
    k: int,
    n: int = CALIBRATION_N,
    prevalence: float = PREVALENCE,
    clinic_difficulty: float = CLINIC_DIFFICULTY,
    seed: int = 54321,
    difficulty_alpha: float = 2.0,
    difficulty_beta: float = 4.0,
) -> CalibrationData:
    """Calibration data over the SAME fused statistic `BurstFusionAnalytic`
    tests -- the mean score and mean per-artifact severity of a k-shot
    untargeted burst -- not over single first shots.

    This is the burst analogue of `src.bench.runner.collect_calibration`:
    same idea (label the null distribution the test statistic will actually
    be compared against), applied to a different statistic.
    """
    rng = np.random.default_rng(seed)
    scores = np.empty(n)
    labels = np.empty(n, dtype=int)
    usabilities = np.empty(n)

    for i in range(n):
        label = int(rng.random() < prevalence)
        difficulty = float(rng.beta(difficulty_alpha, difficulty_beta))
        case = Case(label=label, difficulty=difficulty)
        session = CaptureSession(rng=rng, difficulty=clinic_difficulty)

        shot_scores = []
        deg_sums: dict[str, float] = {}
        for shot_i in range(k):
            capture = session.capture(None if shot_i == 0 else RETAKE_ANY)
            reading = channel.read(case, capture.severities, rng)
            shot_scores.append(reading.score)
            for key, value in reading.degradation.items():
                deg_sums[key] = deg_sums.get(key, 0.0) + value

        scores[i] = float(np.mean(shot_scores))
        labels[i] = label
        usabilities[i] = predicted_usability({key: v / k for key, v in deg_sums.items()})

    return CalibrationData(scores=scores, labels=labels, usabilities=usabilities)


def main() -> None:
    world = build_world()
    banner("E9 -- burst fusion vs sequential accumulation", world)

    docket = make_docket(
        "burst_vs_sequential", n_cases=N_CASES, prevalence=PREVALENCE, budget=K,
        burden=HEADLINE_BURDEN, clinic_difficulty=CLINIC_DIFFICULTY, seed=31,
    )

    # the sequential baseline: normal single-shot-fit calibration
    lr_singleshot = LikelihoodRatioCalibrator(n_strata=world.calibrator.n_strata).fit(
        world.calibration_data.scores, world.calibration_data.labels, world.calibration_data.usabilities
    )
    sequential = score_results(
        "fixed_retake (sequential)",
        run_docket(docket, FixedRetake(), world.channel, lr_singleshot),
        docket.burden,
    )

    # Part 1: burst fusion tested against the WRONG null (single-shot-fit) --
    # the mistake this experiment exists to name.
    burst_mismatched = score_results(
        "burst_fusion (mismatched null)",
        run_docket(docket, BurstFusionAnalytic(k=K), world.channel, lr_singleshot),
        docket.burden,
    )

    # Part 2: burst fusion tested against its OWN null -- the fair comparison.
    print(f"\nfitting a dedicated null for the {K}-shot fused statistic (n={CALIBRATION_N}) ...", flush=True)
    burst_cal_data = collect_burst_calibration(world.channel, K)
    lr_burst = LikelihoodRatioCalibrator(n_strata=world.calibrator.n_strata).fit(
        burst_cal_data.scores, burst_cal_data.labels, burst_cal_data.usabilities
    )
    burst_fair = score_results(
        "burst_fusion (own null)",
        run_docket(docket, BurstFusionAnalytic(k=K), world.channel, lr_burst),
        docket.burden,
    )

    rows = [sequential, burst_mismatched, burst_fair]
    print(f"\nK={K} shots, all arms untargeted, all sound:")
    print(format_leaderboard(rows))

    mismatch_cost = burst_fair.verdicts_per_capture - burst_mismatched.verdicts_per_capture
    fusion_gap = sequential.verdicts_per_capture - burst_fair.verdicts_per_capture
    print(
        f"\ncalibration-mismatch cost (own null - single-shot null): {mismatch_cost:+.4f} VPC"
        f"\nfusion vs sequential, both fairly calibrated: sequential is "
        f"{'ahead' if fusion_gap > 0 else 'behind'} by {abs(fusion_gap):.4f} VPC "
        f"({sequential.verdicts_per_capture:.3f} vs {burst_fair.verdicts_per_capture:.3f})"
    )

    fig = make_figure(sequential, burst_mismatched, burst_fair, mismatch_cost, fusion_gap)
    print(f"\nfigure -> {fig}")

    save_table("e9_burst_vs_sequential", [r.as_dict() for r in rows])
    save_results("e9_burst_vs_sequential", {
        "reader": {"clean_auc": world.clean_auc, "clinic_auc": world.clinic_auc},
        "docket": {"n_cases": N_CASES, "budget": K, "burden": HEADLINE_BURDEN.name},
        "sequential": sequential.as_dict(),
        "burst_mismatched_null": burst_mismatched.as_dict(),
        "burst_own_null": burst_fair.as_dict(),
        "calibration_mismatch_cost_vpc": mismatch_cost,
        "fusion_vs_sequential_gap_vpc": fusion_gap,
        "figure": fig,
    })
    print("results -> results/e9_burst_vs_sequential.json")


def make_figure(sequential, burst_mismatched, burst_fair, mismatch_cost, fusion_gap) -> str:
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.gridspec import GridSpec

    from experiments.common import sample_tooth_crops
    from src.sim.render import render_severities

    fig = plt.figure(figsize=(11, 6.4))
    outer = GridSpec(2, 1, figure=fig, height_ratios=[1.0, 2.1], hspace=0.55)
    gs_top = outer[0].subgridspec(1, 4, wspace=0.15)
    gs_bottom = outer[1].subgridspec(1, 2, wspace=0.6)

    crop = sample_tooth_crops(1, label=1, seed=9)[0]
    for i in range(4):
        ax = fig.add_subplot(gs_top[0, i])
        rng = np.random.default_rng(100 + i)
        severities = {name: float(rng.uniform(0.15, 0.7)) for name in ("blur", "glare", "angle", "low_light")}
        out = render_severities(crop.image, severities, rng=rng)
        ax.imshow(cv2.cvtColor(out.image, cv2.COLOR_BGR2RGB))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"shot {i + 1}", fontsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#9aa0a6")
    fig.text(0.5, 0.945, "one real tooth, K=4 untargeted shots -- sequential bets on each in turn; burst fusion averages them first",
              ha="center", fontsize=9.5, color="#6b6b6b")

    ax = fig.add_subplot(gs_bottom[0, 0])
    rows = [sequential, burst_mismatched, burst_fair]
    labels = ["sequential\n(wealth process)", "burst fusion\n(wrong null)", "burst fusion\n(own null)"]
    colors = ["tab:blue", "tab:red", "tab:orange"]
    vpc = [r.verdicts_per_capture for r in rows]
    bars = ax.bar(labels, vpc, color=colors)
    for bar, v in zip(bars, vpc):
        ax.annotate(f"{v:.3f}", (bar.get_x() + bar.get_width() / 2, v), ha="center", va="bottom", fontsize=9)
    ax.annotate(
        "", xy=(1, burst_fair.verdicts_per_capture), xytext=(1, burst_mismatched.verdicts_per_capture),
        arrowprops=dict(arrowstyle="<->", color="tab:red", lw=1.4),
    )
    ax.text(1.08, (burst_mismatched.verdicts_per_capture + burst_fair.verdicts_per_capture) / 2,
            f"calibration\nmismatch cost\n{mismatch_cost:+.3f} VPC", fontsize=8, color="tab:red", va="center")
    ax.set_ylabel("verdicts per capture")
    ax.set_title("(a) fusing K shots: testing the fused\nstatistic against the wrong null cripples it")
    ax.grid(alpha=0.3, axis="y")

    ax = fig.add_subplot(gs_bottom[0, 1])
    fair_rows = [sequential, burst_fair]
    x = np.arange(len(fair_rows))
    width = 0.35
    b1 = ax.bar(x - width / 2, [r.verdicts_per_capture for r in fair_rows], width, label="verdicts per capture", color="tab:blue")
    ax2 = ax.twinx()
    b2 = ax2.bar(x + width / 2, [r.verdict_accuracy for r in fair_rows], width, label="accuracy", color="tab:green")
    ax.set_xticks(x, ["sequential\n(wealth process)", "burst fusion\n(own null)"])
    ax.set_ylabel("verdicts per capture", color="tab:blue")
    ax2.set_ylabel("accuracy on rendered verdicts", color="tab:green")
    ax.set_title(f"(b) fairly calibrated: sequential ahead by\n{fusion_gap:.3f} VPC, fusion more accurate when it decides")
    ax.legend(handles=[b1, b2], loc="upper center", fontsize=8)

    fig.suptitle("E9: burst fusion vs sequential wealth accumulation, both fairly calibrated", fontsize=12, y=1.0)
    path = figure_path("e9_burst_vs_sequential.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return str(path)


if __name__ == "__main__":
    main()
