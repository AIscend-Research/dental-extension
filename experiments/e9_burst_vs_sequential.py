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

    save_table("e9_burst_vs_sequential", [r.as_dict() for r in rows])
    save_results("e9_burst_vs_sequential", {
        "reader": {"clean_auc": world.clean_auc, "clinic_auc": world.clinic_auc},
        "docket": {"n_cases": N_CASES, "budget": K, "burden": HEADLINE_BURDEN.name},
        "sequential": sequential.as_dict(),
        "burst_mismatched_null": burst_mismatched.as_dict(),
        "burst_own_null": burst_fair.as_dict(),
        "calibration_mismatch_cost_vpc": mismatch_cost,
        "fusion_vs_sequential_gap_vpc": fusion_gap,
    })
    print("results -> results/e9_burst_vs_sequential.json")


if __name__ == "__main__":
    main()
