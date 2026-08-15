"""E10 -- ablate the instrument: does the Docket's ordering survive a dumb head?

Every result so far reads the degradation channel through `SurrogateChannel`'s
`_read_degradation`: a noisy but *unbiased, per-artifact* estimator -- a
stand-in for a trained confidence head that can tell blur from glare from
tilt. That is a real, load-bearing modelling choice (E4 already shows VPC
falls from 0.230 to 0.125 as its noise rises from 0 to 0.40), but noise alone
does not test whether the head's ability to *distinguish artifact types*
matters, only whether it is accurate.

This experiment swaps it for something much dumber: `BlurVarianceChannel`
reports one scalar -- a proxy for what a classical Laplacian-variance
sharpness detector would give you, driven only by the two artifacts that
actually destroy high-frequency image content (blur, jpeg) -- and reports
that SAME number for every one of the five artifact slots, because a
single-number sharpness metric has no way to say "this is glare, not blur".
It is the instrument a team would reach for before building a learned
confidence head at all.

The question: does targeting still beat not-targeting when the "confidence
head" cannot actually tell artifacts apart? If the ordering survives
unweakened, the paper's claim that a learned, type-predicting confidence head
is what makes targeted retaking work is weaker than stated -- a scalar
quality gate would have been enough. If targeting's advantage collapses (or
degrades sharply), that is direct evidence the head's type prediction, not
just its scalar quality signal, is what the targeting advantage depends on.

Run: .venv/bin/python -m experiments.e10_instrument_ablation
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
from src.bench.policies import EvidentialCapture, FixedRetake, SingleShot, UntargetedEvidential
from src.bench.runner import fit_calibrator, run_docket
from src.data.degradation import DEGRADATION_NAMES
from src.evidence.calibration import LikelihoodRatioCalibrator
from src.models.diagnostic import Case, DiagnosticChannel, Reading, SurrogateChannel, predicted_usability

N_CASES = 4000
BUDGET = 4


@dataclass
class BlurVarianceChannel(DiagnosticChannel):
    """Diagnosis channel unchanged; degradation channel = one scalar, broadcast.

    A real Laplacian-variance blur detector responds to loss of
    high-frequency image content -- which `blur` and `jpeg` cause and
    `glare`, `low_light`, `angle` do not -- and returns a single sharpness
    number with no per-artifact breakdown. Reporting that one number for all
    five slots is what "use this heuristic as if it were the confidence
    head" looks like: `predicted_usability`'s worst/mean aggregate still
    responds correctly to overall quality, but `EvidenceView.
    dominant_degradation` (what a subpoena targets) becomes uninformative --
    whichever artifact happens to sort first among five identical values.
    """

    inner: SurrogateChannel
    noise: float = 0.12
    name: str = "blur_variance_heuristic"

    def read(self, case: Case, severities: dict[str, float], rng: np.random.Generator) -> Reading:
        inner_reading = self.inner.read(case, severities, rng)
        sharpness_loss = 0.6 * severities.get("blur", 0.0) + 0.4 * severities.get("jpeg", 0.0)
        est = float(np.clip(sharpness_loss + rng.normal(0.0, self.noise), 0.0, 1.0))
        degradation = {name: est for name in DEGRADATION_NAMES}
        return Reading(
            score=inner_reading.score,
            degradation=degradation,
            usability=predicted_usability(degradation),
            true_quality=inner_reading.true_quality,
        )


def run_arms(channel, calibrator, docket) -> list:
    arms = {
        "single_shot": SingleShot(),
        "fixed_retake": FixedRetake(),
        "untargeted_evidential": UntargetedEvidential(),
        "evidential_capture": EvidentialCapture(),
    }
    return [score_results(name, run_docket(docket, pol, channel, calibrator), docket.burden) for name, pol in arms.items()]


def targeting_gain(rows) -> float:
    by_name = {r.policy: r for r in rows}
    return by_name["evidential_capture"].verdicts_per_capture - by_name["untargeted_evidential"].verdicts_per_capture


def main() -> None:
    world = build_world()
    banner("E10 -- instrument ablation: a trivial blur-variance heuristic", world)

    docket = make_docket(
        "instrument_ablation", n_cases=N_CASES, prevalence=PREVALENCE, budget=BUDGET,
        burden=HEADLINE_BURDEN, clinic_difficulty=CLINIC_DIFFICULTY, seed=41,
    )

    print("\n[baseline] full per-artifact confidence head (SurrogateChannel as-is)")
    lr_full = LikelihoodRatioCalibrator(n_strata=world.calibrator.n_strata).fit(
        world.calibration_data.scores, world.calibration_data.labels, world.calibration_data.usabilities
    )
    rows_full = run_arms(world.channel, lr_full, docket)
    print(format_leaderboard(rows_full))
    gain_full = targeting_gain(rows_full)

    print("\n[ablated] blur-variance heuristic (one scalar, broadcast to all five slots)")
    heuristic_channel = BlurVarianceChannel(inner=world.channel)
    lr_heuristic, heuristic_cal_data = fit_calibrator(
        heuristic_channel, n_strata=world.calibrator.n_strata, calibrator_cls=LikelihoodRatioCalibrator,
        n=8000, prevalence=PREVALENCE, clinic_difficulty=CLINIC_DIFFICULTY, seed=42,
    )
    rows_heuristic = run_arms(heuristic_channel, lr_heuristic, docket)
    print(format_leaderboard(rows_heuristic))
    gain_heuristic = targeting_gain(rows_heuristic)

    print(
        f"\ntargeting gain (evidential_capture VPC - untargeted VPC):"
        f"\n  full confidence head : {gain_full:+.4f}"
        f"\n  blur-variance heuristic: {gain_heuristic:+.4f}"
        f"\n  retained fraction    : {(gain_heuristic / gain_full if gain_full else float('nan')):.1%}"
    )
    if gain_heuristic > 0.8 * gain_full:
        verdict = (
            "ORDERING SURVIVES largely intact -- most of the targeting advantage does not "
            "require a type-predicting head, only a scalar quality gate. The learned-"
            "confidence-head claim needs softening: a much cheaper instrument gets most of the way there."
        )
    elif gain_heuristic > 0.2 * gain_full:
        verdict = (
            "ORDERING PARTIALLY SURVIVES -- targeting still helps with a scalar-only "
            "instrument, but noticeably less than with per-artifact type prediction. "
            "The head's type prediction is doing real, but not all, of the work."
        )
    else:
        verdict = (
            "ORDERING COLLAPSES -- targeting's advantage depends on knowing WHICH artifact "
            "dominates, not just that the shot is degraded. The learned confidence head's "
            "type prediction, not just its scalar quality signal, is load-bearing."
        )
    print(f"\n{verdict}")

    save_table("e10_instrument_ablation", [r.as_dict() for r in rows_full] + [r.as_dict() for r in rows_heuristic])
    save_results("e10_instrument_ablation", {
        "reader": {"clean_auc": world.clean_auc, "clinic_auc": world.clinic_auc},
        "docket": {"n_cases": N_CASES, "budget": BUDGET, "burden": HEADLINE_BURDEN.name},
        "full_head": [r.as_dict() for r in rows_full],
        "blur_variance_heuristic": [r.as_dict() for r in rows_heuristic],
        "targeting_gain_full_head": gain_full,
        "targeting_gain_heuristic": gain_heuristic,
        "retained_fraction": gain_heuristic / gain_full if gain_full else float("nan"),
        "verdict": verdict,
    })
    print("results -> results/e10_instrument_ablation.json")


if __name__ == "__main__":
    main()
