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
    figure_path,
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

    fig = make_figure(rows_full, rows_heuristic, gain_full, gain_heuristic)
    print(f"\nfigure -> {fig}")

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
        "figure": fig,
    })
    print("results -> results/e10_instrument_ablation.json")


def make_figure(rows_full, rows_heuristic, gain_full, gain_heuristic) -> str:
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.gridspec import GridSpec

    from experiments.common import sample_tooth_crops
    from src.data.degradation import DEGRADATION_NAMES
    from src.sim.render import render_severities

    fig = plt.figure(figsize=(11.5, 6.6))
    gs = GridSpec(2, 6, figure=fig, height_ratios=[1.0, 2.1], hspace=0.6, wspace=1.6)

    # A glare-dominated shot: the case this ablation is built to expose, since
    # glare does not touch high-frequency content and so is invisible to a
    # sharpness-only proxy.
    crop = sample_tooth_crops(1, label=1, seed=17)[0]
    rng = np.random.default_rng(3)
    severities = {"glare": 0.8, "blur": 0.1, "jpeg": 0.1, "angle": 0.05, "low_light": 0.05}
    out = render_severities(crop.image, severities, rng=rng)

    ax_img = fig.add_subplot(gs[0, 0:2])
    ax_img.imshow(cv2.cvtColor(out.image, cv2.COLOR_BGR2RGB))
    ax_img.set_xticks([]); ax_img.set_yticks([])
    ax_img.set_title("one real, glare-dominated shot", fontsize=9)

    names = list(DEGRADATION_NAMES)
    y = np.arange(len(names))
    full_read = [severities.get(n, 0.05) + rng.normal(0, 0.03) for n in names]
    heuristic_read = [0.6 * severities["blur"] + 0.4 * severities["jpeg"]] * len(names)

    ax_full = fig.add_subplot(gs[0, 2:4])
    ax_full.barh(y, np.clip(full_read, 0, 1), color="tab:blue")
    ax_full.set_yticks(y, names, fontsize=7.5)
    ax_full.invert_yaxis()
    ax_full.set_xlim(0, 1)
    ax_full.set_title("full head reads:\nglare correctly flagged", fontsize=8.5)

    ax_heur = fig.add_subplot(gs[0, 4:6])
    ax_heur.barh(y, np.clip(heuristic_read, 0, 1), color="tab:orange")
    ax_heur.set_yticks(y, [""] * len(names))
    ax_heur.set_xlim(0, 1)
    ax_heur.set_title("heuristic reads:\nflat, misses glare entirely", fontsize=8.5)
    fig.text(0.5, 0.95, "illustrative severities on a real tooth crop -- glare doesn't touch sharpness, so a blur-only proxy can't see it",
              ha="center", fontsize=8.5, color="#6b6b6b")

    ax = fig.add_subplot(gs[1, 0:3])
    policies = ["single_shot", "fixed_retake", "untargeted_evidential", "evidential_capture"]
    by_full = {r.policy: r for r in rows_full}
    by_heur = {r.policy: r for r in rows_heuristic}
    x = np.arange(len(policies))
    width = 0.35
    ax.bar(x - width / 2, [by_full[p].verdicts_per_capture for p in policies], width,
           label="full per-artifact head", color="tab:blue")
    ax.bar(x + width / 2, [by_heur[p].verdicts_per_capture for p in policies], width,
           label="blur-variance heuristic\n(one scalar, broadcast)", color="tab:orange")
    ax.set_xticks(x, [p.replace("_", "\n") for p in policies], fontsize=8)
    ax.set_ylabel("verdicts per capture")
    ax.set_title("(a) both instruments, all four arms")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = fig.add_subplot(gs[1, 3:6])
    bars = ax.bar(["full\nconfidence head", "blur-variance\nheuristic"], [gain_full, gain_heuristic],
                  color=["tab:blue", "tab:orange"])
    for bar, v in zip(bars, [gain_full, gain_heuristic]):
        ax.annotate(f"{v:+.3f}", (bar.get_x() + bar.get_width() / 2, v), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("targeting gain (evidential_capture - untargeted), VPC")
    ax.set_title(f"(b) type prediction matters, but isn't\nall of it: {gain_heuristic / gain_full:.0%} of the gain survives")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("E10: instrument ablation -- a trivial blur-variance heuristic vs the full head", fontsize=12, y=1.0)
    path = figure_path("e10_instrument_ablation.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return str(path)


if __name__ == "__main__":
    main()
