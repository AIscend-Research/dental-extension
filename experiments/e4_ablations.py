"""E4 -- ablations: which parts of the framework are load-bearing?

Six knobs, each turned independently while everything else is held at the
headline configuration. Two of them are the ones worth reading first:

  PERSISTENCE. `iid_scene_per_shot` redraws the whole scene every shot, which
  is the world an ImageNet-C-style benchmark implicitly assumes: each retake
  is an independent sample from the corruption distribution. If the gains from
  targeted instructions shrink there, then benchmarking capture policies on
  i.i.d. corruptions understates them, and every retake-loop method evaluated
  that way has been measured in the wrong world. This is the ablation that
  justifies building a process simulator at all.

  Note that lowering `persistence` alone does NOT produce that world -- the
  per-session equilibrium survives, so the scene still relaxes toward the same
  room every shot. The rho sweep is reported alongside precisely to show that
  it is nearly flat, and that the i.i.d. arm is the one doing the work.

  EVIDENCE CONSTRUCTION. Conformal-calibrated e-values are provably valid but
  lossy; likelihood-ratio e-values are more powerful but rest on an estimated
  density; a marginal (unstratified) calibrator is the thing most people would
  build by default. E2 already showed all three stay under their nominal error
  rate on the null -- so the question here is not validity but *power*, which
  is where the marginal calibrator is expected to pay for ignoring the
  degradation channel.

The rest: betting strategy, confidence-head quality, instruction side effects,
and stratum count.

Run: .venv/bin/python -m experiments.e4_ablations
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
from src.evidence.calibration import (
    LikelihoodRatioCalibrator,
    MarginalCalibrator,
    StratifiedCalibrator,
)
from src.evidence.ewealth import ConstantBet, DegradationAwareBet
from src.evidence.ladder import BurdenSpec, CLEAR_AND_CONVINCING, PREPONDERANCE
from src.sim.session import SessionConfig

N_CASES = 3000
BUDGET = 4


def docket(seed=21, burden=None, budget=BUDGET):
    return make_docket(
        "ablation", n_cases=N_CASES, prevalence=PREVALENCE, budget=budget,
        burden=burden or HEADLINE_BURDEN, clinic_difficulty=CLINIC_DIFFICULTY, seed=seed,
    )


def row(label, group, r, extra=None):
    d = {
        "ablation": group,
        "setting": label,
        "verdicts_per_capture": r.verdicts_per_capture,
        "verdict_rate": r.verdict_rate,
        "verdict_accuracy": r.verdict_accuracy,
        "refer_rate": r.refer_rate,
        "mean_captures": r.mean_captures,
        "false_conviction_rate": r.false_conviction_rate,
        "false_discharge_rate": r.false_discharge_rate,
        "violated": bool(r.convict_violation or r.discharge_violation),
    }
    d.update(extra or {})
    return d


def show(rows, group):
    print(f"\n[{group}]")
    print(f"  {'setting':<38} {'VPC':>7} {'decided':>8} {'acc':>7} {'shots':>7}  flags")
    for r in rows:
        flag = "VIOLATED" if r["violated"] else ""
        print(
            f"  {r['setting']:<38} {r['verdicts_per_capture']:>7.3f} "
            f"{r['verdict_rate']*100:>7.1f}% {r['verdict_accuracy']:>7.3f} "
            f"{r['mean_captures']:>7.2f}  {flag}"
        )


def main() -> None:
    world = build_world()
    banner("E4 -- ablations", world)
    all_rows = []
    d = docket()

    def fit(cls, **kw):
        return cls(n_strata=kw.pop("n_strata", world.calibrator.n_strata), **kw).fit(
            world.calibration_data.scores,
            world.calibration_data.labels,
            world.calibration_data.usabilities,
        )

    lr = fit(LikelihoodRatioCalibrator)

    # -- 1. evidence construction ------------------------------------------
    rows = []
    for label, cal in [
        ("conformal, stratified (provable)", world.calibrator),
        ("likelihood ratio, stratified", lr),
        ("conformal, marginal (unstratified)", fit(MarginalCalibrator)),
    ]:
        r = score_results("evidential_capture", run_docket(d, EvidentialCapture(), world.channel, cal), d.burden)
        rows.append(row(label, "evidence construction", r))
    show(rows, "1. evidence construction (power, not validity -- see E2)")
    all_rows += rows

    # -- 2. betting strategy ------------------------------------------------
    rows = []
    for label, strat in [
        ("degradation-aware stake", DegradationAwareBet()),
        ("constant stake 0.5", ConstantBet(0.5)),
        ("constant stake 0.8", ConstantBet(0.8)),
        ("constant stake 1.0 (all-in)", ConstantBet(1.0)),
    ]:
        r = score_results(
            "evidential_capture",
            run_docket(d, EvidentialCapture(strategy=strat), world.channel, lr), d.burden,
        )
        rows.append(row(label, "betting strategy", r))
    show(rows, "2. betting strategy -- does staking on image quality pay?")
    all_rows += rows

    # -- 3. confidence-head quality ----------------------------------------
    rows = []
    for head_noise in [0.0, 0.06, 0.12, 0.25, 0.40]:
        w = build_world(head_noise=head_noise)
        cal = LikelihoodRatioCalibrator(n_strata=w.calibrator.n_strata).fit(
            w.calibration_data.scores, w.calibration_data.labels, w.calibration_data.usabilities
        )
        r = score_results("evidential_capture", run_docket(d, EvidentialCapture(), w.channel, cal), d.burden)
        label = "oracle head (noise 0.00)" if head_noise == 0 else f"head noise {head_noise:.2f}"
        rows.append(row(label, "confidence head quality", r, {"head_noise": head_noise}))
    show(rows, "3. confidence-head quality -- everything keys off this channel")
    all_rows += rows

    # -- 4. instruction side effects ---------------------------------------
    rows = []
    for label, scale in [
        ("side effects off (coupling 0.0)", 0.0),
        ("half strength (0.5)", 0.5),
        ("as specified (1.0)", 1.0),
        ("double (2.0)", 2.0),
    ]:
        cfg = SessionConfig(coupling_scale=scale)
        w = build_world(config=cfg)
        cal = LikelihoodRatioCalibrator(n_strata=w.calibrator.n_strata).fit(
            w.calibration_data.scores, w.calibration_data.labels, w.calibration_data.usabilities
        )
        r = score_results(
            "evidential_capture",
            run_docket(d, EvidentialCapture(), w.channel, cal, config=cfg), d.burden,
        )
        rows.append(row(label, "instruction side effects", r, {"coupling_scale": scale}))
    show(rows, "4. instruction side effects -- the cost of fixing one thing")
    all_rows += rows

    # -- 5. persistence: process vs i.i.d. filter ---------------------------
    rows = []
    for label, rho, iid in [
        ("i.i.d. corruption filter", 0.85, True),
        ("shot-to-shot noise only (rho 0.00)", 0.0, False),
        ("weak memory (0.40)", 0.4, False),
        ("as specified (0.85)", 0.85, False),
        ("near-frozen scene (0.98)", 0.98, False),
    ]:
        cfg = SessionConfig(persistence=rho, iid_scene_per_shot=iid)
        w = build_world(config=cfg)
        cal = LikelihoodRatioCalibrator(n_strata=w.calibrator.n_strata).fit(
            w.calibration_data.scores, w.calibration_data.labels, w.calibration_data.usabilities
        )
        arms = {}
        for name, pol in [
            ("single_shot", SingleShot()),
            ("fixed_retake", FixedRetake()),
            ("untargeted_evidential", UntargetedEvidential()),
            ("evidential_capture", EvidentialCapture()),
        ]:
            arms[name] = score_results(name, run_docket(d, pol, w.channel, cal, config=cfg), d.burden)
        r = arms["evidential_capture"]
        rows.append(row(label, "persistence", r, {
            "persistence": rho,
            "iid_scene_per_shot": iid,
            "vpc_single_shot": arms["single_shot"].verdicts_per_capture,
            "vpc_fixed_retake": arms["fixed_retake"].verdicts_per_capture,
            "vpc_untargeted": arms["untargeted_evidential"].verdicts_per_capture,
            "targeting_gain": r.verdicts_per_capture - arms["untargeted_evidential"].verdicts_per_capture,
        }))
    show(rows, "5. persistence -- would an i.i.d. corruption benchmark mislead us?")
    print("     (targeting gain = evidential_capture VPC minus untargeted VPC)")
    for r in rows:
        print(
            f"     {r['setting']:<38} targeting gain {r['targeting_gain']:+.4f} "
            f"| untargeted {r['vpc_untargeted']:.3f} | fixed {r['vpc_fixed_retake']:.3f}"
        )
    all_rows += rows

    # -- 6. stratum count ---------------------------------------------------
    rows = []
    for n_strata in [1, 2, 4, 8, 16]:
        w = build_world(n_strata=n_strata)
        cal = LikelihoodRatioCalibrator(n_strata=n_strata).fit(
            w.calibration_data.scores, w.calibration_data.labels, w.calibration_data.usabilities
        )
        r = score_results(
            "evidential_capture",
            run_docket(d, EvidentialCapture(), w.channel, cal, n_strata=n_strata), d.burden,
        )
        rows.append(row(f"{n_strata} strata", "stratum count", r, {
            "n_strata": n_strata,
            "fallback_strata": sorted(cal.fallback_strata),
        }))
    show(rows, "6. stratum count -- resolution vs calibration points per stratum")
    all_rows += rows

    fig = make_figure(all_rows)
    print(f"\nfigure -> {fig}")
    save_table("e4_ablations", all_rows)
    save_results("e4_ablations", {
        "reader": {"clean_auc": world.clean_auc, "clinic_auc": world.clinic_auc},
        "docket": {"n_cases": N_CASES, "budget": BUDGET, "burden": HEADLINE_BURDEN.name},
        "rows": all_rows,
        "figure": fig,
    })
    print("results -> results/e4_ablations.json")


def make_figure(rows) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # (a) persistence: the headline ablation
    ax = axes[0]
    pr = [r for r in rows if r["ablation"] == "persistence"]
    labels = [r["setting"].split(" (")[0] for r in pr]
    x = np.arange(len(pr))
    width = 0.2
    for k, (key, name) in enumerate([
        ("vpc_single_shot", "single_shot"),
        ("vpc_fixed_retake", "fixed_retake"),
        ("vpc_untargeted", "untargeted_evidential"),
        ("verdicts_per_capture", "evidential_capture"),
    ]):
        ax.bar(x + (k - 1.5) * width, [r[key] for r in pr], width, label=name)
    ax.set_xticks(x, [l.replace(" ", "\n") for l in labels], fontsize=6.5)
    ax.set_ylabel("verdicts per capture")
    ax.set_title("(a) the world you benchmark in\nchanges the answer")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    # (b) confidence-head quality
    ax = axes[1]
    hr = [r for r in rows if r["ablation"] == "confidence head quality"]
    ax.plot([r["head_noise"] for r in hr], [r["verdicts_per_capture"] for r in hr], "o-")
    ax.set_xlabel("confidence-head noise")
    ax.set_ylabel("verdicts per capture")
    ax.set_title("(b) everything keys off\nthe degradation channel")
    ax.grid(alpha=0.3)

    # (c) evidence construction + betting
    ax = axes[2]
    sel = [r for r in rows if r["ablation"] in ("evidence construction", "betting strategy")]
    y = np.arange(len(sel))
    colors = ["tab:blue" if r["ablation"] == "evidence construction" else "tab:orange" for r in sel]
    ax.barh(y, [r["verdicts_per_capture"] for r in sel], color=colors)
    ax.set_yticks(y, [r["setting"] for r in sel], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("verdicts per capture")
    ax.set_title("(c) evidence construction (blue)\nand stake rule (orange)")
    ax.grid(alpha=0.3, axis="x")

    fig.suptitle("E4: which parts are load-bearing", fontsize=12)
    fig.tight_layout()
    path = figure_path("e4_ablations.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


if __name__ == "__main__":
    main()
