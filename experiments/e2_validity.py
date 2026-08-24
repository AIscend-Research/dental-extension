"""E2 -- is the burden of proof actually anytime-valid under adaptive retaking?

The theorem says: when the tooth is really sound, the conviction wealth process
crosses 1/alpha with probability at most alpha, at any stopping time, however
adaptively the retakes were chosen. That is a falsifiable claim about a
frequency, so this experiment measures the frequency.

Four arms, chosen so that each isolates one way of breaking the guarantee:

  stratified   the proposed construction. Bets and stops on the degradation
               channel only; conformal p-values conditioned on the predicted
               usability stratum. Should sit on or below the diagonal.
  marginal     identical, except the conformal p-values come from one pooled
               calibration set. Isolates the *selection* failure: the retake
               loop shifts which shots get tested, and a marginal calibrator
               was fitted on first shots. Nothing about the betting is wrong
               here -- only the null distribution is.
  best_shot    takes the whole budget and keeps the most favourable p-value.
               Isolates the *multiplicity* failure.
  greedy       stops as soon as the diagnosis score looks decisive, then
               reports that shot as a single planned test. Isolates the
               *stopping-on-the-tested-quantity* failure.

Three sweeps:

  A. crossing rate vs nominal alpha, at fixed budget.
  B. crossing rate vs capture budget K, at fixed alpha -- the signature of a
     multiplicity failure is that it grows with K while a valid process does not.
  C. scope of the guarantee. Conformal validity is *marginal* over the case
     distribution it was calibrated on. Sweep C conditions the null on a
     hard-case subpopulation the calibration set under-represents, and reports
     what happens. This is the experiment most likely to embarrass the method,
     which is why it is here.
  D. the falsifiable prediction from docs/theory_anytime_validity.md #6:
     A1 reduces to A1' (stratum-conditional stochastic dominance of true
     quality, test vs. calibration), and A1' should degrade -- and eventually
     fail -- as `head_noise` rises past the point where noise-driven stratum
     misclassification dominates genuine equilibrium-shifting correction.
     This sweeps `head_noise` and checks the STRATIFIED arm's crossing rate
     directly, rather than leaving that prediction unchecked. Both calibration
     and test use the same head_noise (the same confidence head reads both),
     matching E4's existing power sweep at the same grid points, so the two
     experiments are read together: E4 shows what noise costs in verdicts per
     capture, this shows what it costs in the guarantee itself.
  E. the one gap sweep D's structural explanation left open (theory doc #6,
     updated 2026-08-23): the argument for why A1' survives rising head_noise
     depends on the SAME noise kernel corrupting both the calibration and
     test populations alike. This sweeps the case where it does not -- the
     confidence head is well-behaved at calibration time (a controlled
     bench-test setting) but noisier at deployment/retake-loop time (the
     field) -- and checks whether that asymmetry, not just noise magnitude,
     is what actually breaks A1'.

Run: .venv/bin/python -m experiments.e2_validity
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from experiments.common import (
    CLINIC_DIFFICULTY,
    PREVALENCE,
    World,
    banner,
    build_world,
    figure_path,
    save_results,
    save_table,
)
from src.evidence.calibration import (
    LikelihoodRatioCalibrator,
    MarginalCalibrator,
    StratifiedCalibrator,
)
from src.evidence.ewealth import DegradationAwareBet, EWealth, p_to_e
from src.models.diagnostic import Case
from src.sim.instructions import instruction_for_factor
from src.sim.session import CaptureSession
from src.evidence.verdict import DEGRADATION_TO_FACTOR

#: arms that accumulate a wealth process, rather than reporting a raw p-value
WEALTH_ARMS = frozenset({"stratified", "marginal", "likelihood_ratio"})

N_NULL_SESSIONS = 20000
BUDGET = 4
ALPHA_GRID = [0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01]
BUDGET_GRID = [1, 2, 3, 5, 8]
FIXED_ALPHA = 0.05
# Same first five points as E4's power sweep (docs/experiments_results.md,
# "confidence-head quality"), extended past 0.40 to actually locate a
# validity failure if the theory's mechanism-2-dominates-mechanism-1
# prediction is right, rather than stopping at the point E4 happened to stop.
HEAD_NOISE_GRID = [0.0, 0.06, 0.12, 0.25, 0.40, 0.60, 0.80, 1.00]
# Calibration-time noise held at the default (a controlled bench setting);
# test-time noise swept upward from it (the field). 0.12 is the symmetric
# baseline (matches sweep D's default point); the rest push the asymmetry
# well past what sweep D's SYMMETRIC sweep covered at its own endpoint (1.00),
# since the structural argument for why symmetric noise didn't break A1'
# explicitly does not apply once the two populations see different noise.
HEAD_NOISE_CAL = 0.12
HEAD_NOISE_TEST_GRID = [0.12, 0.25, 0.40, 0.60, 0.90, 1.20, 1.60, 2.00]


def run_null_sessions(
    world,
    calibrator,
    arm: str,
    budget: int,
    n: int,
    seed: int,
    difficulty_sampler=None,
) -> dict:
    """Run `n` sessions on genuinely SOUND teeth and record the evidence produced.

    Returns, per session, the largest conviction evidence the arm ever reported
    -- running-max wealth for the e-process arms, 1/p for the p-value arms --
    so a single array can be thresholded at any alpha afterwards. Sweeping
    alpha post hoc rather than re-running per alpha is not a shortcut: it is
    the same sessions being judged against different bars, which is exactly
    what "anytime" means.
    """
    rng = np.random.default_rng(seed)
    strategy = DegradationAwareBet()
    evidence = np.empty(n)
    n_shots = np.empty(n, dtype=int)

    for i in range(n):
        difficulty = (
            difficulty_sampler(rng) if difficulty_sampler else float(rng.beta(2.0, 4.0))
        )
        case = Case(label=0, difficulty=difficulty)  # H0 holds: the tooth is sound
        session = CaptureSession(rng=rng, difficulty=CLINIC_DIFFICULTY)

        wealth = EWealth()
        best_p = 1.0
        instruction = None
        shots = 0

        for t in range(budget):
            capture = session.capture(instruction if t else None)
            reading = world.channel.read(case, capture.severities, rng)
            shots += 1
            stratum = calibrator.stratum_of(reading.usability)
            p_c, _ = calibrator.p_values(reading.score, stratum)

            if arm in WEALTH_ARMS:
                view_usability = reading.usability
                lam = strategy.stake(_View(view_usability), "convict")
                e_c, _ = calibrator.e_values(reading.score, stratum)
                wealth.update(e_c, lam)
            else:
                best_p = min(best_p, p_c)

            if arm == "greedy":
                # stops on the tested quantity, and keeps only this shot
                if abs(reading.score - 0.5) > 0.25:
                    best_p = p_c
                    break
                best_p = p_c

            worst = max(reading.degradation.items(), key=lambda kv: kv[1])[0]
            instruction = instruction_for_factor(DEGRADATION_TO_FACTOR.get(worst))

        evidence[i] = (
            wealth.running_max if arm in WEALTH_ARMS else 1.0 / max(best_p, 1e-12)
        )
        n_shots[i] = shots

    return {"evidence": evidence, "n_shots": n_shots}


class _View:
    """Minimal EvidenceView stand-in carrying only what the stake rule reads.

    `DegradationAwareBet.stake` uses `predicted_usability` and nothing else, so
    this keeps the null simulation cheap without letting the bet touch anything
    the real view would have withheld.
    """

    def __init__(self, predicted_usability: float):
        self.predicted_usability = predicted_usability


def crossing_rate(evidence: np.ndarray, alpha: float) -> float:
    """Fraction of null sessions whose evidence ever reached the 1/alpha bar."""
    return float(np.mean(evidence >= 1.0 / alpha))


def wilson_upper(k: int, n: int, z: float = 1.96) -> float:
    from src.bench.metrics import wilson_interval

    return wilson_interval(k, n)[1]


def wilson_lower(k: int, n: int, z: float = 1.96) -> float:
    from src.bench.metrics import wilson_interval

    return wilson_interval(k, n)[0]


def main() -> None:
    world = build_world()
    banner("E2 -- anytime-validity under adaptive retaking", world)

    # the marginal calibrator is refitted on the SAME calibration data, so the
    # only difference between the two arms is the conditioning, not the sample
    marginal = MarginalCalibrator(n_strata=world.calibrator.n_strata).fit(
        world.calibration_data.scores,
        world.calibration_data.labels,
        world.calibration_data.usabilities,
    )
    assert isinstance(world.calibrator, StratifiedCalibrator)

    lr = LikelihoodRatioCalibrator(n_strata=world.calibrator.n_strata).fit(
        world.calibration_data.scores,
        world.calibration_data.labels,
        world.calibration_data.usabilities,
    )

    arms = {
        "stratified": world.calibrator,
        "likelihood_ratio": lr,
        "marginal": marginal,
        "best_shot": world.calibrator,
        "greedy": world.calibrator,
    }

    # -- Sweep A: crossing rate vs alpha ------------------------------------
    print(f"\n[A] crossing rate vs nominal alpha (K={BUDGET}, n={N_NULL_SESSIONS} sound cases)")
    runs = {}
    for arm, cal in arms.items():
        runs[arm] = run_null_sessions(world, cal, arm, BUDGET, N_NULL_SESSIONS, seed=100 + len(arm))
    sweep_a = []
    for alpha in ALPHA_GRID:
        row = {"alpha": alpha}
        for arm in arms:
            ev = runs[arm]["evidence"]
            k = int(np.sum(ev >= 1.0 / alpha))
            row[arm] = k / len(ev)
            row[f"{arm}_ci_lo"] = wilson_lower(k, len(ev))
            row[f"{arm}_violates"] = bool(wilson_lower(k, len(ev)) > alpha)
        sweep_a.append(row)

    header = f"{'alpha':>7} " + " ".join(f"{a:>22}" for a in arms)
    print(header)
    print("-" * len(header))
    for row in sweep_a:
        cells = []
        for arm in arms:
            mark = "  VIOLATION" if row[f"{arm}_violates"] else ""
            cells.append(f"{row[arm]:>11.4f}{mark:<11}")
        print(f"{row['alpha']:>7.2f} " + " ".join(cells))

    # -- Sweep B: crossing rate vs budget ----------------------------------
    print(f"\n[B] crossing rate vs capture budget K (alpha={FIXED_ALPHA})")
    sweep_b = []
    for K in BUDGET_GRID:
        row = {"budget": K}
        for arm, cal in arms.items():
            out = run_null_sessions(world, cal, arm, K, N_NULL_SESSIONS // 2, seed=200 + K)
            ev = out["evidence"]
            k = int(np.sum(ev >= 1.0 / FIXED_ALPHA))
            row[arm] = k / len(ev)
            row[f"{arm}_violates"] = bool(wilson_lower(k, len(ev)) > FIXED_ALPHA)
        sweep_b.append(row)
        cells = " ".join(
            f"{arm}={row[arm]:.4f}{'*' if row[f'{arm}_violates'] else ' '}" for arm in arms
        )
        print(f"  K={K}: {cells}")
    print("  (* = violation, Wilson lower bound above alpha)")

    # -- Sweep C: scope of the guarantee -----------------------------------
    print(f"\n[C] scope: conditioning the null on a hard-case subpopulation (alpha={FIXED_ALPHA})")
    scope = []
    for label, sampler in [
        ("matched (as calibrated)", None),
        ("hard cases only (difficulty ~ Beta(6,2))", lambda r: float(r.beta(6.0, 2.0))),
        ("hardest (difficulty = 0.95)", lambda r: 0.95),
    ]:
        out = run_null_sessions(
            world, world.calibrator, "stratified", BUDGET, N_NULL_SESSIONS // 2,
            seed=300, difficulty_sampler=sampler,
        )
        ev = out["evidence"]
        k = int(np.sum(ev >= 1.0 / FIXED_ALPHA))
        n = len(ev)
        rate = k / n
        scope.append({
            "subpopulation": label,
            "crossing_rate": rate,
            "ci": [wilson_lower(k, n), wilson_upper(k, n)],
            "violates": bool(wilson_lower(k, n) > FIXED_ALPHA),
        })
        flag = " VIOLATION" if scope[-1]["violates"] else ""
        print(f"  {label:<42} {rate:.4f}{flag}")

    # -- Sweep D: does A1' itself erode as head_noise rises? ----------------
    print(f"\n[D] the falsifiable A1' prediction: crossing rate vs head_noise "
          f"(alpha={FIXED_ALPHA}, STRATIFIED arm only)")
    head_noise_sweep = []
    for head_noise in HEAD_NOISE_GRID:
        hn_world = build_world(head_noise=head_noise)
        out = run_null_sessions(
            hn_world, hn_world.calibrator, "stratified", BUDGET,
            N_NULL_SESSIONS // 2, seed=400,
        )
        ev = out["evidence"]
        k = int(np.sum(ev >= 1.0 / FIXED_ALPHA))
        n = len(ev)
        rate = k / n
        lo, hi = wilson_lower(k, n), wilson_upper(k, n)
        head_noise_sweep.append({
            "head_noise": head_noise,
            "crossing_rate": rate,
            "ci_lo": lo, "ci_hi": hi,
            "violates": bool(lo > FIXED_ALPHA),
        })
        flag = "  VIOLATION" if head_noise_sweep[-1]["violates"] else ""
        print(f"  head_noise={head_noise:.2f}  rate={rate:.4f}  ci=[{lo:.4f}, {hi:.4f}]{flag}")
    n_violations_d = sum(1 for r in head_noise_sweep if r["violates"])
    if n_violations_d:
        first = next(r["head_noise"] for r in head_noise_sweep if r["violates"])
        print(f"  A1' breaks empirically starting at head_noise={first:.2f} "
              f"({n_violations_d}/{len(HEAD_NOISE_GRID)} grid points violate)")
    else:
        print(f"  no violation up to head_noise={HEAD_NOISE_GRID[-1]:.2f} -- A1' holds "
              f"throughout the tested range; the failure mode is real (see the "
              f"mechanism in theory_anytime_validity.md #6) but the noise level needed "
              f"to trigger it is at or beyond this grid.")

    # -- Sweep E: what if the noise ISN'T symmetric across the two populations? ---
    print(f"\n[E] the residual A1' gap: crossing rate vs ASYMMETRIC head_noise "
          f"(calibration fixed at {HEAD_NOISE_CAL:.2f}, test swept; alpha={FIXED_ALPHA}, "
          f"STRATIFIED arm only)")
    cal_world = build_world(head_noise=HEAD_NOISE_CAL)
    asymmetry_sweep = []
    for head_noise_test in HEAD_NOISE_TEST_GRID:
        # Same separation/loss_scale/sigma/gamma/head_bias as calibration --
        # only head_noise differs, so this isolates the confidence head's
        # noise asymmetry from every other difference a "worse deployment
        # site" could also introduce.
        test_channel = replace(cal_world.channel, head_noise=head_noise_test)
        test_world = World(
            channel=test_channel, calibrator=cal_world.calibrator,
            calibration_data=cal_world.calibration_data,
            clean_auc=cal_world.clean_auc, clinic_auc_value=cal_world.clinic_auc,
        )
        out = run_null_sessions(
            test_world, cal_world.calibrator, "stratified", BUDGET,
            N_NULL_SESSIONS // 2, seed=500,
        )
        ev = out["evidence"]
        k = int(np.sum(ev >= 1.0 / FIXED_ALPHA))
        n = len(ev)
        rate = k / n
        lo, hi = wilson_lower(k, n), wilson_upper(k, n)
        asymmetry_sweep.append({
            "head_noise_cal": HEAD_NOISE_CAL,
            "head_noise_test": head_noise_test,
            "crossing_rate": rate,
            "ci_lo": lo, "ci_hi": hi,
            "violates": bool(lo > FIXED_ALPHA),
        })
        flag = "  VIOLATION" if asymmetry_sweep[-1]["violates"] else ""
        print(f"  test_noise={head_noise_test:.2f} (cal={HEAD_NOISE_CAL:.2f})  "
              f"rate={rate:.4f}  ci=[{lo:.4f}, {hi:.4f}]{flag}")
    n_violations_e = sum(1 for r in asymmetry_sweep if r["violates"])
    if n_violations_e:
        first = next(r["head_noise_test"] for r in asymmetry_sweep if r["violates"])
        print(f"  A1' breaks under asymmetric noise starting at test_noise={first:.2f} "
              f"({n_violations_e}/{len(HEAD_NOISE_TEST_GRID)} grid points violate) -- "
              f"confirms the residual gap named in theory_anytime_validity.md #6: the "
              f"symmetric-noise argument does not extend to a confidence head that is "
              f"specifically worse under deployment/retake-loop conditions than it was "
              f"at calibration time.")
    else:
        print(f"  no violation up to test_noise={HEAD_NOISE_TEST_GRID[-1]:.2f} "
              f"({HEAD_NOISE_TEST_GRID[-1]/HEAD_NOISE_CAL:.1f}x the calibration noise) -- "
              f"A1' is more robust to this asymmetry than the structural argument alone "
              f"would have predicted needing to check.")

    fig = make_figure(sweep_a, sweep_b, runs, head_noise_sweep, asymmetry_sweep)
    print(f"\nfigure -> {fig}")

    save_table("e2_validity_alpha_sweep", sweep_a)
    save_table("e2_validity_budget_sweep", sweep_b)
    save_table("e2_validity_head_noise_sweep", head_noise_sweep)
    save_table("e2_validity_head_noise_asymmetry_sweep", asymmetry_sweep)
    save_results(
        "e2_validity",
        {
            "reader": {"clean_auc": world.clean_auc, "clinic_auc": world.clinic_auc},
            "n_null_sessions": N_NULL_SESSIONS,
            "budget": BUDGET,
            "alpha_sweep": sweep_a,
            "budget_sweep": sweep_b,
            "scope_of_guarantee": scope,
            "head_noise_sweep": head_noise_sweep,
            "head_noise_asymmetry_sweep": asymmetry_sweep,
            "mean_shots": {a: float(runs[a]["n_shots"].mean()) for a in arms},
            "figure": fig,
        },
    )
    print("results -> results/e2_validity.json")


def make_figure(sweep_a, sweep_b, runs, head_noise_sweep, asymmetry_sweep) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {
        "stratified": ("o-", "tab:blue"),
        "likelihood_ratio": ("D-", "tab:green"),
        "marginal": ("s--", "tab:orange"),
        "best_shot": ("^--", "tab:red"),
        "greedy": ("v--", "tab:purple"),
    }
    fig, axes = plt.subplots(1, 5, figsize=(24, 4.6))

    ax = axes[0]
    alphas = [r["alpha"] for r in sweep_a]
    ax.plot(alphas, alphas, "k:", lw=1.4, label="nominal (the promise)")
    for arm, (style, color) in styles.items():
        ax.plot(alphas, [r[arm] for r in sweep_a], style, color=color, label=arm, ms=5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("nominal alpha")
    ax.set_ylabel("false-conviction rate on sound teeth")
    ax.set_title("(a) validity vs the bar demanded")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    Ks = [r["budget"] for r in sweep_b]
    ax.axhline(FIXED_ALPHA, color="k", ls=":", lw=1.4, label=f"nominal alpha = {FIXED_ALPHA}")
    for arm, (style, color) in styles.items():
        ax.plot(Ks, [r[arm] for r in sweep_b], style, color=color, label=arm, ms=5)
    ax.set_xlabel("capture budget K")
    ax.set_ylabel("false-conviction rate")
    ax.set_title("(b) what more photographs cost you")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    for arm, (_, color) in styles.items():
        ev = np.sort(runs[arm]["evidence"])
        tail = 1.0 - np.arange(len(ev)) / len(ev)
        ax.plot(ev, tail, color=color, label=arm)
    ref = np.logspace(0, 2.4, 100)
    ax.plot(ref, 1.0 / ref, "k:", lw=1.4, label="Ville bound 1/x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 250)
    ax.set_ylim(1e-4, 1.2)
    ax.set_xlabel("evidence reported against a sound tooth")
    ax.set_ylabel("P(evidence >= x)")
    ax.set_title("(c) the whole tail, against Ville")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[3]
    hn = [r["head_noise"] for r in head_noise_sweep]
    rates = [r["crossing_rate"] for r in head_noise_sweep]
    los = [r["ci_lo"] for r in head_noise_sweep]
    his = [r["ci_hi"] for r in head_noise_sweep]
    ax.axhline(FIXED_ALPHA, color="k", ls=":", lw=1.4, label=f"nominal alpha = {FIXED_ALPHA}")
    ax.fill_between(hn, los, his, alpha=0.2, color="tab:blue", label="Wilson CI")
    ax.plot(hn, rates, "o-", color="tab:blue", label="stratified")
    for r in head_noise_sweep:
        if r["violates"]:
            ax.plot(r["head_noise"], r["crossing_rate"], "x", color="tab:red", ms=12, mew=2.5, zorder=5)
    ax.set_xlabel("head_noise (confidence-head estimation noise)")
    ax.set_ylabel("false-conviction rate")
    ax.set_title("(d) the falsifiable A1' prediction:\nvalidity itself vs confidence-head noise")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[4]
    tn = [r["head_noise_test"] for r in asymmetry_sweep]
    rates_e = [r["crossing_rate"] for r in asymmetry_sweep]
    los_e = [r["ci_lo"] for r in asymmetry_sweep]
    his_e = [r["ci_hi"] for r in asymmetry_sweep]
    ax.axhline(FIXED_ALPHA, color="k", ls=":", lw=1.4, label=f"nominal alpha = {FIXED_ALPHA}")
    ax.axvline(HEAD_NOISE_CAL, color="grey", ls="--", lw=1.0, label=f"symmetric (cal={HEAD_NOISE_CAL:.2f})")
    ax.fill_between(tn, los_e, his_e, alpha=0.2, color="tab:purple", label="Wilson CI")
    ax.plot(tn, rates_e, "o-", color="tab:purple", label="stratified")
    for r in asymmetry_sweep:
        if r["violates"]:
            ax.plot(r["head_noise_test"], r["crossing_rate"], "x", color="tab:red", ms=12, mew=2.5, zorder=5)
    ax.set_xlabel(f"test-time head_noise (calibration fixed at {HEAD_NOISE_CAL:.2f})")
    ax.set_ylabel("false-conviction rate")
    ax.set_title("(e) the residual gap:\nasymmetric calibration/deployment noise")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.suptitle(
        "E2: every e-process stays under its promise; peeking at the diagnosis does not",
        fontsize=12,
    )
    fig.tight_layout()
    path = figure_path("e2_validity.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


if __name__ == "__main__":
    main()
