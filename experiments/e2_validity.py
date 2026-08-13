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

Run: .venv/bin/python -m experiments.e2_validity
"""

from __future__ import annotations

import numpy as np

from experiments.common import (
    CLINIC_DIFFICULTY,
    PREVALENCE,
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

    fig = make_figure(sweep_a, sweep_b, runs)
    print(f"\nfigure -> {fig}")

    save_table("e2_validity_alpha_sweep", sweep_a)
    save_table("e2_validity_budget_sweep", sweep_b)
    save_results(
        "e2_validity",
        {
            "reader": {"clean_auc": world.clean_auc, "clinic_auc": world.clinic_auc},
            "n_null_sessions": N_NULL_SESSIONS,
            "budget": BUDGET,
            "alpha_sweep": sweep_a,
            "budget_sweep": sweep_b,
            "scope_of_guarantee": scope,
            "mean_shots": {a: float(runs[a]["n_shots"].mean()) for a in arms},
            "figure": fig,
        },
    )
    print("results -> results/e2_validity.json")


def make_figure(sweep_a, sweep_b, runs) -> str:
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
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

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
