"""E1 -- does the simulator behave like a capture *process* rather than a filter?

Everything downstream rests on the claim that a retake loop produces
correlated, causally-responsive, non-exchangeable captures. That claim is
checkable, so this experiment checks it instead of asserting it. Five
measurements, each of which could come back negative:

  1. PERSISTENCE. Within-session lag-1 correlation of scene severity, against
     the same quantity after shuffling shots across sessions. An i.i.d.
     corruption filter scores ~0 by construction; if this simulator scores ~0
     too, it is a filter wearing a costume.

  2. INSTRUCTION RESPONSE. Change in the targeted factor after a targeted
     instruction, versus after an untargeted "just retake". If these are
     equal, the instruction space is decoration.

  3. SIDE EFFECTS. The empirically recovered coupling matrix -- what actually
     happens to the other factors when one is corrected. This should recover
     the structure in `src.sim.state.COUPLING`, and in particular the
     glare<->darkness trade-off that makes greedy instruction policies
     oscillate.

  4. NON-EXCHANGEABILITY. A permutation test on the within-session ordering of
     usability. Standard split-conformal assumes exchangeability; if this test
     rejects, that assumption is false here and the stratified construction in
     `src/evidence/calibration.py` is not paranoia.

  5. FATIGUE. Whether long retake loops degrade the operator, so that retaking
     has a cost beyond the capture budget.

Run: .venv/bin/python -m experiments.e1_simulator
"""

from __future__ import annotations

import numpy as np

from experiments.common import banner, figure_path, save_results, save_table
from src.sim.instructions import RETAKE_ANY, instruction_for_factor
from src.sim.session import CaptureSession, SessionConfig
from src.sim.state import COUPLING, FACTORS

N_SESSIONS = 4000
N_SHOTS = 5
CLINIC_DIFFICULTY = 0.5


def mean_severity(capture) -> float:
    return float(np.mean([capture.state.factors[f] for f in FACTORS]))


# ---------------------------------------------------------------------------
# 1. persistence
# ---------------------------------------------------------------------------


def measure_persistence(seed: int = 0) -> dict:
    """Lag-1 correlation of mean severity within a session vs across sessions.

    The comparison arm matters. A raw within-session correlation could be high
    simply because sessions differ from one another (between-session variance
    inflates any pooled correlation). So the control re-pairs each shot with a
    shot from a *different* session at the same index, which destroys the
    within-session link while preserving every marginal distribution.
    """
    rng = np.random.default_rng(seed)
    traj = np.empty((N_SESSIONS, N_SHOTS))
    for i in range(N_SESSIONS):
        s = CaptureSession(rng=rng, difficulty=CLINIC_DIFFICULTY)
        for t in range(N_SHOTS):
            traj[i, t] = mean_severity(s.capture(RETAKE_ANY if t else None))

    within = np.corrcoef(traj[:, :-1].ravel(), traj[:, 1:].ravel())[0, 1]
    perm = rng.permutation(N_SESSIONS)
    across = np.corrcoef(traj[:, :-1].ravel(), traj[perm][:, 1:].ravel())[0, 1]
    return {
        "lag1_within_session": float(within),
        "lag1_shuffled_across_sessions": float(across),
        "trajectories": traj,
    }


# ---------------------------------------------------------------------------
# 2 + 3. instruction response and side effects
# ---------------------------------------------------------------------------


def measure_instruction_response(seed: int = 1, n: int = 3000) -> dict:
    """Per-factor effect of targeting it, and the leakage onto every other factor.

    Each trial takes one shot, issues one instruction, takes a second shot, and
    records the change in every factor. Comparing the targeted arm against the
    untargeted arm on the *same* factor isolates the instruction's effect from
    the process noise a retake gets for free.
    """
    rng = np.random.default_rng(seed)
    # delta[arm][target_factor][observed_factor] -> list of changes
    deltas: dict[str, dict[str, dict[str, list[float]]]] = {
        "targeted": {t: {f: [] for f in FACTORS} for t in FACTORS},
        "untargeted": {t: {f: [] for f in FACTORS} for t in FACTORS},
    }
    complied_count = {t: 0 for t in FACTORS}
    trials = {t: 0 for t in FACTORS}

    for arm in ("targeted", "untargeted"):
        for i in range(n):
            target = FACTORS[i % len(FACTORS)]
            s = CaptureSession(rng=rng, difficulty=CLINIC_DIFFICULTY)
            before = s.capture()
            instruction = instruction_for_factor(target) if arm == "targeted" else RETAKE_ANY
            after = s.capture(instruction)
            for f in FACTORS:
                deltas[arm][target][f].append(
                    after.state.factors[f] - before.state.factors[f]
                )
            if arm == "targeted":
                trials[target] += 1
                complied_count[target] += int(after.complied)

    response = []
    for target in FACTORS:
        t_mean = float(np.mean(deltas["targeted"][target][target]))
        u_mean = float(np.mean(deltas["untargeted"][target][target]))
        response.append({
            "factor": target,
            "targeted_delta": t_mean,
            "untargeted_delta": u_mean,
            "instruction_effect": t_mean - u_mean,
            "compliance_rate": complied_count[target] / max(1, trials[target]),
        })

    # empirical coupling: leakage onto other factors, net of the untargeted arm
    measured_coupling = {}
    for target in FACTORS:
        row = {}
        for f in FACTORS:
            if f == target:
                continue
            row[f] = float(
                np.mean(deltas["targeted"][target][f]) - np.mean(deltas["untargeted"][target][f])
            )
        measured_coupling[target] = row

    return {
        "response": response,
        "measured_coupling": measured_coupling,
        "specified_coupling": COUPLING,
    }


# ---------------------------------------------------------------------------
# 4. non-exchangeability
# ---------------------------------------------------------------------------


def test_exchangeability(seed: int = 2, n: int = 3000, n_perm: int = 2000) -> dict:
    """Permutation test: is the within-session usability sequence exchangeable?

    Under exchangeability, the shot index carries no information, so the mean
    usability at each position is the same and the statistic

        T = mean_i ( u_last(i) - u_first(i) )

    has a null distribution obtained by permuting positions within each
    session independently. This is exact under the null, needs no distributional
    assumption, and directly targets the assumption split-conformal makes.

    The sessions here run a realistic retake loop -- retake while the shot
    looks unusable, targeting the worst factor -- because that is the sequence
    the calibrator would actually face. An arbitrary fixed-instruction sequence
    would understate the effect.
    """
    rng = np.random.default_rng(seed)
    seqs = np.empty((n, N_SHOTS))
    for i in range(n):
        s = CaptureSession(rng=rng, difficulty=CLINIC_DIFFICULTY)
        instruction = None
        for t in range(N_SHOTS):
            cap = s.capture(instruction if t else None)
            seqs[i, t] = cap.state.usability()
            worst = max(s.scene.factors.items(), key=lambda kv: kv[1])[0]
            instruction = instruction_for_factor(worst)

    observed = float(np.mean(seqs[:, -1] - seqs[:, 0]))
    null = np.empty(n_perm)
    for b in range(n_perm):
        # permute positions independently within every session
        idx = np.argsort(rng.random(seqs.shape), axis=1)
        permuted = np.take_along_axis(seqs, idx, axis=1)
        null[b] = float(np.mean(permuted[:, -1] - permuted[:, 0]))

    # two-sided p-value, with the standard +1 so it can never be exactly 0
    p = (1 + int(np.sum(np.abs(null) >= abs(observed)))) / (n_perm + 1)
    return {
        "statistic_last_minus_first": observed,
        "permutation_p_value": float(p),
        "null_sd": float(null.std()),
        "mean_usability_by_position": seqs.mean(axis=0).tolist(),
        "sequences": seqs,
    }


# ---------------------------------------------------------------------------
# 5. fatigue
# ---------------------------------------------------------------------------


def measure_fatigue(seed: int = 3, n: int = 2000, n_shots: int = 8) -> dict:
    """Does tremor rise over a long retake loop even while other factors fall?"""
    rng = np.random.default_rng(seed)
    tremor = np.empty((n, n_shots))
    fatigue = np.empty((n, n_shots))
    for i in range(n):
        s = CaptureSession(rng=rng, difficulty=CLINIC_DIFFICULTY)
        instruction = None
        for t in range(n_shots):
            cap = s.capture(instruction if t else None)
            tremor[i, t] = cap.state.factors["tremor"]
            fatigue[i, t] = cap.operator_fatigue
            # deliberately never ask for STEADY, so any tremor rise is fatigue
            # rather than a failure to request the fix
            instruction = instruction_for_factor("glare")
    return {
        "tremor_by_shot": tremor.mean(axis=0).tolist(),
        "fatigue_by_shot": fatigue.mean(axis=0).tolist(),
    }


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------


def make_figure(persistence, response, exch, fatigue) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # (a) instruction effect per factor
    ax = axes[0, 0]
    factors = [r["factor"] for r in response["response"]]
    x = np.arange(len(factors))
    ax.bar(x - 0.2, [r["targeted_delta"] for r in response["response"]], 0.4, label="targeted")
    ax.bar(x + 0.2, [r["untargeted_delta"] for r in response["response"]], 0.4, label="untargeted")
    ax.set_xticks(x, factors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("change in the targeted factor")
    ax.set_title("(a) instructions move the thing they name")
    ax.legend(fontsize=8)

    # (b) measured coupling
    ax = axes[0, 1]
    mat = np.array([
        [response["measured_coupling"][t].get(f, 0.0) for f in FACTORS] for t in FACTORS
    ])
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-np.abs(mat).max(), vmax=np.abs(mat).max())
    ax.set_xticks(range(len(FACTORS)), FACTORS, rotation=45, ha="right")
    ax.set_yticks(range(len(FACTORS)), FACTORS)
    ax.set_xlabel("effect on")
    ax.set_ylabel("instruction corrected")
    ax.set_title("(b) side effects, recovered empirically")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # (c) usability by shot index -- the selection effect
    ax = axes[1, 0]
    ax.plot(range(1, N_SHOTS + 1), exch["mean_usability_by_position"], "o-")
    ax.set_xlabel("shot index in session")
    ax.set_ylabel("mean true usability")
    ax.set_title(
        f"(c) retake loop shifts the population\n(permutation p = {exch['permutation_p_value']:.4f})"
    )

    # (d) fatigue
    ax = axes[1, 1]
    ax.plot(range(1, len(fatigue["tremor_by_shot"]) + 1), fatigue["tremor_by_shot"], "o-", label="tremor")
    ax.plot(range(1, len(fatigue["fatigue_by_shot"]) + 1), fatigue["fatigue_by_shot"], "s--", label="operator fatigue")
    ax.set_xlabel("shot index")
    ax.set_title("(d) retaking tires the operator")
    ax.legend(fontsize=8)

    fig.suptitle("E1: the capture simulator is a process, not a filter", fontsize=12)
    fig.tight_layout()
    path = figure_path("e1_simulator.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def main() -> None:
    banner("E1 -- capture simulator validation")

    print("[1/5] persistence ...", flush=True)
    persistence = measure_persistence()
    print(
        f"  lag-1 correlation within session : {persistence['lag1_within_session']:.4f}\n"
        f"  same, shuffled across sessions   : {persistence['lag1_shuffled_across_sessions']:.4f}"
    )

    print("[2/5] instruction response + [3/5] side effects ...", flush=True)
    response = measure_instruction_response()
    for r in response["response"]:
        print(
            f"  {r['factor']:<9} targeted {r['targeted_delta']:+.4f} | "
            f"untargeted {r['untargeted_delta']:+.4f} | "
            f"effect {r['instruction_effect']:+.4f} | compliance {r['compliance_rate']:.2f}"
        )
    print("  measured coupling (net of the untargeted arm):")
    for target, row in response["measured_coupling"].items():
        pretty = ", ".join(f"{f}{v:+.4f}" for f, v in row.items())
        print(f"    fixing {target:<9} -> {pretty}")

    print("[4/5] exchangeability permutation test ...", flush=True)
    exch = test_exchangeability()
    print(
        f"  mean usability, last minus first shot: {exch['statistic_last_minus_first']:+.4f}\n"
        f"  permutation p-value                  : {exch['permutation_p_value']:.5f} "
        f"(null sd {exch['null_sd']:.5f})"
    )

    print("[5/5] fatigue ...", flush=True)
    fatigue = measure_fatigue()
    print(f"  tremor by shot : {np.round(fatigue['tremor_by_shot'], 4).tolist()}")
    print(f"  fatigue by shot: {np.round(fatigue['fatigue_by_shot'], 4).tolist()}")

    fig = make_figure(persistence, response, exch, fatigue)
    print(f"\nfigure -> {fig}")

    save_table("e1_instruction_response", response["response"])
    save_results(
        "e1_simulator",
        {
            "persistence": {
                k: v for k, v in persistence.items() if k != "trajectories"
            },
            "instruction_response": response["response"],
            "measured_coupling": response["measured_coupling"],
            "specified_coupling": COUPLING,
            "exchangeability": {k: v for k, v in exch.items() if k != "sequences"},
            "fatigue": fatigue,
            "figure": fig,
            "config": {
                "n_sessions": N_SESSIONS,
                "n_shots": N_SHOTS,
                "clinic_difficulty": CLINIC_DIFFICULTY,
            },
        },
    )
    print("results -> results/e1_simulator.json")


if __name__ == "__main__":
    main()
