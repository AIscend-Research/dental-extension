"""Leaderboard columns, and the guarantee audit that sits beside them.

The headline metric is **justified verdicts per capture**:

    VPC = (cases decided) / (photographs taken)

A policy that decides everything after one shot scores 1.0 and is almost
certainly cheating; a policy that escalates everything scores 0.0 and is
useless. Neither is punished by accuracy alone, which is why accuracy is not
the headline.

What stops VPC from being gamed is the pair of columns next to it. The
validity theorem promises that, when the tooth is really sound, the wealth
process crosses the conviction threshold with probability at most
alpha_convict -- at any stopping time, however adaptive. So the empirical
false-conviction rate among truly sound cases is a direct, falsifiable test of
the guarantee, and the same holds mirrored for false discharges. Both come
with Wilson intervals, and a violation is only declared when the interval's
*lower* bound clears alpha, so Monte Carlo noise cannot manufacture one.

One asymmetry to keep in mind when reading the numbers: the verdict machine
escalates when both burdens are met at once, which can only remove convictions.
The observed false-conviction rate is therefore slightly conservative relative
to what the theorem bounds. That direction is safe -- it cannot hide a
violation, only understate a satisfied bound.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from src.bench.policies import SessionResult
from src.evidence.ladder import BurdenSpec
from src.evidence.verdict import VerdictOutcome


def _nanmean(values: list[float]) -> float:
    """Mean of the non-NaN entries, or NaN if there are none.

    np.nanmean warns on an all-NaN slice, which happens whenever results carry
    no usability trace (hand-built rows in tests, or an arm that never
    records one). NaN is the right answer there, not a warning.
    """
    finite = [v for v in values if not math.isnan(v)]
    return float(np.mean(finite)) if finite else float("nan")


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Wilson rather than normal-approximation because the rates being tested are
    small (alpha = 0.05 and below) and the normal interval is badly behaved
    near zero -- it can dip below 0 and understate uncertainty exactly where
    the guarantee check lives.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass
class LeaderboardRow:
    """One policy's line on the Docket leaderboard."""

    policy: str
    n_cases: int
    n_captures: int
    mean_captures: float
    verdict_rate: float
    verdicts_per_capture: float
    verdict_accuracy: float
    refer_rate: float
    # guarantee audit
    false_conviction_rate: float
    false_conviction_ci: tuple[float, float]
    alpha_convict: float
    convict_violation: bool
    false_discharge_rate: float
    false_discharge_ci: tuple[float, float]
    alpha_discharge: float
    discharge_violation: bool
    guaranteed: bool
    # secondary
    sensitivity: float
    specificity: float
    mean_final_usability: float

    def as_dict(self) -> dict:
        return asdict(self)


def score_results(
    policy_name: str,
    results: list[SessionResult],
    burden: BurdenSpec,
) -> LeaderboardRow:
    """Score one policy's run over a docket.

    `guaranteed` is False whenever any session in the run read the diagnosis
    channel to decide when to stop. For those arms the error columns are still
    reported -- that is the point -- but they are reported as measurements, not
    as bounds that were promised and kept.
    """
    n = len(results)
    if n == 0:
        raise ValueError("no results to score")

    captures = np.array([r.n_captures for r in results])
    labels = np.array([r.label for r in results])
    outcomes = [r.outcome for r in results]

    decided = np.array([o in (VerdictOutcome.CARIES, VerdictOutcome.SOUND) for o in outcomes])
    correct = np.array([r.correct is True for r in results])

    # guarantee audit: false convictions among truly sound cases
    sound = labels == 0
    diseased = labels == 1
    convicted_sound = int(
        sum(1 for r, s in zip(results, sound) if s and r.outcome is VerdictOutcome.CARIES)
    )
    discharged_diseased = int(
        sum(1 for r, d in zip(results, diseased) if d and r.outcome is VerdictOutcome.SOUND)
    )
    n_sound, n_diseased = int(sound.sum()), int(diseased.sum())

    fcr = convicted_sound / n_sound if n_sound else float("nan")
    fdr = discharged_diseased / n_diseased if n_diseased else float("nan")
    fcr_ci = wilson_interval(convicted_sound, n_sound)
    fdr_ci = wilson_interval(discharged_diseased, n_diseased)

    total_captures = int(captures.sum())
    convicted = np.array([o is VerdictOutcome.CARIES for o in outcomes])
    discharged = np.array([o is VerdictOutcome.SOUND for o in outcomes])

    return LeaderboardRow(
        policy=policy_name,
        n_cases=n,
        n_captures=total_captures,
        mean_captures=float(captures.mean()),
        verdict_rate=float(decided.mean()),
        verdicts_per_capture=float(decided.sum() / total_captures) if total_captures else float("nan"),
        verdict_accuracy=float(correct[decided].sum() / decided.sum()) if decided.any() else float("nan"),
        refer_rate=float(1.0 - decided.mean()),
        false_conviction_rate=fcr,
        false_conviction_ci=fcr_ci,
        alpha_convict=burden.convict.alpha,
        # a violation needs the CI's LOWER bound above alpha, so noise alone
        # cannot declare one
        convict_violation=bool(fcr_ci[0] > burden.convict.alpha) if n_sound else False,
        false_discharge_rate=fdr,
        false_discharge_ci=fdr_ci,
        alpha_discharge=burden.discharge.alpha,
        discharge_violation=bool(fdr_ci[0] > burden.discharge.alpha) if n_diseased else False,
        guaranteed=not any(r.peeked for r in results),
        # sensitivity/specificity over cases where a verdict was actually rendered
        sensitivity=float(convicted[diseased].sum() / max(1, (decided & diseased).sum())),
        specificity=float(discharged[sound].sum() / max(1, (decided & sound).sum())),
        mean_final_usability=_nanmean([r.final_usability for r in results]),
    )


def format_leaderboard(rows: list[LeaderboardRow]) -> str:
    """Fixed-width leaderboard, sorted by the headline metric.

    Violated guarantees are marked so a good VPC cannot be read without its
    caveat sitting on the same line.
    """
    header = (
        f"{'policy':<24} {'VPC':>6} {'verdict%':>9} {'acc':>6} {'refer%':>7} "
        f"{'shots':>6} {'FCR':>7} {'a_c':>5} {'FDR':>7} {'a_d':>5}  status"
    )
    lines = [header, "-" * len(header)]
    for r in sorted(rows, key=lambda x: -x.verdicts_per_capture):
        if not r.guaranteed:
            status = "UNGUARANTEED"
            if r.convict_violation or r.discharge_violation:
                status += " + VIOLATED"
        elif r.convict_violation or r.discharge_violation:
            status = "VIOLATED"
        else:
            status = "ok"
        lines.append(
            f"{r.policy:<24} {r.verdicts_per_capture:>6.3f} {r.verdict_rate * 100:>8.1f}% "
            f"{r.verdict_accuracy:>6.3f} {r.refer_rate * 100:>6.1f}% {r.mean_captures:>6.2f} "
            f"{r.false_conviction_rate:>7.4f} {r.alpha_convict:>5.2f} "
            f"{r.false_discharge_rate:>7.4f} {r.alpha_discharge:>5.2f}  {status}"
        )
    return "\n".join(lines)
