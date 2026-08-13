"""The Docket: a benchmark for evidential capture.

The task. Given a capture budget of K photographs and a mandated standard of
proof, decide as many cases as possible while meeting that standard. The score
is not accuracy. Accuracy is trivially maximised by never deciding anything
hard; and it is trivially inflated by deciding everything and getting most of
them right. The score is **justified verdicts per capture**, subject to the
guarantee actually holding.

That subjection is what makes the leaderboard non-gameable. A policy can lift
its verdict rate to 1.0 by ignoring the burden of proof entirely -- and the
benchmark will report, in the adjacent column, that its false-conviction rate
blew through the alpha it claimed. Both unsound baselines in `policies.py` do
exactly this, and they are included because the gap between "looks better" and
"is valid" is the result worth publishing.

A `Docket` is a frozen, seeded specification: the cases, their difficulties,
the clinic difficulty, the budget, and the burden. Two people running the same
Docket id run the same benchmark.

Modules:
    docket    -- the benchmark specification and case generation
    policies  -- capture policies, sound and unsound
    runner    -- executes a policy over a docket
    metrics   -- leaderboard columns, with binomial CIs on the guarantee checks
"""

from src.bench.docket import Docket, DocketCase, make_docket
from src.bench.metrics import LeaderboardRow, score_results
from src.bench.policies import POLICIES, CapturePolicy, SessionResult, policy_by_name
from src.bench.runner import run_docket, fit_calibrator

__all__ = [
    "CapturePolicy",
    "Docket",
    "DocketCase",
    "LeaderboardRow",
    "POLICIES",
    "SessionResult",
    "fit_calibrator",
    "make_docket",
    "policy_by_name",
    "run_docket",
    "score_results",
]
