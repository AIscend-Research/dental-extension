"""Tests for the shared-budget allocator (src/bench/allocation.py)."""

from __future__ import annotations

import numpy as np
import pytest

from src.bench.allocation import run_docket_global_budget
from src.bench.docket import make_docket
from src.bench.metrics import score_results
from src.bench.runner import fit_calibrator, run_docket
from src.bench.policies import EvidentialCapture
from src.evidence.ladder import BurdenSpec, PREPONDERANCE
from src.models.diagnostic import SurrogateChannel, calibrate_loss_scale, calibrate_separation


@pytest.fixture(scope="module")
def world():
    channel = SurrogateChannel()
    calibrate_separation(channel, 0.88, n=3000, seed=0)
    calibrate_loss_scale(channel, 0.78, n=1500, seed=1)
    calibrator, _ = fit_calibrator(channel, n=4000, seed=7)
    return channel, calibrator


def test_every_case_gets_at_least_one_shot(world):
    channel, calibrator = world
    docket = make_docket("alloc_smoke", n_cases=200, budget=4, seed=1)
    results = run_docket_global_budget(docket, channel, calibrator, total_budget=800)
    assert len(results) == len(docket)
    assert all(r.n_captures >= 1 for r in results)


def test_total_spend_never_exceeds_the_budget(world):
    channel, calibrator = world
    docket = make_docket("alloc_budget", n_cases=200, budget=4, seed=2)
    total_budget = len(docket) * docket.budget
    results = run_docket_global_budget(docket, channel, calibrator, total_budget=total_budget)
    assert sum(r.n_captures for r in results) <= total_budget


def test_no_case_exceeds_max_per_case(world):
    channel, calibrator = world
    docket = make_docket("alloc_cap", n_cases=100, budget=4, seed=3)
    results = run_docket_global_budget(
        docket, channel, calibrator, total_budget=len(docket) * docket.budget, max_per_case=6
    )
    assert all(r.n_captures <= 6 for r in results)


def test_results_are_sound_never_peeked(world):
    """The allocator's priority signal is predicted usability only -- it must
    never read the diagnosis channel to decide who gets the next shot."""
    channel, calibrator = world
    docket = make_docket("alloc_sound", n_cases=100, budget=4, seed=4)
    results = run_docket_global_budget(docket, channel, calibrator, total_budget=len(docket) * docket.budget)
    assert all(r.peeked is False for r in results)


def test_reallocating_never_produces_a_worse_vpc_than_uniform_on_an_easy_docket(world):
    """Not a tight guarantee in general, but on a docket where uniform K
    already wastes captures on early deciders, giving them to hard cases
    should not make things worse."""
    channel, calibrator = world
    docket = make_docket("alloc_compare", n_cases=1500, budget=4, seed=5)
    total_budget = len(docket) * docket.budget

    uniform = score_results(
        "uniform", run_docket(docket, EvidentialCapture(), channel, calibrator), docket.burden
    )
    allocated = score_results(
        "shared", run_docket_global_budget(docket, channel, calibrator, total_budget=total_budget), docket.burden
    )
    assert allocated.verdicts_per_capture >= uniform.verdicts_per_capture - 0.01


def test_validity_holds_on_an_all_sound_null_docket(world):
    """Empirical check of the independence argument in the module docstring:
    the shared scheduler should not inflate the false-conviction rate above
    its nominal alpha, mirroring E2's single-case check."""
    channel, calibrator = world
    burden = BurdenSpec.symmetric(PREPONDERANCE)
    docket = make_docket("alloc_null", n_cases=4000, prevalence=0.0, budget=4, burden=burden, seed=6)
    total_budget = len(docket) * docket.budget
    results = run_docket_global_budget(docket, channel, calibrator, total_budget=total_budget)
    row = score_results("null", results, burden)
    # Wilson lower bound above alpha is the violation condition used everywhere
    # else in this codebase (src/bench/metrics.py) -- reuse the same bar.
    assert not row.convict_violation
