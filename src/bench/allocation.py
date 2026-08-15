"""Evidential capture under one shared capture budget, instead of a fixed K each.

`src.bench.runner.run_docket` gives every case the same budget K. A case
whose first photo is already excellent spends the same allowance as one that
needs every retake -- which is not how a real clinic with a fixed total
number of photographs in a session would want to spend them.

`run_docket_global_budget` spends the identical *total* number of
photographs (N cases x K, so it is directly comparable to the uniform
baseline at equal cost) but lets a greedy allocator decide, shot by shot,
which case gets the next one: among cases not yet decided, the one whose most
recent shot looks most usable gets priority, on the theory that it is closest
to being rescued into a verdict. Every case still gets at least one shot
before any reallocation happens, because a session cannot be instructed
before it has produced something to react to (`CaptureSession.capture`).

A validity note worth being explicit about, since this is the one place a
policy's stopping behaviour depends on OTHER cases' data: whether case i
receives its next shot at a given round depends on every other pending
case's predicted usability. That does not touch A2/A3 for case i, because
cases are independent processes (separate `CaptureSession`, separate `rng`)
and the priority signal is degradation-channel-only (`usabilities[-1]`,
never a diagnosis score) -- so the event "case i's shot at round r" is
independent of case i's own null hypothesis and reveals nothing about it.
Case i's wealth process is therefore still a supermartingale with respect to
the joint filtration across all cases, and Ville's inequality applies exactly
as before. This is argued, not proved as a theorem here; `main()` in
`experiments/e11_capture_budgets.py` checks it empirically the way E2 checks
the single-case guarantee, on an all-sound null docket.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

from src.bench.docket import Docket, to_model_case
from src.bench.policies import SessionResult, _correctness
from src.evidence.calibration import Calibrator
from src.evidence.ewealth import BettingStrategy, DegradationAwareBet
from src.evidence.verdict import VerdictMachine, VerdictOutcome
from src.models.diagnostic import DiagnosticChannel
from src.sim.instructions import RETAKE_ANY
from src.sim.session import CaptureSession, SessionConfig


@dataclass
class _CaseState:
    case_id: int
    label: int
    session: CaptureSession
    machine: VerdictMachine
    case_obj: object
    rng: np.random.Generator
    instruction: object = None
    n_captures: int = 0
    outcome: VerdictOutcome = VerdictOutcome.PENDING
    usabilities: list = field(default_factory=list)
    scores: list = field(default_factory=list)
    qualities: list = field(default_factory=list)
    instructions: list = field(default_factory=list)


def run_docket_global_budget(
    docket: Docket,
    channel: DiagnosticChannel,
    calibrator: Calibrator,
    total_budget: int | None = None,
    max_per_case: int | None = None,
    strategy: BettingStrategy | None = None,
    n_strata: int = 4,
    config: SessionConfig | None = None,
) -> list[SessionResult]:
    """Run evidential capture over `docket` under ONE shared budget.

    Args:
        total_budget: total photographs to spend across the whole docket.
            Defaults to `len(docket) * docket.budget`, i.e. the same total
            cost as running `run_docket` with the uniform per-case budget --
            the fair basis for comparison.
        max_per_case: cap on any single case's allocation, so one hard case
            cannot absorb the entire shared budget. Defaults to 4x the
            uniform per-case budget.
    """
    n = len(docket)
    total_budget = total_budget if total_budget is not None else n * docket.budget
    max_per_case = max_per_case if max_per_case is not None else docket.budget * 4
    strategy = strategy or DegradationAwareBet()

    cases: dict[int, _CaseState] = {}
    for dc in docket.cases:
        rng = np.random.default_rng(dc.seed)
        session = CaptureSession(rng=rng, difficulty=docket.clinic_difficulty, config=config)
        machine = VerdictMachine(calibrator=calibrator, burden=docket.burden, strategy=strategy, n_strata=n_strata)
        cases[dc.case_id] = _CaseState(
            case_id=dc.case_id, label=dc.label, session=session, machine=machine,
            case_obj=to_model_case(dc), rng=rng,
        )

    def take_shot(cs: _CaseState) -> bool:
        """One more shot for this case. Returns True iff now decided."""
        capture = cs.session.capture(cs.instruction if cs.n_captures > 0 else None)
        reading = channel.read(cs.case_obj, capture.severities, cs.rng)
        cs.n_captures += 1
        cs.usabilities.append(reading.usability)
        cs.scores.append(reading.score)
        cs.qualities.append(reading.true_quality)

        verdict = cs.machine.observe(
            score=reading.score, degradation=reading.degradation,
            predicted_usability=reading.usability,
            captures_remaining=max_per_case - cs.n_captures,
        )
        cs.outcome = verdict.outcome
        if verdict.is_terminal:
            return True
        cs.instruction = verdict.instruction or RETAKE_ANY
        cs.instructions.append(cs.instruction.name if cs.instruction else "none")
        return False

    # Phase 1: everyone gets their mandatory first shot.
    heap: list[tuple[float, int]] = []
    spent = 0
    for cs in cases.values():
        done = take_shot(cs)
        spent += 1
        if not done:
            heapq.heappush(heap, (-cs.usabilities[-1], cs.case_id))

    # Phase 2: greedy round-robin on the shared remainder -- highest predicted
    # usability (most rescuable) gets the next shot.
    while heap and spent < total_budget:
        _, cid = heapq.heappop(heap)
        cs = cases[cid]
        if cs.outcome is not VerdictOutcome.PENDING or cs.n_captures >= max_per_case:
            continue
        done = take_shot(cs)
        spent += 1
        if not done:
            heapq.heappush(heap, (-cs.usabilities[-1], cs.case_id))

    for _, cid in heap:
        cs = cases[cid]
        if cs.outcome is VerdictOutcome.PENDING:
            cs.outcome = VerdictOutcome.REFER

    results = []
    for dc in docket.cases:
        cs = cases[dc.case_id]
        if cs.outcome is VerdictOutcome.PENDING:  # budget exhausted before its first shot's follow-up
            cs.outcome = VerdictOutcome.REFER
        results.append(SessionResult(
            case_id=cs.case_id, label=cs.label, outcome=cs.outcome, n_captures=cs.n_captures,
            correct=_correctness(cs.outcome, cs.label), peeked=False,
            wealth_convict=cs.machine.convict.running_max, wealth_discharge=cs.machine.discharge.running_max,
            final_usability=cs.usabilities[-1] if cs.usabilities else float("nan"),
            mean_true_quality=float(np.mean(cs.qualities)) if cs.qualities else float("nan"),
            instructions=tuple(cs.instructions), usability_trace=tuple(cs.usabilities),
            score_trace=tuple(cs.scores),
        ))
    return results
