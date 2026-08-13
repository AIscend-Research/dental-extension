"""Capture policies: the entries on the leaderboard.

A policy is a complete procedure for adjudicating one case: when to shoot,
what to ask for, when to stop, and how to turn the accumulated evidence into a
verdict. They are written as whole procedures rather than as "a stopping rule
plugged into a shared machine" because the unsound baselines genuinely differ
in how they convert evidence to a verdict -- forcing them through the sound
machine would make them artificially safe and turn the comparison into a straw
man.

The arms, and what each one isolates:

    single_shot           no retakes at all. The status quo for a phone app.
    fixed_retake_k        always burn K shots, untargeted. Isolates "does
                          retaking help at all", separate from targeting.
    untargeted_evidential evidential stopping, but every subpoena is a bare
                          "retake". Isolates the value of *naming* the fault.
    evidential_capture    the proposed method: evidential stopping,
                          degradation-targeted subpoenas, degradation-aware
                          stakes.
    oracle_instruction    as above, but the subpoena reads the true latent
                          state. Upper bound on how much better a perfect
                          confidence head could make the instructions.
    greedy_diagnostic     UNSOUND. Retakes while the diagnosis looks
                          ambiguous, then reports the last shot's p-value as
                          if it were a single planned test.
    naive_best_shot       UNSOUND. Takes all K shots and reports the most
                          favourable p-value among them.

The two unsound arms are not caricatures. `greedy_diagnostic` is the obvious
thing to build -- retake when unsure -- and `naive_best_shot` is what "use the
best photo" means when you implement it. They are unsound for one reason:
their stopping rule and their reported evidence both depend on the diagnosis
channel, which is exactly the dependence the validity theorem forbids. That is
visible in the code as the fact that they alone require a `PeekingView`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.evidence.calibration import Calibrator
from src.evidence.ewealth import BettingStrategy, DegradationAwareBet
from src.evidence.ladder import BurdenSpec
from src.evidence.verdict import (
    DEGRADATION_TO_FACTOR,
    VerdictMachine,
    VerdictOutcome,
)
from src.models.diagnostic import Case, DiagnosticChannel
from src.sim.instructions import Instruction, RETAKE_ANY, instruction_for_factor
from src.sim.session import CaptureSession


@dataclass
class SessionResult:
    """Everything one adjudicated case produced.

    Attributes:
        outcome: the verdict, including REFER.
        n_captures: photographs actually taken. The denominator of the
            headline metric.
        correct: whether a rendered verdict matched the truth; None if the
            case was escalated (an escalation is neither right nor wrong).
        peeked: whether this arm used the diagnosis channel to decide when to
            stop. Recorded per session so the scorer can refuse to report a
            guarantee for arms that forfeited it.
    """

    case_id: int
    label: int
    outcome: VerdictOutcome
    n_captures: int
    correct: bool | None
    peeked: bool = False
    wealth_convict: float = 1.0
    wealth_discharge: float = 1.0
    final_usability: float = float("nan")
    mean_true_quality: float = float("nan")
    instructions: tuple[str, ...] = ()
    usability_trace: tuple[float, ...] = ()
    score_trace: tuple[float, ...] = ()


@dataclass
class PolicyContext:
    """Everything a policy needs to adjudicate one case."""

    session: CaptureSession
    case: Case
    channel: DiagnosticChannel
    calibrator: Calibrator
    burden: BurdenSpec
    budget: int
    rng: np.random.Generator
    n_strata: int = 4


class CapturePolicy:
    """Interface for a docket entry."""

    name = "base"
    #: whether this policy reads the diagnosis channel when deciding to stop.
    #: Arms with this set True have forfeited the anytime-validity guarantee,
    #: and `src/bench/metrics.py` reports their error rates as unguaranteed.
    peeks = False

    def run(self, ctx: PolicyContext, case_id: int) -> SessionResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Sound arms: stopping and betting read the degradation channel only.
# ---------------------------------------------------------------------------


@dataclass
class EvidentialCapture(CapturePolicy):
    """The proposed method.

    Bets on each shot with a stake set by the degradation channel, stops the
    instant either burden is met (checking the running maximum, so the stop is
    anytime-valid), and when neither is met issues a subpoena naming the
    dominant predicted artifact.

    `targeted=False` degrades it to the untargeted control; `oracle_instruction`
    lets it cheat on the instruction *only*, to bound how much a perfect
    confidence head could buy.
    """

    targeted: bool = True
    oracle_instruction: bool = False
    strategy: BettingStrategy = field(default_factory=DegradationAwareBet)
    #: if False, run the full budget instead of stopping early. Isolates the
    #: value of *stopping* from the value of the evidence machinery.
    early_stop: bool = True
    name: str = "evidential_capture"

    def run(self, ctx: PolicyContext, case_id: int) -> SessionResult:
        machine = VerdictMachine(
            calibrator=ctx.calibrator,
            burden=ctx.burden,
            strategy=self.strategy,
            n_strata=ctx.n_strata,
        )
        instruction: Instruction | None = None
        outcome = VerdictOutcome.REFER
        instructions: list[str] = []
        usabilities: list[float] = []
        scores: list[float] = []
        qualities: list[float] = []
        n = 0

        for shot_i in range(ctx.budget):
            capture = ctx.session.capture(instruction if shot_i > 0 else None)
            reading = ctx.channel.read(ctx.case, capture.severities, ctx.rng)
            n += 1
            usabilities.append(reading.usability)
            scores.append(reading.score)
            qualities.append(reading.true_quality)

            verdict = machine.observe(
                score=reading.score,
                degradation=reading.degradation,
                predicted_usability=reading.usability,
                captures_remaining=ctx.budget - shot_i - 1,
            )
            outcome = verdict.outcome
            if verdict.is_terminal and self.early_stop:
                break
            if verdict.is_terminal and not self.early_stop:
                # keep shooting but remember that the burden was met; the
                # running-maximum check means a later dip cannot revoke it
                pass

            instruction = self._choose_instruction(verdict.instruction, ctx)
            instructions.append(instruction.name if instruction else "none")

        if outcome is VerdictOutcome.PENDING:
            outcome = VerdictOutcome.REFER

        return _result(
            case_id, ctx, outcome, n, machine, usabilities, scores, qualities, instructions,
            peeked=False,
        )

    def _choose_instruction(self, proposed: Instruction | None, ctx: PolicyContext) -> Instruction:
        if not self.targeted:
            return RETAKE_ANY
        if self.oracle_instruction:
            # read the true latent state: an upper bound, never a deployable policy
            factors = ctx.session.scene.factors
            worst = max(factors.items(), key=lambda kv: kv[1])[0]
            return instruction_for_factor(worst)
        return proposed or RETAKE_ANY


@dataclass
class SingleShot(CapturePolicy):
    """One photograph, no retakes. The status quo a phone screening app ships.

    Still uses the verdict machine, so it can decline to decide -- which is
    the only thing that keeps the comparison honest. A single-shot arm that
    was forced to answer would lose on accuracy for a reason that has nothing
    to do with capture policy.
    """

    strategy: BettingStrategy = field(default_factory=DegradationAwareBet)
    name: str = "single_shot"

    def run(self, ctx: PolicyContext, case_id: int) -> SessionResult:
        inner = EvidentialCapture(strategy=self.strategy)
        one_shot_ctx = PolicyContext(
            session=ctx.session,
            case=ctx.case,
            channel=ctx.channel,
            calibrator=ctx.calibrator,
            burden=ctx.burden,
            budget=1,
            rng=ctx.rng,
            n_strata=ctx.n_strata,
        )
        return inner.run(one_shot_ctx, case_id)


@dataclass
class FixedRetake(CapturePolicy):
    """Always take exactly K shots, untargeted, then decide on the total evidence.

    The burst-photography baseline: more frames, merged, no feedback loop.
    Isolates how much of the benefit is "more photons" versus "the right
    photons". In a persistent scene the answer should be: not much, because
    the room is still the room.
    """

    strategy: BettingStrategy = field(default_factory=DegradationAwareBet)
    name: str = "fixed_retake"

    def run(self, ctx: PolicyContext, case_id: int) -> SessionResult:
        inner = EvidentialCapture(targeted=False, strategy=self.strategy, early_stop=False)
        return inner.run(ctx, case_id)


@dataclass
class UntargetedEvidential(CapturePolicy):
    """Evidential stopping, but every subpoena is a bare "please retake"."""

    strategy: BettingStrategy = field(default_factory=DegradationAwareBet)
    name: str = "untargeted_evidential"

    def run(self, ctx: PolicyContext, case_id: int) -> SessionResult:
        return EvidentialCapture(targeted=False, strategy=self.strategy).run(ctx, case_id)


@dataclass
class OracleInstruction(CapturePolicy):
    """Evidential capture whose subpoenas read the true latent scene state.

    Not deployable -- it is the ceiling on instruction quality, and the gap to
    `evidential_capture` is exactly the cost of the confidence head's errors.
    """

    strategy: BettingStrategy = field(default_factory=DegradationAwareBet)
    name: str = "oracle_instruction"

    def run(self, ctx: PolicyContext, case_id: int) -> SessionResult:
        return EvidentialCapture(
            targeted=True, oracle_instruction=True, strategy=self.strategy
        ).run(ctx, case_id)


# ---------------------------------------------------------------------------
# Unsound arms: stopping depends on the diagnosis channel. These forfeit the
# guarantee, and the benchmark exists partly to show what that costs.
# ---------------------------------------------------------------------------


@dataclass
class GreedyDiagnostic(CapturePolicy):
    """Retake while the diagnosis looks ambiguous; report the last shot as one test.

    This is the natural implementation of "ask for another photo when the
    model is unsure", and it is invalid twice over. The stopping rule reads
    the diagnosis score, so the sequence of shots is selected on the very
    quantity being tested; and the final p-value is compared to alpha as
    though a single pre-planned test had been run, ignoring every earlier look.
    """

    ambiguity_band: float = 0.25
    name: str = "greedy_diagnostic"
    peeks: bool = True

    def run(self, ctx: PolicyContext, case_id: int) -> SessionResult:
        instruction: Instruction | None = None
        usabilities: list[float] = []
        scores: list[float] = []
        qualities: list[float] = []
        instructions: list[str] = []
        p_c = p_d = 1.0
        n = 0

        for shot_i in range(ctx.budget):
            capture = ctx.session.capture(instruction if shot_i > 0 else None)
            reading = ctx.channel.read(ctx.case, capture.severities, ctx.rng)
            n += 1
            usabilities.append(reading.usability)
            scores.append(reading.score)
            qualities.append(reading.true_quality)

            stratum = ctx.calibrator.stratum_of(reading.usability)
            p_c, p_d = ctx.calibrator.p_values(reading.score, stratum)

            # THE VIOLATION: the stopping rule reads the diagnosis score.
            decisive = abs(reading.score - 0.5) > self.ambiguity_band
            if decisive or shot_i == ctx.budget - 1:
                break

            worst = max(reading.degradation.items(), key=lambda kv: kv[1])[0]
            instruction = instruction_for_factor(DEGRADATION_TO_FACTOR.get(worst))
            instructions.append(instruction.name)

        outcome = _verdict_from_p(p_c, p_d, ctx.burden)
        return _result_from_p(
            case_id, ctx, outcome, n, p_c, p_d, usabilities, scores, qualities,
            instructions, peeked=True,
        )


@dataclass
class NaiveBestShot(CapturePolicy):
    """Take all K shots, then report the most favourable p-value among them.

    "Use the best photo" implemented literally. The selection is over K looks
    at the same null, so the reported p-value is a minimum of K p-values being
    compared against a threshold meant for one. Type-I error inflates roughly
    like 1 - (1 - alpha)^K.
    """

    name: str = "naive_best_shot"
    peeks: bool = True

    def run(self, ctx: PolicyContext, case_id: int) -> SessionResult:
        instruction: Instruction | None = None
        usabilities: list[float] = []
        scores: list[float] = []
        qualities: list[float] = []
        instructions: list[str] = []
        best_c = best_d = 1.0
        n = 0

        for shot_i in range(ctx.budget):
            capture = ctx.session.capture(instruction if shot_i > 0 else None)
            reading = ctx.channel.read(ctx.case, capture.severities, ctx.rng)
            n += 1
            usabilities.append(reading.usability)
            scores.append(reading.score)
            qualities.append(reading.true_quality)

            stratum = ctx.calibrator.stratum_of(reading.usability)
            p_c, p_d = ctx.calibrator.p_values(reading.score, stratum)
            # THE VIOLATION: keep the most favourable look, on both sides.
            best_c, best_d = min(best_c, p_c), min(best_d, p_d)

            worst = max(reading.degradation.items(), key=lambda kv: kv[1])[0]
            instruction = instruction_for_factor(DEGRADATION_TO_FACTOR.get(worst))
            instructions.append(instruction.name)

        outcome = _verdict_from_p(best_c, best_d, ctx.burden)
        return _result_from_p(
            case_id, ctx, outcome, n, best_c, best_d, usabilities, scores, qualities,
            instructions, peeked=True,
        )


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _verdict_from_p(p_convict: float, p_discharge: float, burden: BurdenSpec) -> VerdictOutcome:
    """Fixed-alpha verdict from raw p-values -- the pre-e-value way of deciding.

    Both sides passing at once is contradictory evidence, so it escalates,
    matching VerdictMachine's handling and keeping the comparison like-for-like.
    """
    met_c = p_convict <= burden.convict.alpha
    met_d = p_discharge <= burden.discharge.alpha
    if met_c and met_d:
        return VerdictOutcome.REFER
    if met_c:
        return VerdictOutcome.CARIES
    if met_d:
        return VerdictOutcome.SOUND
    return VerdictOutcome.REFER


def _correctness(outcome: VerdictOutcome, label: int) -> bool | None:
    if outcome is VerdictOutcome.CARIES:
        return label == 1
    if outcome is VerdictOutcome.SOUND:
        return label == 0
    return None


def _result(case_id, ctx, outcome, n, machine, usabilities, scores, qualities, instructions, peeked):
    return SessionResult(
        case_id=case_id,
        label=ctx.case.label,
        outcome=outcome,
        n_captures=n,
        correct=_correctness(outcome, ctx.case.label),
        peeked=peeked,
        wealth_convict=machine.convict.running_max,
        wealth_discharge=machine.discharge.running_max,
        final_usability=usabilities[-1] if usabilities else float("nan"),
        mean_true_quality=float(np.mean(qualities)) if qualities else float("nan"),
        instructions=tuple(instructions),
        usability_trace=tuple(usabilities),
        score_trace=tuple(scores),
    )


def _result_from_p(
    case_id, ctx, outcome, n, p_c, p_d, usabilities, scores, qualities, instructions, peeked
):
    """Result for arms that never build a wealth process.

    Their `wealth_*` columns carry 1/p, the closest comparable quantity, so
    the leaderboard's wealth columns stay populated. Flagged via `peeked` so
    nothing downstream mistakes them for calibrated e-values.
    """
    return SessionResult(
        case_id=case_id,
        label=ctx.case.label,
        outcome=outcome,
        n_captures=n,
        correct=_correctness(outcome, ctx.case.label),
        peeked=peeked,
        wealth_convict=float(1.0 / max(p_c, 1e-12)),
        wealth_discharge=float(1.0 / max(p_d, 1e-12)),
        final_usability=usabilities[-1] if usabilities else float("nan"),
        mean_true_quality=float(np.mean(qualities)) if qualities else float("nan"),
        instructions=tuple(instructions),
        usability_trace=tuple(usabilities),
        score_trace=tuple(scores),
    )


POLICIES: dict[str, type[CapturePolicy]] = {
    "single_shot": SingleShot,
    "fixed_retake": FixedRetake,
    "untargeted_evidential": UntargetedEvidential,
    "evidential_capture": EvidentialCapture,
    "oracle_instruction": OracleInstruction,
    "greedy_diagnostic": GreedyDiagnostic,
    "naive_best_shot": NaiveBestShot,
}


def policy_by_name(name: str, **kwargs) -> CapturePolicy:
    if name not in POLICIES:
        raise KeyError(f"unknown policy {name!r}; have {sorted(POLICIES)}")
    return POLICIES[name](**kwargs)
