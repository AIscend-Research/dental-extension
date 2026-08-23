"""One-step lookahead over the coupling matrix: a smarter subpoena rule.

E3 found that `oracle_instruction` -- which reads the *true* latent scene and
still just names whichever factor is currently worst -- does not beat the
deployable `evidential_capture` arm. That is a real negative result about the
*decision rule*, not about the confidence head: both arms use the same greedy
policy ("subpoena the worst factor"), so perfect information about the current
scene cannot help a rule that never looks past the correction it is about to
ask for. `docs/simulator_grounding.md`'s coupling matrix predicts exactly the
failure mode this leaves on the table: fixing glare can push tilt and darkness
up (glare -> tilt 0.35, glare -> darkness 0.30), so "fix the worst factor" can
choose a correction that trades one problem for a bigger one.

This module scores each candidate instruction by *projecting* its expected
effect one step ahead -- using the known coupling structure and a population
operator model, not the true operator -- and picks whichever leaves the shot
most usable in expectation. It never reads anything the deployable arm could
not read: the projection is a function of `EvidenceView.degradation` and
`EvidenceView.shot_index`, both already G_t-measurable, so a policy built on
it inherits the anytime-validity guarantee unchanged (see
docs/theory_anytime_validity.md, A2-A3).

What the projection deliberately does NOT model: the true operator's actual
skill/haste/fatigue (unobservable, and using it would be peeking at something
outside G_t in spirit even if not in the letter -- the whole point is a
policy that would work on an operator it has never met), and the glare
repositioning mechanic (a complied REDUCE_GLARE has some chance of moving the
hotspot rather than dimming it, which the degradation channel's *effective*
glare reading cannot distinguish after the fact). Both are named
simplifications, not silent ones.
"""

from __future__ import annotations

import numpy as np

from src.models.diagnostic import predicted_usability
from src.sim.instructions import Instruction, instruction_for_factor
from src.sim.state import COUPLING, DEFAULT_FATIGUE_PER_SHOT, FACTOR_TO_DEGRADATION, FACTORS, Operator

# The operator this module projects against: population defaults (Operator's
# own field defaults), never the session's true operator. Fatigue is the one
# state that IS legally knowable -- it is a deterministic function of
# shot_index, which is part of EvidenceView -- so it is set per call rather
# than defaulted.
_POP_SKILL = Operator().skill
_POP_HASTE = Operator().haste


def _expected_removal_fraction(fatigue: float) -> float:
    """E[compliance] * E[correction strength] for the population operator.

    `Operator.correction_strength` draws Beta(a, b) with a = mean*conc,
    b = (1-mean)*conc, so its mean is exactly `mean` regardless of
    concentration -- no sampling needed to get the expectation right.
    """
    compliance = float(np.clip((0.45 + 0.5 * _POP_SKILL) * (1.0 - 0.4 * fatigue), 0.0, 1.0))
    mean_strength = 0.25 + 0.55 * _POP_SKILL
    return compliance * mean_strength


def _degradation_to_factors(degradation: dict[str, float]) -> dict[str, float]:
    return {factor: float(degradation.get(deg_name, 0.0)) for factor, deg_name in FACTOR_TO_DEGRADATION.items()}


def _factors_to_degradation(factors: dict[str, float], template: dict[str, float]) -> dict[str, float]:
    out = dict(template)
    for factor, deg_name in FACTOR_TO_DEGRADATION.items():
        out[deg_name] = float(np.clip(factors[factor], 0.0, 1.0))
    return out


def project_degradation(
    degradation: dict[str, float],
    target_factor: str | None,
    shot_index: int,
    coupling_scale: float = 1.0,
) -> dict[str, float]:
    """Project next-shot degradation if `target_factor` is corrected.

    `target_factor=None` (the untargeted control) returns `degradation`
    unchanged, matching `RETAKE_ANY`'s real effect: nothing was asked for, so
    nothing about the scene moves except process noise, which this
    deterministic projection does not model (it would wash out in
    expectation anyway).
    """
    if target_factor is None:
        return dict(degradation)

    # `shot_index` is the shot just captured (0-indexed); the correction
    # this projects lands on the *next* shot, i.e. after shot_index+1 shots
    # have tired the operator (session.py ticks fatigue once per completed
    # capture, starting from 0). Using shot_index directly here understated
    # fatigue by one DEFAULT_FATIGUE_PER_SHOT for every candidate alike.
    fatigue = float(np.clip((shot_index + 1) * DEFAULT_FATIGUE_PER_SHOT, 0.0, 1.0))
    removal = _expected_removal_fraction(fatigue)

    factors = _degradation_to_factors(degradation)
    before = factors.get(target_factor, 0.0)
    after = before * (1.0 - removal)
    magnitude = before - after
    factors[target_factor] = after

    for affected, leak in COUPLING.get(target_factor, {}).items():
        bump = leak * coupling_scale * _POP_HASTE * magnitude
        factors[affected] = float(np.clip(factors.get(affected, 0.0) + bump, 0.0, 1.0))

    return _factors_to_degradation(factors, degradation)


def best_lookahead_instruction(degradation: dict[str, float], shot_index: int) -> Instruction:
    """The subpoena whose *projected* next-shot usability is highest.

    Candidates are the four targetable factors plus the untargeted control
    (projected usability = current usability, i.e. "no correction helps more
    than nothing" is a real possible answer, not excluded by construction).
    Ties keep the untargeted control, which is the conservative choice: it
    costs nothing but a capture, while a targeted instruction that turns out
    not to help still paid the coupling cost.
    """
    baseline = predicted_usability(degradation)
    best_factor: str | None = None
    best_usability = baseline
    for factor in FACTORS:
        projected = project_degradation(degradation, factor, shot_index)
        u = predicted_usability(projected)
        if u > best_usability:
            best_usability = u
            best_factor = factor
    return instruction_for_factor(best_factor)
