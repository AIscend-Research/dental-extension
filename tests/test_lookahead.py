"""Tests for the one-step lookahead instruction policy (src/sim/lookahead.py)."""

from __future__ import annotations

from src.models.diagnostic import predicted_usability
from src.sim.instructions import RETAKE_ANY
from src.sim.lookahead import best_lookahead_instruction, project_degradation
from src.sim.state import COUPLING, FACTOR_TO_DEGRADATION, FACTORS


def _flat_degradation(value: float = 0.5) -> dict[str, float]:
    return {name: value for name in FACTOR_TO_DEGRADATION.values()} | {"jpeg": value}


def test_untargeted_projection_is_unchanged():
    deg = _flat_degradation(0.6)
    projected = project_degradation(deg, None, shot_index=0)
    assert projected == deg
    # must be a copy, not the same object, so callers can't mutate the input by accident
    assert projected is not deg


def test_targeted_projection_reduces_the_named_factor():
    deg = _flat_degradation(0.6)
    projected = project_degradation(deg, "tremor", shot_index=0)
    blur_name = FACTOR_TO_DEGRADATION["tremor"]
    assert projected[blur_name] < deg[blur_name]


def test_targeted_projection_leaks_into_coupled_factors():
    """Fixing glare should bump tilt/darkness up, per COUPLING's glare row."""
    deg = _flat_degradation(0.6)
    projected = project_degradation(deg, "glare", shot_index=0)
    tilt_name = FACTOR_TO_DEGRADATION["tilt"]
    darkness_name = FACTOR_TO_DEGRADATION["darkness"]
    assert COUPLING["glare"]["tilt"] > 0  # sanity check on the fixture itself
    assert projected[tilt_name] > deg[tilt_name]
    assert projected[darkness_name] > deg[darkness_name]


def test_projection_never_leaves_the_unit_interval():
    deg = _flat_degradation(0.95)
    for factor in FACTORS:
        projected = project_degradation(deg, factor, shot_index=0)
        for v in projected.values():
            assert 0.0 <= v <= 1.0


def test_higher_shot_index_means_a_more_fatigued_operator_helps_less():
    """Fatigue erodes compliance, so the same correction should improve
    usability less at a late shot than an early one."""
    deg = _flat_degradation(0.6)
    early = project_degradation(deg, "tremor", shot_index=0)
    late = project_degradation(deg, "tremor", shot_index=10)
    blur_name = FACTOR_TO_DEGRADATION["tremor"]
    assert early[blur_name] <= late[blur_name]


def test_best_instruction_targets_the_factor_that_helps_most():
    # tremor is badly degraded and cheap to fix (low coupling leakage back onto
    # it from other corrections); everything else is already clean.
    deg = {name: 0.05 for name in FACTOR_TO_DEGRADATION.values()} | {"jpeg": 0.05}
    deg[FACTOR_TO_DEGRADATION["tremor"]] = 0.9
    instruction = best_lookahead_instruction(deg, shot_index=0)
    assert instruction.target == "tremor"


def test_best_instruction_is_untargeted_when_nothing_helps():
    # a pristine shot: no correction can improve on doing nothing
    deg = _flat_degradation(0.0)
    instruction = best_lookahead_instruction(deg, shot_index=0)
    assert instruction is RETAKE_ANY


def test_best_instruction_never_decreases_projected_usability():
    """Whatever it picks, the projected usability must be >= doing nothing --
    the function should never recommend a correction that makes things worse
    on its own projection."""
    deg = _flat_degradation(0.6)
    baseline = predicted_usability(deg)
    instruction = best_lookahead_instruction(deg, shot_index=0)
    projected = project_degradation(deg, instruction.target, shot_index=0)
    assert predicted_usability(projected) >= baseline - 1e-9
