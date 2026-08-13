"""Tests for the capture-session process (src/sim/)."""

from __future__ import annotations

import numpy as np
import pytest

from src.sim.instructions import RETAKE_ANY, REDUCE_GLARE, instruction_for_factor
from src.sim.render import render_severities
from src.sim.session import CaptureSession, SessionConfig
from src.sim.state import FACTORS, Operator, SceneState, sample_initial_scene


def test_scene_requires_all_factors():
    with pytest.raises(ValueError, match="missing factors"):
        SceneState(factors={"glare": 0.5})


def test_severities_and_usability_stay_in_range():
    rng = np.random.default_rng(0)
    for difficulty in (0.0, 0.5, 1.0):
        for _ in range(200):
            scene = sample_initial_scene(rng, difficulty)
            for name, sev in scene.severities().items():
                assert 0.0 <= sev <= 1.0, name
            assert 0.0 <= scene.usability() <= 1.0


def test_effective_glare_depends_on_where_the_hotspot_sits():
    """Same brightness, different position: only the overlapping one should hurt."""
    base = {f: 0.0 for f in FACTORS}
    base["glare"] = 0.9
    on_tooth = SceneState(factors=dict(base), glare_azimuth=0.0)
    off_tooth = SceneState(factors=dict(base), glare_azimuth=0.5)
    assert on_tooth.effective_glare() > 0.8
    assert off_tooth.effective_glare() < 0.01


def test_first_capture_cannot_follow_an_instruction():
    session = CaptureSession(rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="nothing yet to give feedback on"):
        session.capture(REDUCE_GLARE)


def test_untargeted_retake_does_not_fix_anything_on_average():
    """RETAKE_ANY buys a fresh draw of noise, not a change in the room."""
    rng = np.random.default_rng(1)
    deltas = []
    for _ in range(800):
        s = CaptureSession(rng=rng, difficulty=0.5)
        before = s.capture()
        after = s.capture(RETAKE_ANY)
        deltas.append(
            np.mean([after.state.factors[f] - before.state.factors[f] for f in FACTORS])
        )
    assert abs(float(np.mean(deltas))) < 0.02


def test_targeted_instruction_reduces_its_own_factor():
    rng = np.random.default_rng(2)
    for factor in FACTORS:
        deltas = []
        for _ in range(400):
            s = CaptureSession(rng=rng, difficulty=0.6)
            before = s.capture()
            after = s.capture(instruction_for_factor(factor))
            deltas.append(after.state.factors[factor] - before.state.factors[factor])
        assert float(np.mean(deltas)) < -0.02, f"{factor} did not improve"


def test_sessions_are_persistent_but_iid_mode_is_not():
    """The i.i.d. control must actually destroy the session's memory."""
    def lag1(config):
        rng = np.random.default_rng(3)
        traj = []
        for _ in range(600):
            s = CaptureSession(rng=rng, difficulty=0.5, config=config)
            row = [
                np.mean([s.capture(RETAKE_ANY if t else None).state.factors[f] for f in FACTORS])
                for t in range(4)
            ]
            traj.append(row)
        traj = np.array(traj)
        return float(np.corrcoef(traj[:, :-1].ravel(), traj[:, 1:].ravel())[0, 1])

    assert lag1(SessionConfig()) > 0.8
    assert abs(lag1(SessionConfig(iid_scene_per_shot=True))) < 0.15


def test_persistence_alone_does_not_produce_an_iid_world():
    """Guards the E4 ablation: rho=0 keeps the equilibrium, so memory survives."""
    rng = np.random.default_rng(4)
    traj = []
    for _ in range(600):
        s = CaptureSession(rng=rng, difficulty=0.5, config=SessionConfig(persistence=0.0))
        traj.append([
            np.mean([s.capture(RETAKE_ANY if t else None).state.factors[f] for f in FACTORS])
            for t in range(4)
        ])
    traj = np.array(traj)
    corr = float(np.corrcoef(traj[:, :-1].ravel(), traj[:, 1:].ravel())[0, 1])
    assert corr > 0.5, "rho=0 should still be correlated via the session equilibrium"


def test_fatigue_accumulates_and_saturates():
    op = Operator(skill=0.5, haste=0.5)
    for _ in range(50):
        op = op.tire()
    assert op.fatigue == pytest.approx(1.0)
    assert 0.0 <= op.compliance_probability() <= 1.0


def test_render_at_zero_severity_is_a_no_op():
    img = np.full((48, 48, 3), 128, np.uint8)
    out = render_severities(img, {n: 0.0 for n in ("blur", "glare", "angle", "low_light", "jpeg")})
    assert np.array_equal(out.image, img)
    assert out.severities == {}


def test_render_rejects_unknown_degradations_and_bad_shapes():
    img = np.full((32, 32, 3), 100, np.uint8)
    with pytest.raises(ValueError, match="unknown degradation"):
        render_severities(img, {"sparkle": 0.5})
    with pytest.raises(ValueError, match="BGR uint8"):
        render_severities(np.zeros((32, 32), np.uint8), {"blur": 0.5})


def test_render_remaps_boxes_through_geometry():
    """Only `angle` moves content, and boxes must move with it."""
    img = np.full((64, 64, 3), 120, np.uint8)
    boxes = np.array([[10.0, 10.0, 20.0, 20.0]])
    rng = np.random.default_rng(0)
    moved = render_severities(img, {"angle": 0.9}, boxes=boxes, rng=rng)
    assert moved.boxes is not None
    assert not np.allclose(moved.boxes, boxes)
    still = render_severities(img, {"blur": 0.9}, boxes=boxes, rng=rng)
    assert np.allclose(still.boxes, boxes)


def test_render_is_reproducible_given_a_generator():
    img = np.random.default_rng(0).integers(0, 255, (40, 40, 3), dtype=np.uint8)
    sev = {"glare": 0.6, "blur": 0.4}
    a = render_severities(img, sev, rng=np.random.default_rng(7)).image
    b = render_severities(img, sev, rng=np.random.default_rng(7)).image
    assert np.array_equal(a, b)
