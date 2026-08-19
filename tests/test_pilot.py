"""Checks for the phone-pilot machinery. `pytest -q tests/test_pilot.py`.

There are no photographs to test against -- that is the whole point of the
pilot -- so these tests do the next best thing: they synthesise a "real" image
with the simulator itself and check that the fitting machinery recovers the
severity it was given. That validates the recovery, not the realism. Realism
is what the photographs decide, and no test here can stand in for them.
"""

import csv

import cv2
import numpy as np
import pytest

from src.data.degradation import DEGRADATION_NAMES
from src.pilot.plan import (
    CONDITIONS,
    Condition,
    build_shot_list,
    coverage,
    read_manifest,
    write_manifest,
)
from src.pilot.realism import (
    SENSITIVE_STATS,
    STAT_NAMES,
    _render,
    artifact_stats,
    compare_arms,
    cross_arm_calibration,
    fit_angle_severity,
    fit_severity,
    stat_distance,
)
from src.pilot.registration import register_photo


def _radiograph(h=400, w=560, seed=0):
    """A stand-in with the texture registration and the statistics need.

    Flat noise has no repeatable ORB features and a uniform spectrum; blobs on
    a gradient give both, which is enough to exercise every path here.
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w), np.float32)
    img += np.linspace(40, 140, w)[None, :]
    for _ in range(60):
        cx, cy = rng.integers(0, w), rng.integers(0, h)
        cv2.circle(img, (int(cx), int(cy)), int(rng.integers(6, 30)),
                   float(rng.uniform(60, 220)), -1)
    img = cv2.GaussianBlur(img, (5, 5), 1.5)
    img += rng.normal(0, 4, img.shape)
    return np.stack([np.clip(img, 0, 255).astype(np.uint8)] * 3, axis=-1)


# --- plan -----------------------------------------------------------------


def test_shot_list_covers_every_modelled_factor():
    shots = build_shot_list(["a.png", "b.png"], seed=0)
    covered = {s.stresses for s in shots if s.stresses}
    # jpeg is applied in transport, not with the hands -- see CONDITIONS.
    assert covered == set(DEGRADATION_NAMES) - {"jpeg"}
    assert len(shots) == 2 * sum(c.replicates for c in CONDITIONS)


def test_condition_order_is_shuffled_within_a_film_but_ids_are_stable():
    a = build_shot_list(["a.png"], seed=1)
    b = build_shot_list(["a.png"], seed=2)
    assert [s.condition for s in a] != [s.condition for s in b]
    assert sorted(s.shot_id for s in a) == sorted(s.shot_id for s in b)
    assert [s.order for s in a] == list(range(1, len(a) + 1))


def test_condition_rejects_an_unmodelled_factor():
    with pytest.raises(ValueError, match="not a known degradation"):
        Condition("moire_severe", "moire", "photograph a printed halftone")


def test_manifest_round_trips_with_operator_columns(tmp_path):
    shots = build_shot_list(["a.png"], seed=0)
    shots[0].filename, shots[0].device = "IMG_0001.jpg", "Pixel 7a"
    path = write_manifest(shots, tmp_path / "manifest.csv")
    assert read_manifest(path) == shots


def test_generated_manifest_leaves_operator_fields_empty(tmp_path):
    path = write_manifest(build_shot_list(["a.png"], seed=0), tmp_path / "m.csv")
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    assert all(r["filename"] == "" and r["device"] == "" for r in rows)


def test_coverage_flags_factors_with_no_photographs():
    shots = build_shot_list(["a.png"], seed=0)
    for shot in shots:
        if shot.stresses != "glare":
            shot.filename = f"{shot.shot_id}.jpg"
    cov = coverage(shots)
    assert cov.uncovered_factors == ["glare"]
    assert not cov.complete
    assert cov.taken == cov.planned - 2  # glare_mild + glare_severe


# --- statistics -----------------------------------------------------------


@pytest.mark.parametrize("name,stat,direction", [
    ("blur", "sharpness", -1),
    ("blur", "hf_ratio", -1),
    ("glare", "glare_area", +1),
    ("low_light", "luma_median", -1),
    ("jpeg", "blockiness", +1),
])
def test_each_statistic_moves_the_way_its_degradation_should(name, stat, direction):
    img = _radiograph()
    clean = artifact_stats(img)[stat]
    heavy = artifact_stats(_render("opencv", name, img, 0.8, seed=3))[stat]
    assert direction * (heavy - clean) > 0, f"{name} did not move {stat} as expected"


def test_stat_distance_is_zero_for_identical_images_and_positive_otherwise():
    img = _radiograph()
    stats = artifact_stats(img)
    assert stat_distance(stats, stats) == 0.0
    assert stat_distance(stats, artifact_stats(_render("opencv", "blur", img, 0.7, seed=1))) > 0


# --- severity fitting -----------------------------------------------------


@pytest.mark.parametrize("name,true_severity", [
    ("blur", 0.3), ("blur", 0.8), ("low_light", 0.5), ("jpeg", 0.9), ("glare", 0.7),
])
def test_fit_recovers_the_severity_it_was_rendered_at(name, true_severity):
    img = _radiograph()
    # A different seed from the sweep's, so the fit cannot win by matching the
    # exact random draw (glare position, motion-blur direction) rather than
    # the severity.
    real = _render("opencv", name, img, true_severity, seed=101)
    fit = fit_severity(real, img, name, arm="opencv", seed=0)
    assert abs(fit.severity - true_severity) <= 0.2, fit.curve
    assert fit.identifiable


def test_fit_refuses_angle_and_says_what_to_use_instead():
    img = _radiograph()
    with pytest.raises(ValueError, match="fit_angle_severity"):
        fit_severity(img, img, "angle")


def test_fit_refuses_unregistered_images():
    img = _radiograph()
    with pytest.raises(ValueError, match="register the photograph"):
        fit_severity(cv2.resize(img, (280, 200)), img, "blur")


def test_per_stat_gap_reports_every_statistic_not_just_the_fitted_ones():
    img = _radiograph()
    fit = fit_severity(_render("opencv", "blur", img, 0.5, seed=101), img, "blur")
    assert set(fit.per_stat_gap) == set(STAT_NAMES)


def test_a_gradient_the_simulator_cannot_produce_shows_up_as_an_unfitted_gap():
    """The check that lets the pilot find a *missing* degradation.

    Neither arm models one-sided illumination, so a photograph with a
    brightness ramp should leave a large `illum_gradient` gap even after the
    fitted statistics are matched as well as the arm can match them.
    """
    img = _radiograph()
    ramp = np.linspace(-40, 40, img.shape[1], dtype=np.float32)[None, :, None]
    lit = np.clip(img.astype(np.float32) + ramp, 0, 255).astype(np.uint8)
    lit_gap = abs(fit_severity(lit, img, "blur").per_stat_gap["illum_gradient"])
    clean_gap = abs(fit_severity(img, img, "blur").per_stat_gap["illum_gradient"])
    assert clean_gap < 0.02, "a clean image should leave essentially no gradient gap"
    assert lit_gap > 0.1, "the ramp should survive the best fit any arm can make"


# --- registration ---------------------------------------------------------


def test_registration_recovers_a_known_perspective_warp():
    ref = _radiograph()
    h, w = ref.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[90, 60], [w + 40, 30], [w + 70, h + 20], [60, h + 55]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    photo = cv2.warpPerspective(ref, matrix, (w + 160, h + 120), borderValue=(25, 25, 25))

    result = register_photo(photo, ref)
    assert result.ok, result.reason
    assert result.reprojection_error < 2.0
    assert result.warped.shape == ref.shape
    # The warp should put the film back where it started, background removed.
    assert np.corrcoef(result.warped[..., 0].ravel(), ref[..., 0].ravel())[0, 1] > 0.9


def test_registration_fails_loudly_on_an_unrelated_image():
    result = register_photo(_radiograph(seed=1), _radiograph(seed=2))
    assert not result.ok
    assert result.reason
    assert result.warped is None or result.n_inliers < 25


def test_angle_severity_is_recovered_from_the_homography():
    ref = _radiograph()
    for true_severity in (0.2, 0.9):
        photo = _render("opencv", "angle", ref, true_severity, seed=7)
        reg = register_photo(photo, ref)
        assert reg.ok, reg.reason
        fit = fit_angle_severity(reg.homography, ref.shape[:2])
        assert abs(fit.severity - true_severity) <= 0.2, fit.curve


# --- arm comparison and calibration ---------------------------------------


def test_compare_arms_fits_both_arms_and_only_calls_a_winner_when_it_is_clear():
    ref = _radiograph()
    pairs = [(_render("opencv", "blur", ref, 0.6, seed=200 + i), ref) for i in range(3)]
    cmp_ = compare_arms(pairs, "blur")
    assert cmp_.n_pairs == 3
    assert set(cmp_.median_residual) == {"opencv", "albumentations"}
    # Identical residuals must not be reported as a win either way.
    cmp_.median_residual = {"opencv": 1.0, "albumentations": 1.05}
    assert cmp_.winner is None
    cmp_.median_residual = {"opencv": 1.0, "albumentations": 2.0}
    assert cmp_.winner == "opencv"


def test_cross_arm_calibration_returns_a_severity_map():
    rows = cross_arm_calibration(_radiograph(), "blur", severities=(0.3, 0.7))
    assert [r[0] for r in rows] == [0.3, 0.7]
    assert all(0.0 <= r[1] <= 1.0 for r in rows)
    # Monotone: a more severe source should not match a *less* severe target.
    assert rows[1][1] >= rows[0][1]


def test_every_modelled_degradation_has_sensitive_statistics_or_a_stated_reason():
    assert set(SENSITIVE_STATS) == set(DEGRADATION_NAMES) - {"angle"}
    for names in SENSITIVE_STATS.values():
        assert set(names) <= set(STAT_NAMES)
