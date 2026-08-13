"""Runnable checks for the albumentations degradation ablation arm.

Mirrors tests/test_degradation.py's checks so both pipelines are held to the
same contract (shape/dtype preserved, severity 0 is a no-op, seeded runs are
reproducible), since that contract is what makes them swappable for the
Phase 2 ablation comparison (see demo_degradation_compare.py).
"""

import numpy as np

from src.data.degradation import DEGRADATION_NAMES
from src.data.degradation_albumentations import (
    angle_with_boxes,
    apply_degradations,
    blur,
    jpeg,
)


def _demo_image(size=256):
    grad = np.tile(np.linspace(0, 255, size, dtype=np.uint8), (size, 1))
    return np.stack([grad] * 3, axis=-1)


def _box_image(size=256):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[100:150, 80:130] = 255
    return img, np.array([[80.0, 100.0, 50.0, 50.0]])  # COCO x, y, w, h


def test_output_shape_and_dtype_preserved():
    img = _demo_image()
    r = apply_degradations(img, seed=0)
    assert r.image.shape == img.shape
    assert r.image.dtype == np.uint8


def test_severity_zero_is_noop():
    img = _demo_image()
    assert np.array_equal(blur(img, 0.0), img)
    assert np.array_equal(jpeg(img, 0.0), img)


def test_labels_align_to_registry():
    img = _demo_image()
    r = apply_degradations(img, seed=1)
    vec = r.label_vector()
    assert len(vec) == len(DEGRADATION_NAMES)
    for name, sev in r.severities.items():
        assert vec[DEGRADATION_NAMES.index(name)] == np.float32(sev)


def test_seed_is_reproducible():
    # albumentations transforms do NOT respect global random/np seeding --
    # this locks in that apply_degradations' own seeding scheme (routing
    # through A.Compose(..., seed=...) per call) actually works.
    img = _demo_image()
    a = apply_degradations(img, seed=42)
    b = apply_degradations(img, seed=42)
    assert np.array_equal(a.image, b.image)
    assert a.severities == b.severities


def test_boxes_follow_the_affine_warp():
    """Same box contract as the hand-built arm -- the two must stay swappable."""
    img, boxes = _box_image()
    warped, moved, kept = angle_with_boxes(img, 0.8, boxes, seed=0)
    assert len(kept) == len(moved)
    if len(moved):  # albumentations drops fully out-of-frame boxes
        ys, xs = np.where(warped[:, :, 0] > 128)
        if len(xs):
            x, y, w, h = moved[0]
            assert x <= xs.min() + 3 and x + w >= xs.max() - 3
            assert y <= ys.min() + 3 and y + h >= ys.max() - 3


def test_apply_degradations_returns_remapped_boxes():
    img, boxes = _box_image()
    r = apply_degradations(img, which=["angle"], severity_range=(0.7, 0.9), seed=3, boxes=boxes)
    assert r.boxes is not None
    assert r.boxes.shape[1] == 4 if len(r.boxes) else True


def test_non_geometric_degradations_leave_boxes_alone():
    img, boxes = _box_image()
    r = apply_degradations(
        img, which=["blur", "jpeg", "low_light"], severity_range=(0.5, 0.9), seed=5, boxes=boxes
    )
    assert np.allclose(r.boxes, boxes)


def test_boxes_are_opt_in():
    img, _ = _box_image()
    assert apply_degradations(img, seed=0).boxes is None


def test_rejects_non_3channel_input():
    grayscale = np.zeros((64, 64), dtype=np.uint8)
    try:
        apply_degradations(grayscale, seed=0)
        assert False, "expected ValueError for non-3-channel input"
    except ValueError:
        pass


if __name__ == "__main__":
    for fn in [
        test_output_shape_and_dtype_preserved,
        test_severity_zero_is_noop,
        test_labels_align_to_registry,
        test_seed_is_reproducible,
        test_boxes_follow_the_affine_warp,
        test_apply_degradations_returns_remapped_boxes,
        test_non_geometric_degradations_leave_boxes_alone,
        test_boxes_are_opt_in,
        test_rejects_non_3channel_input,
    ]:
        fn()
        print("PASS", fn.__name__)
