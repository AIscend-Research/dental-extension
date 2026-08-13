"""Runnable checks for the degradation pipeline. `pytest -q` or run directly."""

import numpy as np

from src.data.degradation import (
    DEGRADATION_NAMES,
    angle_with_matrix,
    apply_degradations,
    make_burst,
    blur,
    jpeg,
    transform_boxes,
)


def _demo_image(size=256):
    grad = np.tile(np.linspace(0, 255, size, dtype=np.uint8), (size, 1))
    return np.stack([grad] * 3, axis=-1)


def _box_image(size=256):
    """Black image with one white square, so a box can be checked against pixels."""
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
    # every applied degradation shows up as a positive severity in the vector
    for name, sev in r.severities.items():
        assert vec[DEGRADATION_NAMES.index(name)] == np.float32(sev)


def test_seed_is_reproducible():
    img = _demo_image()
    a = apply_degradations(img, seed=42)
    b = apply_degradations(img, seed=42)
    assert np.array_equal(a.image, b.image)
    assert a.severities == b.severities


def test_burst_returns_requested_frames():
    img = _demo_image()
    shots = make_burst(img, n_shots=3, seed=7)
    assert len(shots) == 3
    assert all(s.image.shape == img.shape for s in shots)


def test_identity_matrix_leaves_boxes_untouched():
    boxes = np.array([[10.0, 20.0, 30.0, 40.0]])
    out = transform_boxes(boxes, np.eye(3), 256, 256)
    assert np.allclose(out, boxes)
    # empty input must not blow up (images with no annotations are common)
    assert transform_boxes(np.zeros((0, 4)), np.eye(3), 256, 256).shape == (0, 4)


def test_boxes_follow_the_angle_warp():
    """The bug this guards: angle() moves pixels, so boxes must move with it.

    Degrading an image without remapping its ground truth trains the detector
    against silently misaligned labels -- no crash, just a worse model.
    """
    img, boxes = _box_image()
    warped, matrix = angle_with_matrix(img, severity=0.8)
    moved = transform_boxes(boxes, matrix, img.shape[1], img.shape[0])

    # the warp actually did something, so a static box would now be wrong
    assert not np.allclose(moved, boxes, atol=1.0)

    # the remapped box must still contain the white square's new location
    ys, xs = np.where(warped[:, :, 0] > 128)
    assert len(xs) > 0, "white square vanished -- fixture is wrong, not the code"
    x, y, w, h = moved[0]
    assert x <= xs.min() + 2 and x + w >= xs.max() - 2
    assert y <= ys.min() + 2 and y + h >= ys.max() - 2


def test_apply_degradations_returns_remapped_boxes():
    img, boxes = _box_image()
    r = apply_degradations(img, which=["angle"], severity_range=(0.7, 0.9), seed=3, boxes=boxes)
    assert r.boxes is not None and r.boxes.shape == (1, 4)
    assert not np.allclose(r.boxes, boxes, atol=1.0)
    # boxes stay inside the frame
    assert r.boxes[0][0] >= 0 and r.boxes[0][1] >= 0
    assert r.boxes[0][0] + r.boxes[0][2] <= img.shape[1] + 1e-6
    assert r.boxes[0][1] + r.boxes[0][3] <= img.shape[0] + 1e-6


def test_non_geometric_degradations_leave_boxes_alone():
    img, boxes = _box_image()
    r = apply_degradations(
        img, which=["blur", "jpeg", "low_light"], severity_range=(0.5, 0.9), seed=5, boxes=boxes
    )
    assert np.allclose(r.boxes, boxes)


def test_boxes_are_opt_in():
    img, _ = _box_image()
    assert apply_degradations(img, seed=0).boxes is None


def test_burst_gives_each_shot_its_own_boxes():
    # angle jitter is independent per frame, so one shared box array would be
    # wrong for at least two of the three frames
    img, boxes = _box_image()
    shots = make_burst(img, n_shots=3, seed=11, boxes=boxes)
    assert all(s.boxes is not None and s.boxes.shape == (1, 4) for s in shots)


def test_apply_degradations_rejects_non_3channel_input():
    # glare() silently corrupts a 2D (H, W) grayscale image into (H, W, H) via
    # a numpy broadcasting accident instead of erroring -- apply_degradations
    # must fail loudly on the bad shape rather than let that through.
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
        test_burst_returns_requested_frames,
        test_identity_matrix_leaves_boxes_untouched,
        test_boxes_follow_the_angle_warp,
        test_apply_degradations_returns_remapped_boxes,
        test_non_geometric_degradations_leave_boxes_alone,
        test_boxes_are_opt_in,
        test_burst_gives_each_shot_its_own_boxes,
        test_apply_degradations_rejects_non_3channel_input,
    ]:
        fn()
        print("PASS", fn.__name__)
