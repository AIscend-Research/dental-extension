"""Runnable checks for the degradation pipeline. `pytest -q` or run directly."""

import numpy as np

from src.data.degradation import (
    DEGRADATION_NAMES,
    apply_degradations,
    make_burst,
    blur,
    jpeg,
)


def _demo_image(size=256):
    grad = np.tile(np.linspace(0, 255, size, dtype=np.uint8), (size, 1))
    return np.stack([grad] * 3, axis=-1)


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
        test_apply_degradations_rejects_non_3channel_input,
    ]:
        fn()
        print("PASS", fn.__name__)
