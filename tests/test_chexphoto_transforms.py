"""Check the CheXphoto port against the vendored original implementation.

`src/data/chexphoto_transforms.py` claims to be a port of CheXphoto's released
synthetic transformations, not an interpretation of them. That claim is only
worth anything if something checks it, so this module runs the vendored
reference (`third_party/chexphoto/`, MIT) side by side with the port.

Two classes of check, because the transforms fall into two classes:

* **Deterministic** (blur, motion, brightness_down, contrast_down) -- compared
  pixel for pixel, with a tolerance for backend differences (PIL's Gaussian
  blur is a repeated-box approximation; OpenCV's is a true Gaussian kernel).
* **Stochastic** (glare, tilt, moire, rotation, translation) -- the reference
  draws from the global `numpy.random`/`random` state and the port from a
  passed `Generator`, so the same seed cannot produce the same draw. These are
  compared on summary statistics averaged over repeated applications: mean
  intensity, contrast, and the black-border fraction that the geometric
  transforms produce. A direction-of-warp bug -- the one real hazard in this
  port, since PIL applies perspective coefficients as a destination-to-source
  map -- shifts those statistics by tens of grey levels, far outside tolerance.

The reference needs Pillow and SciPy, which are deliberately not in
`requirements-core.txt`; every test here skips cleanly without them. The
port's own invariants are tested below without any reference dependency.
"""

from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.data.chexphoto_transforms import (
    LEVELS,
    PERTURBATION_NAMES,
    apply_perturbation,
    photographic,
    severity_to_level,
)

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "third_party" / "chexphoto"

pil = pytest.importorskip  # alias to keep the skip lines short


def _test_image(size: int = 128) -> np.ndarray:
    """A structured BGR image -- noise would hide warping and blur differences."""
    yy, xx = np.mgrid[0:size, 0:size]
    grey = (120 + 60 * np.sin(xx / 12.0) * np.cos(yy / 17.0)).astype(np.uint8)
    return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)


def _load_reference(name: str):
    """Import one vendored reference module, shimming the APIs it predates.

    Pillow 10 removed `Image.ANTIALIAS` and NumPy 1.24 removed `np.float`; the
    reference is from 2020 and uses both. Shimming at import time keeps
    `third_party/` a verbatim copy of upstream.
    """
    Image = pytest.importorskip("PIL.Image")
    pytest.importorskip("scipy.stats")
    if not hasattr(Image, "ANTIALIAS"):
        Image.ANTIALIAS = Image.LANCZOS
    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]

    path = REFERENCE_DIR / f"{name}.py"
    if not path.exists():
        pytest.skip(f"vendored reference {path} not present")
    spec = importlib.util.spec_from_file_location(f"chexphoto_ref_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _to_pil(img_bgr: np.ndarray):
    Image = pytest.importorskip("PIL.Image")
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def _from_pil(img) -> np.ndarray:
    return cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def _summary(img: np.ndarray) -> np.ndarray:
    """[mean intensity, contrast, black-pixel fraction] -- the port's failure modes."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return np.array([float(g.mean()), float(g.std()), float((g < 5).mean())])


# --------------------------------------------------------------------------
# Deterministic transforms: compared pixel for pixel
# --------------------------------------------------------------------------

DETERMINISTIC = [
    # (name, reference function, max mean-absolute pixel difference)
    ("blur", "blur_mapping", 1.5),
    ("motion", "motion_mapping", 0.01),
    ("brightness_down", "brightness_down_mapping", 1.0),
    ("contrast_down", "contrast_down_mapping", 1.0),
]


@pytest.mark.parametrize("name, ref_fn, tol", DETERMINISTIC)
@pytest.mark.parametrize("level", LEVELS)
def test_deterministic_transforms_match_reference(name, ref_fn, tol, level):
    module = _load_reference(name)
    img = _test_image()

    np.random.seed(0)
    reference = _from_pil(getattr(module, ref_fn)(level, _to_pil(img)))
    ported = apply_perturbation(img, name, level, np.random.default_rng(0))

    assert ported.shape == reference.shape
    mad = float(np.abs(reference.astype(np.float64) - ported.astype(np.float64)).mean())
    assert mad <= tol, f"{name} L{level}: mean abs diff {mad:.3f} > {tol}"


# --------------------------------------------------------------------------
# Stochastic transforms: compared on summary statistics
# --------------------------------------------------------------------------

STOCHASTIC = [
    # (name, reference function, tolerances on [mean, std, black fraction])
    ("glare_matte", "glare_matte_mapping", (4.0, 3.0, 0.05)),
    ("glare_glossy", "glare_glossy_mapping", (4.0, 3.0, 0.05)),
    ("tilt", "tilt_mapping", (4.0, 3.0, 0.05)),
    ("rotation", "rotation_mapping", (4.0, 3.0, 0.05)),
    ("moire", "moire_mapping", (4.0, 4.0, 0.05)),
    ("translation", "translation_mapping", (9.0, 6.0, 0.09)),
]

N_DRAWS = 12


@pytest.mark.parametrize("name, ref_fn, tol", STOCHASTIC)
@pytest.mark.parametrize("level", (1, 3))
def test_stochastic_transforms_match_reference_in_distribution(name, ref_fn, tol, level):
    module = _load_reference(name)
    img = _test_image()
    h, w = img.shape[:2]

    np.random.seed(7)
    random.seed(7)
    ref_stats = []
    for _ in range(N_DRAWS):
        out = _from_pil(getattr(module, ref_fn)(level, _to_pil(img)))
        if out.shape[:2] != (h, w):  # only `rotation` expands the canvas
            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_AREA)
        ref_stats.append(_summary(out))

    rng = np.random.default_rng(3)
    port_stats = [_summary(apply_perturbation(img, name, level, rng)) for _ in range(N_DRAWS)]

    ref_mean = np.mean(ref_stats, axis=0)
    port_mean = np.mean(port_stats, axis=0)
    diff = np.abs(ref_mean - port_mean)
    assert np.all(diff <= np.asarray(tol)), (
        f"{name} L{level}: reference {np.round(ref_mean, 2)} vs port {np.round(port_mean, 2)}"
    )


# --------------------------------------------------------------------------
# Port invariants -- no reference dependency, always run
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", PERTURBATION_NAMES)
def test_every_perturbation_preserves_shape_and_dtype(name):
    img = _test_image(96)
    out = apply_perturbation(img, name, 2, np.random.default_rng(0))
    assert out.shape == img.shape
    assert out.dtype == np.uint8


@pytest.mark.parametrize("name", ["blur", "motion", "glare_matte", "moire", "tilt"])
def test_damage_increases_with_level(name):
    """Higher CheXphoto level must mean a more corrupted image, monotonically.

    Measured as distance from the clean image, averaged over draws so the
    stochastic transforms are not judged on one lucky sample.
    """
    img = _test_image()
    rng = np.random.default_rng(11)
    distances = []
    for level in LEVELS:
        d = [
            float(np.abs(apply_perturbation(img, name, level, rng).astype(np.float64) - img).mean())
            for _ in range(6)
        ]
        distances.append(float(np.mean(d)))
    assert distances == sorted(distances), f"{name} not monotone in level: {np.round(distances, 2)}"


def test_photographic_recipe_applies_all_three_stages():
    """The released recipe is glare_matte -> moire -> tilt; all three must bite."""
    img = _test_image()
    out = photographic(img, 3, np.random.default_rng(5))
    assert out.shape == img.shape
    # tilt leaves black borders, glare pushes the mean up, moire adds banding
    g = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    assert (g < 5).mean() > 0.05, "no black border -- tilt stage did not apply"
    assert not np.array_equal(out, img)


def test_unknown_perturbation_and_level_are_rejected():
    img = _test_image(64)
    with pytest.raises(KeyError):
        apply_perturbation(img, "not_a_transform", 1)
    with pytest.raises(ValueError):
        apply_perturbation(img, "blur", 0)
    with pytest.raises(ValueError):
        apply_perturbation(img, "blur", 5)


def test_severity_to_level_maps_zero_to_no_perturbation():
    assert severity_to_level(0.0) is None
    assert severity_to_level(-0.1) is None
    assert severity_to_level(0.01) == 1
    assert severity_to_level(0.25) == 1
    assert severity_to_level(0.26) == 2
    assert severity_to_level(0.75) == 3
    assert severity_to_level(1.0) == 4
    assert severity_to_level(2.0) == 4


def test_rejects_non_bgr_input():
    with pytest.raises(ValueError):
        apply_perturbation(np.zeros((32, 32), dtype=np.uint8), "blur", 1)
    with pytest.raises(ValueError):
        apply_perturbation(np.zeros((32, 32, 3), dtype=np.float32), "blur", 1)
