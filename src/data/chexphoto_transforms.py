"""CheXphoto's photographic corruption model, ported to numpy/OpenCV.

This is a port of the synthetic transformations released with CheXphoto
(Phillips, Rajpurkar et al., ML4H 2020) -- `transforms/*.py` in
https://github.com/stanfordmlgroup/cheXphoto, MIT License, vendored verbatim
under `third_party/chexphoto/` for reference and for the equivalence test.

**Why port instead of import.** The reference implementation is PIL + SciPy
and depends on three APIs removed from current Pillow/NumPy (`Image.ANTIALIAS`,
`np.float`, and `Image` resampling constants). This project's runtime
dependency set is numpy + OpenCV (`requirements-core.txt`), and every other
image path in `src/` is BGR uint8. Porting keeps one image convention and one
dependency set; `tests/test_chexphoto_transforms.py` checks the port against
the vendored original so "ported" is a verified claim rather than an assertion.

**Why it exists at all.** E13 tested this project's framework on a second
modality using this project's *own* capture simulator. That leaves the obvious
objection open: the simulator and the method were designed together, so a
result under that simulator could be an artifact of it. Running the same
framework under a *third party's* corruption model -- one written by the
authors of the closest prior work, for a different modality, before this
project existed -- is the test that objection actually calls for. See
`experiments/e14_chexphoto_headtohead.py`.

**Fidelity, stated precisely.** Level parameters (blur radii, moire gaps and
opacities, glare covariances, tilt degrees, brightness/contrast factors) are
copied exactly from the reference. What differs is the resampling and
filtering backend: PIL's Gaussian blur, LANCZOS resize and integer-rounded
`ImageEnhance` arithmetic are not bit-identical to OpenCV's. The port is
therefore numerically close, not bit-exact, and the test asserts closeness
with stated tolerances rather than equality.

**What this is not.** It is not the CheXphoto *dataset*. The natural
photographs (Nokia10k, iPhone1k, and the 250-image film subset) are gated
behind a Stanford data agreement -- see `docs/chexphoto_access.md`. This
module covers the synthetic half of CheXphoto, which is the half that was
released as code and can therefore be applied to any radiograph.
"""

from __future__ import annotations

import cv2
import numpy as np

# CheXphoto's own severity axis: the reference exposes exactly four levels.
LEVELS = (1, 2, 3, 4)

# The paper's "photographic synthetic" recipe, in order:
#   python synthesize.py --perturbation glare_matte --perturbation2 moire \
#                        --perturbation3 tilt
PHOTOGRAPHIC_RECIPE = ("glare_matte", "moire", "tilt")


def _check(img: np.ndarray) -> np.ndarray:
    if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
        raise ValueError(f"expected BGR uint8 (H, W, 3), got shape {img.shape} dtype {img.dtype}")
    return img


def _rng(rng: np.random.Generator | None) -> np.random.Generator:
    return np.random.default_rng() if rng is None else rng


def _alpha_composite_white(img: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Composite a white layer with per-pixel alpha in [0, 255] over `img`.

    The reference does this with PIL's `alpha_composite` onto an RGBA copy;
    over an opaque base that reduces to the standard `src*(1-a) + 255*a`.
    """
    a = (alpha.astype(np.float32) / 255.0)[:, :, None]
    out = img.astype(np.float32) * (1.0 - a) + 255.0 * a
    return np.clip(out, 0, 255).astype(np.uint8)


def identity(img: np.ndarray, level: int = 1, rng: np.random.Generator | None = None) -> np.ndarray:
    return _check(img).copy()


def blur(img: np.ndarray, level: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Gaussian defocus. Reference radii: 1.5 / 3 / 6 / 10 by level.

    PIL's `ImageFilter.GaussianBlur(radius=r)` treats `radius` as the standard
    deviation, so the port passes it as OpenCV's sigma with an auto-sized
    kernel.
    """
    _check(img)
    radius = {1: 1.5, 2: 3.0, 3: 6.0, 4: 10.0}[int(level)]
    return cv2.GaussianBlur(img, (0, 0), sigmaX=radius, sigmaY=radius, borderType=cv2.BORDER_REPLICATE)


def motion(img: np.ndarray, level: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Horizontal motion blur. Reference kernel sizes: 2 / 10 / 25 / 45."""
    _check(img)
    size = {1: 2, 2: 10, 3: 25, 4: 45}[int(level)]
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[(size - 1) // 2, :] = 1.0 / size
    return cv2.filter2D(img, -1, kernel)


def glare_matte(img: np.ndarray, level: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Soft specular highlight: a 2-D Gaussian white mask at a random centre.

    Reference parameters, unchanged: isotropic covariance `(50*level)**2`, the
    PDF renormalised to a peak of `100*level`, and the whole alpha channel
    clamped at `150 + 20*level`.
    """
    _check(img)
    rng = _rng(rng)
    level = int(level)
    h, w = img.shape[:2]
    cx = rng.uniform(0, w)
    cy = rng.uniform(0, h)
    var = float((level * 50) ** 2)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Isotropic Gaussian: the normalising constant cancels in the rescaling
    # below, so only the exponent matters.
    quad = ((xx - cx) ** 2 + (yy - cy) ** 2) / var
    pdf = np.exp(-0.5 * quad)
    peak = float(pdf.max())
    if peak <= 0:  # centre far outside a small image; nothing to composite
        return img.copy()
    alpha = np.minimum(150.0 + 20.0 * level, pdf * (100.0 * level) / peak)
    return _alpha_composite_white(img, alpha)


def glare_glossy(img: np.ndarray, level: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Hard-edged specular block in one of nine screen positions."""
    _check(img)
    rng = _rng(rng)
    level = int(level)
    h, w = img.shape[:2]
    box_w = level * 0.1 * w + rng.uniform(0, 0.05) * w
    box_h = level * 0.1 * h + rng.uniform(0, 0.05) * h
    value = 150.0 + 20.0 * level + rng.uniform(0, 20)
    location = int(rng.integers(1, 10))

    alpha = np.zeros((h, w), dtype=np.float32)
    bw, bh = int(box_w), int(box_h)

    def band(center_frac: float, extent: int, limit: int) -> tuple[int, int]:
        start = int(center_frac * limit - extent / 2)
        return max(0, start), max(0, start + extent)

    # The reference indexes rows with `height_box` and columns with `width_box`
    # for the corner cases, and uses fractional centres for the edge/centre
    # ones. Kept as-is, including its own axis conventions.
    if location == 1:  # top left
        alpha[:bh, :bw] = value
    elif location == 2:  # left centre
        r0, r1 = band(rng.uniform(0.3, 0.7), bh, h)
        alpha[r0:r1, :bw] = value
    elif location == 3:  # bottom left
        alpha[h - bh : h, :bw] = value
    elif location == 4:  # top centre
        c0, c1 = band(rng.uniform(0.3, 0.7), bw, w)
        alpha[:bh, c0:c1] = value
    elif location == 5:  # centre
        r0, r1 = band(rng.uniform(0.3, 0.7), bh, h)
        c0, c1 = band(rng.uniform(0.3, 0.7), bw, w)
        alpha[r0:r1, c0:c1] = value
    elif location == 6:  # bottom centre
        c0, c1 = band(rng.uniform(0.3, 0.7), bw, w)
        alpha[h - bh : h, c0:c1] = value
    elif location == 7:  # top right
        alpha[:bh, w - bw : w] = value
    elif location == 8:  # right centre
        r0, r1 = band(rng.uniform(0.3, 0.7), bh, h)
        alpha[r0:r1, w - bw : w] = value
    else:  # bottom right
        alpha[h - bh : h, w - bw : w] = value

    return _alpha_composite_white(img, alpha)


def brightness_down(img: np.ndarray, level: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Darkening. Reference factor: `1 / (1 + f*0.4 + U(-0.01, 0.01))`.

    `f` is 0.5 at level 1 and the level itself above that -- a discontinuity in
    the reference (level 2's factor is 2, not 1), preserved deliberately.
    """
    _check(img)
    rng = _rng(rng)
    level = int(level)
    f = 0.5 if level == 1 else float(level)
    factor = 1.0 / (1.0 + f * 0.4 + rng.uniform(-0.01, 0.01))
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def contrast_down(img: np.ndarray, level: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Contrast reduction toward the image's mean grey, as PIL's ImageEnhance."""
    _check(img)
    rng = _rng(rng)
    level = int(level)
    f = 0.5 if level == 1 else float(level)
    factor = 1.0 / (1.0 + f * 0.4 + rng.uniform(-0.01, 0.01))
    # PIL blends against a flat image filled with the rounded mean of the
    # L-converted (grayscale) image.
    grey = float(np.round(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()))
    out = grey + (img.astype(np.float32) - grey) * factor
    return np.clip(out, 0, 255).astype(np.uint8)


def tilt(img: np.ndarray, level: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Off-axis capture: perspective warp with corners jittered by 0.05*level."""
    _check(img)
    rng = _rng(rng)
    level = int(level)
    h, w = img.shape[:2]
    d = level * 0.05
    src = np.float32(
        [
            [w * rng.uniform(0, d), h * rng.uniform(0, d)],
            [w * rng.uniform(1 - d, 1), h * rng.uniform(0, d)],
            [w * rng.uniform(1 - d, 1), h * rng.uniform(1 - d, 1)],
            [w * rng.uniform(0, d), h * rng.uniform(1 - d, 1)],
        ]
    )
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    # PIL applies perspective coefficients as a destination -> source map, so
    # the reference's `find_coeffs(jittered, corners)` zooms the frame *out*
    # and leaves black borders. WARP_INVERSE_MAP reproduces that direction;
    # warping forward instead would silently collapse the tilt to a no-op.
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h),
                               flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def rotation(img: np.ndarray, level: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """In-plane rotation by 5*level degrees, canvas expanded, black borders.

    Unlike the rest, this changes the output size -- the reference does too.
    Callers that need a fixed size (the channel below does) resize afterwards.
    """
    _check(img)
    rng = _rng(rng)
    h, w = img.shape[:2]
    rot = 5.0 * int(level) * (-1.0 if rng.integers(0, 2) else 1.0)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), rot, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w // 2
    M[1, 2] += nh / 2 - h // 2
    return cv2.warpAffine(img, M, (nw, nh))


def translation(img: np.ndarray, level: int, rng: np.random.Generator | None = None,
                buffer: int = 100) -> np.ndarray:
    """Framing error: shift by 20*level px inside a `buffer` px border, then rescale.

    The reference hardcodes a 100 px border, which is fine for full-size chest
    radiographs and degenerate on small crops -- on a 96 px crop it pushes most
    of the frame off-image (>68% black at level 1). The default keeps the
    reference's value so the equivalence test compares like with like; E14 does
    not use this perturbation for exactly that reason.
    """
    _check(img)
    rng = _rng(rng)
    level = int(level)
    h, w = img.shape[:2]
    dx = level * 20 * (-1 if rng.integers(0, 2) else 1)
    dy = level * 20 * (-1 if rng.integers(0, 2) else 1)
    src = np.float32(
        [
            [dx + buffer, dy + buffer],
            [w - buffer + dx, buffer + dy],
            [w - buffer + dx, h - buffer + dy],
            [buffer + dx, h - buffer + dy],
        ]
    )
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(src, dst)  # dst -> src, as in `tilt`
    return cv2.warpPerspective(img, M, (w, h),
                               flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def _moire_base_mask(size: int, thickness: int, gap: float, opacity: float, darkness: float) -> np.ndarray:
    """Parallel semi-transparent lines: (size, size, 2) of [colour, alpha]."""
    rows = np.arange(size)
    # The reference computes `remainder(arange, thickness + gap) < thickness`
    # with a float gap, so line spacing is non-integer by design.
    dark = np.remainder(rows, thickness + gap) < thickness
    alpha = np.where(dark, opacity * 255.0, 0.0).astype(np.float32)
    colour = np.full(size, (1.0 - darkness) * 255.0, dtype=np.float32)
    mask = np.zeros((size, size, 2), dtype=np.float32)
    mask[:, :, 0] = colour[:, None]
    mask[:, :, 1] = alpha[:, None]
    return mask


def _warp_moire_mask(mask: np.ndarray, out_w: int, out_h: int, angle: float, spread: float,
                     offset: tuple[float, float]) -> np.ndarray:
    """Warp, rotate, offset and crop the line mask, as the reference does."""
    mh, mw = mask.shape[:2]
    src = np.float32([[0, 0], [mw, 0], [mw, mh], [0, mh]])
    dst = np.float32(
        [
            [0, mh * (0.5 - spread / 2)],
            [mw, 0],
            [mw, mh],
            [0, mh * (0.5 + spread / 2)],
        ]
    )
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(mask, M, (mw, mh), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    R = cv2.getRotationMatrix2D((mw / 2, mh / 2), angle, 1.0)
    rotated = cv2.warpAffine(warped, R, (mw, mh), flags=cv2.INTER_LINEAR)

    left = int((mw - out_w) // 2 + offset[0])
    upper = int((mh - out_h) // 2 + offset[1])
    left = int(np.clip(left, 0, max(0, mw - out_w)))
    upper = int(np.clip(upper, 0, max(0, mh - out_h)))
    return rotated[upper : upper + out_h, left : left + out_w]


def moire(img: np.ndarray, level: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Screen-photography interference: two warped line masks over an upsampled image.

    Reference (gap, opacity) by level: (20, 0.05) / (4.5, 0.15) / (2.5, 0.35) /
    (1, 0.5), with `upsample_factor=2`, `thickness=1`, `darkness=1.0`, and two
    masks near 90 degrees. Upsampling before compositing and downsampling after
    is what produces the actual beat pattern -- it is not an implementation
    detail, so it is kept.

    This artifact has no counterpart in this project's own five-artifact
    taxonomy, which was built for photographing films on a lightbox rather than
    photographing a monitor (`docs/simulator_grounding.md`).
    """
    _check(img)
    rng = _rng(rng)
    level = int(level)
    gap, opacity = {1: (20.0, 0.05), 2: (4.5, 0.15), 3: (2.5, 0.35), 4: (1.0, 0.5)}[level]

    h, w = img.shape[:2]
    up = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
    uh, uw = up.shape[:2]
    mask_dim = max(uh, uw) * 2
    base = _moire_base_mask(mask_dim, thickness=1, gap=gap, opacity=opacity, darkness=1.0)

    params = [
        (90.0, 0.5, (rng.uniform(0, 100), rng.uniform(0, 100))),
        (90.0 + rng.normal(0, 1), 0.5, (rng.uniform(0, 100), rng.uniform(0, 100))),
    ]
    out = up.astype(np.float32)
    for angle, spread, offset in params:
        m = _warp_moire_mask(base, uw, uh, angle, spread, offset)
        a = (m[:, :, 1] / 255.0)[:, :, None]
        colour = m[:, :, 0][:, :, None]
        out = out * (1.0 - a) + colour * a

    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_LANCZOS4)


PERTURBATIONS = {
    "identity": identity,
    "blur": blur,
    "motion": motion,
    "glare_matte": glare_matte,
    "glare_glossy": glare_glossy,
    "brightness_down": brightness_down,
    "contrast_down": contrast_down,
    "tilt": tilt,
    "rotation": rotation,
    "translation": translation,
    "moire": moire,
}

PERTURBATION_NAMES = list(PERTURBATIONS)


def apply_perturbation(img: np.ndarray, name: str, level: int,
                       rng: np.random.Generator | None = None) -> np.ndarray:
    """Apply one named CheXphoto perturbation at one of its four levels."""
    if name not in PERTURBATIONS:
        raise KeyError(f"unknown CheXphoto perturbation {name!r}; have {PERTURBATION_NAMES}")
    if int(level) not in LEVELS:
        raise ValueError(f"CheXphoto levels are {LEVELS}, got {level}")
    out = PERTURBATIONS[name](img, int(level), rng)
    if out.shape[:2] != img.shape[:2]:  # only `rotation` expands the canvas
        out = cv2.resize(out, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
    return out


def photographic(img: np.ndarray, level: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """CheXphoto's released photographic recipe: glare_matte, then moire, then tilt."""
    out = img
    for name in PHOTOGRAPHIC_RECIPE:
        out = apply_perturbation(out, name, level, rng)
    return out


def severity_to_level(severity: float) -> int | None:
    """Map a [0, 1] severity from this project's simulator onto CheXphoto's 1-4.

    CheXphoto's axis is ordinal with four rungs and no zero; this project's is
    continuous with 0.0 meaning "no artifact". The mapping splits [0, 1] into
    four equal bands and returns None below the first, so a zero severity means
    "do not apply this perturbation" rather than "apply the mildest one".

    It is a stated convention, not a measured correspondence -- E14 measures
    the actual damage each rung causes rather than assuming this mapping makes
    the two axes comparable.
    """
    s = float(severity)
    if s <= 0.0:
        return None
    return int(np.clip(np.ceil(s * len(LEVELS)), 1, len(LEVELS)))
