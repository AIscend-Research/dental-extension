"""Measure how far the synthetic arms are from a real photograph.

The pilot's scientific question is narrow and answerable: for each artifact the
simulator models, does a photograph taken under that condition look like the
simulator's version of it, and if so at what severity? This module answers it
with image statistics rather than opinion, because "the OpenCV glare looks more
like a lightbox reflection" (`TASKS.md`, Stream 1) is currently the strongest
claim in the repo and it is an eyeball judgement on one image.

How the fit works. Each degradation drives a small set of statistics --
blur destroys high-frequency energy, glare raises the bright tail, JPEG puts
an 8-pixel grid into the gradients. `fit_severity` sweeps the simulator's
severity, renders the source radiograph at each step, and picks the severity
whose statistics land closest to the photograph's, measured only on the
statistics that degradation is supposed to move. The residual at that best
severity is the honest number: it says how close the simulator can get *at its
best setting*, which is the question, rather than how close it happens to be
at some default.

Two limits, stated here because they bound what the pilot can conclude:

- **Geometry is invisible to these statistics.** Registration warps the
  photograph back into the source frame, which by construction undoes the
  off-axis capture. `angle` severity is therefore recovered from the
  registration homography (`fit_angle_severity`), not from pixel statistics,
  and `fit_severity` refuses it rather than silently returning a fit that
  measures nothing.
- **A small residual is not proof of realism.** These statistics are
  summaries; two images can match on all ten and still differ in ways a
  detector cares about. A *large* residual is strong evidence against an arm;
  a small one is weak evidence for it. The corresponding strong test is
  downstream -- whether a detector's accuracy on real photographs matches its
  accuracy on synthetic ones -- and it needs the trained detector.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Sequence

import cv2
import numpy as np

from src.data import degradation as opencv_arm

#: Statistics, in a fixed order. Everything downstream (vectors, distances,
#: report tables) is aligned to this tuple.
STAT_NAMES: tuple[str, ...] = (
    "sharpness",       # log Laplacian variance -- destroyed by blur
    "hf_ratio",        # share of FFT power above quarter-Nyquist
    "glare_area",      # fraction of near-saturated pixels
    "glare_excess",    # bright tail above the median
    "luma_median",     # overall exposure
    "noise_sigma",     # robust high-frequency noise estimate
    "blockiness",      # 8-pixel DCT grid visible in the gradients
    "moire",           # periodic spectral peak, from print halftone/screen beat
    "contrast",        # global standard deviation
    "illum_gradient",  # low-frequency brightness ramp across the frame
)

#: Which statistics each degradation is supposed to move. Fits are scored on
#: these only; scoring on all ten would let an unrelated mismatch (a print's
#: illumination ramp, say) pick the severity of the blur.
SENSITIVE_STATS: dict[str, tuple[str, ...]] = {
    "blur": ("sharpness", "hf_ratio"),
    "glare": ("glare_area", "glare_excess", "luma_median"),
    "low_light": ("luma_median", "noise_sigma", "contrast"),
    "jpeg": ("blockiness", "hf_ratio"),
    # "angle" is absent on purpose -- see the module docstring.
}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32) / 255.0


def _blockiness(gray: np.ndarray) -> float:
    """Energy on the 8-pixel JPEG grid, over energy off it.

    JPEG quantises 8x8 DCT blocks independently, so the block boundaries stop
    agreeing with each other and every eighth gradient column is larger than
    its neighbours. Differencing on-grid against off-grid cancels the image's
    own content, which is the point -- a bare gradient magnitude would mostly
    measure how detailed the radiograph is.
    """
    dh = np.abs(np.diff(gray, axis=1))
    dv = np.abs(np.diff(gray, axis=0))
    on_h = dh[:, 7::8].mean() if dh.shape[1] > 8 else 0.0
    on_v = dv[7::8, :].mean() if dv.shape[0] > 8 else 0.0
    off_h = np.delete(dh, np.arange(7, dh.shape[1], 8), axis=1).mean() if dh.shape[1] > 8 else 0.0
    off_v = np.delete(dv, np.arange(7, dv.shape[0], 8), axis=0).mean() if dv.shape[0] > 8 else 0.0
    return float(((on_h - off_h) + (on_v - off_v)) / 2.0)


def _spectrum(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Power spectrum and the matching normalised radius, both flattened."""
    windowed = gray * np.outer(np.hanning(gray.shape[0]), np.hanning(gray.shape[1]))
    power = np.abs(np.fft.fftshift(np.fft.fft2(windowed))) ** 2
    h, w = gray.shape
    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None] * 2.0   # 1.0 == Nyquist
    fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :] * 2.0
    return power.ravel(), np.sqrt(fy ** 2 + fx ** 2).ravel()


def _hf_ratio(power: np.ndarray, radius: np.ndarray) -> float:
    total = power.sum()
    if total <= 0:
        return 0.0
    return float(power[radius > 0.25].sum() / total)


def _moire(power: np.ndarray, radius: np.ndarray) -> float:
    """How far the strongest mid-frequency peak stands above its band.

    Print halftone screens and the beat between a screen and the sensor grid
    put a *narrow* peak into the mid-band; ordinary anatomy and ordinary noise
    are broad there. The peak-over-median ratio is what separates them, and it
    is the one artifact in this table that no arm in the repo models at all --
    which makes it the statistic most likely to produce a real finding.
    """
    band = (radius > 0.15) & (radius < 0.60)
    if not band.any():
        return 0.0
    values = power[band]
    median = float(np.median(values))
    if median <= 0:
        return 0.0
    return float(np.log10(values.max() / median))


def _illum_gradient(gray: np.ndarray) -> float:
    """Amplitude of the best-fit brightness plane, in units of image range.

    A photograph lit from one side is brighter on that side. Neither synthetic
    arm produces this: `glare` is a local blob and `low_light` is a global
    gamma. If the pilot's photographs show a consistent ramp, that is a
    missing degradation, not a tuning error.
    """
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    ys, xs = np.mgrid[0:32, 0:32]
    design = np.stack([xs.ravel() / 31.0, ys.ravel() / 31.0, np.ones(32 * 32)], axis=1)
    coeffs, *_ = np.linalg.lstsq(design, small.ravel(), rcond=None)
    return float(np.hypot(coeffs[0], coeffs[1]))


def artifact_stats(img: np.ndarray) -> dict[str, float]:
    """Ten scale-sensitive artifact statistics for one image.

    Args:
        img: BGR uint8 (or 2D grayscale uint8).

    Returns:
        Dict keyed by `STAT_NAMES`.

    Note: several of these depend on pixel scale -- `blockiness` needs the
    native 8-pixel grid, `sharpness` and `hf_ratio` change under resampling.
    Only compare statistics between images of the same size, which is what
    `src.pilot.registration` produces (it warps every photograph into its
    source radiograph's frame).
    """
    gray = _gray(img)
    power, radius = _spectrum(gray)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    smoothed = cv2.medianBlur((gray * 255).astype(np.uint8), 3).astype(np.float32) / 255.0
    residual = gray - smoothed
    return {
        "sharpness": float(np.log1p(lap.var() * 1e4)),
        "hf_ratio": _hf_ratio(power, radius),
        "glare_area": float((gray >= 0.92).mean()),
        "glare_excess": float(np.percentile(gray, 99.5) - np.median(gray)),
        "luma_median": float(np.median(gray)),
        "noise_sigma": float(1.4826 * np.median(np.abs(residual - np.median(residual)))),
        "blockiness": _blockiness(gray),
        "moire": _moire(power, radius),
        "contrast": float(gray.std()),
        "illum_gradient": _illum_gradient(gray),
    }


def stat_vector(stats: dict[str, float] | np.ndarray) -> np.ndarray:
    """Stats dict -> array in `STAT_NAMES` order (arrays pass through)."""
    if isinstance(stats, np.ndarray):
        return stats.astype(np.float64)
    return np.array([stats[name] for name in STAT_NAMES], dtype=np.float64)


def stat_distance(
    a: dict[str, float] | np.ndarray,
    b: dict[str, float] | np.ndarray,
    scale: np.ndarray | None = None,
    names: Sequence[str] | None = None,
) -> float:
    """Mean absolute difference over `names`, in units of `scale`.

    The statistics have wildly different natural ranges (`sharpness` is a log,
    `glare_area` is a fraction), so an unscaled distance would be whichever
    statistic happens to be largest. `scale` is normally the spread of each
    statistic across the severity sweep being fitted, which makes the distance
    read as "how many sweeps-worth of difference".
    """
    names = tuple(STAT_NAMES if names is None else names)
    idx = [STAT_NAMES.index(n) for n in names]
    va, vb = stat_vector(a)[idx], stat_vector(b)[idx]
    denom = np.ones(len(idx)) if scale is None else np.asarray(scale, dtype=np.float64)[idx]
    denom = np.where(denom > 1e-12, denom, 1.0)
    return float(np.mean(np.abs(va - vb) / denom))


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

ArmFn = Callable[[np.ndarray, float], np.ndarray]


def get_arm(arm: str) -> dict[str, ArmFn]:
    """Look up a degradation arm by name: "opencv" or "albumentations".

    Albumentations is imported lazily so this module stays usable when only
    the OpenCV arm is installed -- the same reason `src/models/*.py` guard
    their torch imports.
    """
    if arm == "opencv":
        return opencv_arm.DEGRADATIONS
    if arm == "albumentations":
        from src.data import degradation_albumentations as albu_arm

        return albu_arm.DEGRADATIONS
    raise ValueError(f"unknown arm {arm!r}; expected 'opencv' or 'albumentations'")


def _render(arm: str, name: str, img: np.ndarray, severity: float, seed: int) -> np.ndarray:
    """Apply one degradation at one severity, reproducibly.

    The OpenCV arm draws from the `random` and `numpy.random` globals (glare
    position, motion-blur direction, per-corner warp), so seeding those is the
    only way to make a severity sweep repeatable; the albumentations arm takes
    a seed argument instead. Both paths are covered here so callers do not
    have to know which arm they are on.
    """
    fns = get_arm(arm)
    if name not in fns:
        raise ValueError(f"arm {arm!r} has no degradation {name!r}")
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    try:
        return fns[name](img, severity, seed=seed)  # type: ignore[call-arg]
    except TypeError:
        return fns[name](img, severity)


def _mean_stats(arm: str, name: str, img: np.ndarray, severity: float,
                repeats: int, seed: int) -> np.ndarray:
    """Stat vector averaged over `repeats` draws at one severity.

    Both arms randomise placement -- glare lands somewhere different every
    call -- so a single draw at severity 0.5 is a noisy estimate of what
    severity 0.5 means. Averaging is what makes the sweep monotone enough to
    fit against.
    """
    return np.mean(
        [stat_vector(artifact_stats(_render(arm, name, img, severity, seed + i)))
         for i in range(repeats)],
        axis=0,
    )


# ---------------------------------------------------------------------------
# Fitting a severity to a real photograph
# ---------------------------------------------------------------------------

DEFAULT_GRID: tuple[float, ...] = tuple(round(x, 2) for x in np.linspace(0.0, 1.0, 11))


@dataclass
class SeverityFit:
    """The severity at which one arm best explains one photograph."""

    degradation: str
    arm: str
    severity: float
    #: Distance at the best severity, in sweep-spread units. This is the
    #: realism number: how close the arm gets when tuned in its favour.
    residual: float
    #: (severity, distance) over the whole grid. Worth plotting -- a flat
    #: curve means the fit is unidentifiable, which is a different failure
    #: from a high residual and reads the same in a single number.
    curve: list[tuple[float, float]] = field(default_factory=list)
    #: Signed real-minus-synthetic gap on every statistic, not only the fitted
    #: ones. The unfitted entries are where both arms may be missing something.
    per_stat_gap: dict[str, float] = field(default_factory=dict)

    @property
    def identifiable(self) -> bool:
        """Does the sweep actually distinguish severities for this photograph?"""
        if len(self.curve) < 3:
            return False
        distances = [d for _, d in self.curve]
        return (max(distances) - min(distances)) > 0.5 * min(distances)


def fit_severity(
    real: np.ndarray,
    reference: np.ndarray,
    degradation: str,
    arm: str = "opencv",
    grid: Sequence[float] = DEFAULT_GRID,
    repeats: int = 3,
    seed: int = 0,
) -> SeverityFit:
    """Find the severity at which `arm` reproduces `real` from `reference`.

    Args:
        real: the registered photograph (BGR uint8), already warped into the
            reference frame by `src.pilot.registration.register_photo`.
        reference: the source radiograph the film was printed from.
        degradation: a key of `SENSITIVE_STATS`. "angle" is rejected -- see
            `fit_angle_severity`.
        arm: "opencv" or "albumentations".
        grid: severities to try.
        repeats: draws averaged per severity, to damp the arms' internal
            randomness.
        seed: base seed for those draws.

    Raises:
        ValueError: on an unknown or geometric degradation, or if the two
            images differ in size (register first -- statistics computed at
            different scales are not comparable).
    """
    if degradation == "angle":
        raise ValueError(
            "angle severity cannot be fitted from registered statistics -- "
            "registration undoes the geometry by construction. "
            "Use fit_angle_severity(homography, shape) instead."
        )
    if degradation not in SENSITIVE_STATS:
        raise ValueError(f"no sensitive statistics defined for {degradation!r}")
    if real.shape[:2] != reference.shape[:2]:
        raise ValueError(
            f"real {real.shape[:2]} and reference {reference.shape[:2]} differ in size; "
            "register the photograph onto the reference before fitting"
        )

    names = SENSITIVE_STATS[degradation]
    real_stats = artifact_stats(real)
    sweep = np.stack([_mean_stats(arm, degradation, reference, s, repeats, seed)
                      for s in grid])
    # Scale each statistic by how much this sweep moves it, so a statistic the
    # degradation barely touches cannot dominate the distance.
    scale = sweep.std(axis=0)
    curve = [(float(s), stat_distance(real_stats, sweep[i], scale, names))
             for i, s in enumerate(grid)]
    best_i = int(np.argmin([d for _, d in curve]))
    best_vec = sweep[best_i]
    return SeverityFit(
        degradation=degradation,
        arm=arm,
        severity=float(grid[best_i]),
        residual=curve[best_i][1],
        curve=curve,
        per_stat_gap={n: float(stat_vector(real_stats)[i] - best_vec[i])
                      for i, n in enumerate(STAT_NAMES)},
    )


def _angle_descriptor(matrix: np.ndarray, shape: tuple[int, int]) -> float:
    """How far a homography moves corners, beyond what a similarity explains.

    Scale and in-plane rotation are not evidence of an off-axis shot -- they
    are just where the photographer stood and how they held the phone. What
    marks an off-axis capture is the part a similarity transform *cannot*
    absorb: the trapezoid. This returns that residual as a fraction of the
    frame diagonal.
    """
    h, w = shape
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(corners, matrix.astype(np.float64)).reshape(4, 2)
    similarity, _ = cv2.estimateAffinePartial2D(corners.reshape(4, 2), mapped)
    if similarity is None:
        return float("nan")
    fitted = cv2.transform(corners, similarity).reshape(4, 2)
    return float(np.mean(np.linalg.norm(mapped - fitted, axis=1)) / np.hypot(w, h))


def fit_angle_severity(
    homography: np.ndarray,
    shape: tuple[int, int],
    grid: Sequence[float] = DEFAULT_GRID,
    repeats: int = 9,
    seed: int = 0,
) -> SeverityFit:
    """Recover `angle` severity from the registration homography.

    The registration step already measured the photograph's geometry: the
    homography that maps the photograph onto its source *is* the off-axis
    capture, inverted. This compares its trapezoid residual against the
    residuals `src.data.degradation.angle_with_matrix` produces at each
    severity, and returns the closest.

    Args:
        homography: photo -> reference, from `RegistrationResult.homography`.
        shape: (height, width) of the reference.
        grid / repeats / seed: as `fit_severity`. `repeats` is higher here
            because the simulator draws all four corner offsets independently,
            so a single draw is a poor estimate of a severity's typical
            trapezoid.
    """
    observed = _angle_descriptor(np.linalg.inv(np.asarray(homography, dtype=np.float64)), shape)
    curve: list[tuple[float, float]] = []
    for s in grid:
        draws = []
        for i in range(repeats):
            random.seed(seed + i)
            np.random.seed((seed + i) % (2 ** 32))
            _, matrix = opencv_arm.angle_with_matrix(
                np.zeros((shape[0], shape[1], 3), np.uint8), float(s))
            draws.append(_angle_descriptor(matrix, shape))
        curve.append((float(s), abs(observed - float(np.median(draws)))))
    best_i = int(np.argmin([d for _, d in curve]))
    return SeverityFit(
        degradation="angle",
        arm="opencv",
        severity=float(grid[best_i]),
        residual=curve[best_i][1],
        curve=curve,
        per_stat_gap={"corner_residual_fraction": observed},
    )


# ---------------------------------------------------------------------------
# Arm comparison and cross-arm calibration
# ---------------------------------------------------------------------------


@dataclass
class ArmComparison:
    """Both arms fitted to the same photographs, for one degradation."""

    degradation: str
    n_pairs: int
    #: arm -> median residual at its own best severity. Lower is more real.
    median_residual: dict[str, float] = field(default_factory=dict)
    #: arm -> median fitted severity, i.e. where the real photographs sit on
    #: that arm's severity scale. Two arms disagreeing here is the calibration
    #: mismatch `TASKS.md` records for low_light, now as a number.
    median_severity: dict[str, float] = field(default_factory=dict)
    fits: dict[str, list[SeverityFit]] = field(default_factory=dict)

    @property
    def winner(self) -> str | None:
        """The arm with the lower median residual, or None if they tie.

        "Tie" is within 10% of the better arm -- with a pilot of tens of
        photographs, a smaller difference than that is not a result.
        """
        if len(self.median_residual) < 2:
            return None
        ranked = sorted(self.median_residual.items(), key=lambda kv: kv[1])
        (best, best_v), (_, next_v) = ranked[0], ranked[1]
        return best if next_v > 1.1 * best_v else None


def compare_arms(
    pairs: Sequence[tuple[np.ndarray, np.ndarray]],
    degradation: str,
    arms: Sequence[str] = ("opencv", "albumentations"),
    **fit_kwargs,
) -> ArmComparison:
    """Fit every arm to every (registered photo, reference) pair.

    This is the measurement that replaces `TASKS.md`'s eyeballed
    "OpenCV's glare looks more like a real lightbox reflection" with a
    comparison that can come out either way.
    """
    fits: dict[str, list[SeverityFit]] = {}
    for arm in arms:
        fits[arm] = [fit_severity(real, ref, degradation, arm=arm, **fit_kwargs)
                     for real, ref in pairs]
    return ArmComparison(
        degradation=degradation,
        n_pairs=len(pairs),
        median_residual={a: float(np.median([f.residual for f in fs])) if fs else float("nan")
                         for a, fs in fits.items()},
        median_severity={a: float(np.median([f.severity for f in fs])) if fs else float("nan")
                         for a, fs in fits.items()},
        fits=fits,
    )


def cross_arm_calibration(
    reference: np.ndarray,
    degradation: str,
    severities: Sequence[float] = (0.2, 0.4, 0.6, 0.8, 1.0),
    grid: Sequence[float] = DEFAULT_GRID,
    source_arm: str = "opencv",
    target_arm: str = "albumentations",
    repeats: int = 3,
    seed: int = 0,
) -> list[tuple[float, float, float]]:
    """Map `source_arm` severities onto the `target_arm` severities that match.

    This needs no photographs -- it compares two simulators to each other --
    and it settles a caveat already recorded in `TASKS.md`: at the same
    nominal severity the two arms are not equally severe (low_light most
    obviously), so severity-conditioned results cannot be pooled across arms
    without a translation. This is that translation, measured on a real image
    rather than assumed to be the identity.

    Returns:
        (source_severity, matched_target_severity, residual) per input
        severity. A matched severity pinned at the end of `grid` means the
        target arm cannot reach the source arm's severity at all.
    """
    out: list[tuple[float, float, float]] = []
    target_sweep = np.stack([_mean_stats(target_arm, degradation, reference, s, repeats, seed)
                             for s in grid])
    scale = target_sweep.std(axis=0)
    names = SENSITIVE_STATS[degradation]
    for s in severities:
        source_vec = _mean_stats(source_arm, degradation, reference, float(s), repeats, seed)
        dists = [stat_distance(source_vec, target_sweep[i], scale, names) for i in range(len(grid))]
        best = int(np.argmin(dists))
        out.append((float(s), float(grid[best]), float(dists[best])))
    return out
