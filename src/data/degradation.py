"""
Synthetic smartphone-artifact degradation pipeline.

This is the core of Phase 2. It takes a clean dental image (a photographed
lightbox film, a printed X-ray, or a raw DENTEX panoramic) and applies the
kinds of artifacts you get when someone snaps it with a phone in a rural
clinic: motion/defocus blur, specular glare off the film or lightbox, a bad
shooting angle, poor lighting, and JPEG compression from messaging apps.

Every degraded image comes back with a label dict recording which artifacts
were applied and at what severity. Those labels are the weak-supervision
signal the confidence head trains on in Phase 4, so keep them accurate if you
add new degradations.

Dependencies: numpy + opencv only. No detectron2, no torch. That is deliberate
-- you can build and test this the moment you clone, before anyone has the
detector stack running.

Severity convention: every degradation takes `severity` in [0.0, 1.0].
0.0 is a no-op, 1.0 is "barely usable". Keep that contract; the eval code in
Phase 4 sweeps severity and assumes it is monotonic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Individual degradations. Each takes (img_bgr_uint8, severity) -> img_bgr_uint8
# ---------------------------------------------------------------------------


def blur(img: np.ndarray, severity: float) -> np.ndarray:
    """Defocus + slight motion blur. Kernel grows with severity."""
    if severity <= 0:
        return img
    # odd kernel size from 3 up to ~25
    k = int(round(3 + severity * 22))
    k = k + 1 if k % 2 == 0 else k
    defocus = cv2.GaussianBlur(img, (k, k), sigmaX=severity * 6)
    # add a touch of directional motion blur on top
    m = max(3, int(round(severity * 15)))
    kernel = np.zeros((m, m), np.float32)
    kernel[m // 2, :] = 1.0 / m
    angle = random.uniform(0, 180)
    rot = cv2.getRotationMatrix2D((m / 2, m / 2), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (m, m))
    s = kernel.sum()
    if s > 0:
        kernel /= s
    return cv2.filter2D(defocus, -1, kernel)


def glare(img: np.ndarray, severity: float) -> np.ndarray:
    """Specular highlight -- a bright soft blob, like lightbox reflection."""
    if severity <= 0:
        return img
    h, w = img.shape[:2]
    overlay = np.zeros((h, w), np.float32)
    # one or two hotspots
    for _ in range(random.randint(1, 2)):
        cx, cy = random.randint(0, w), random.randint(0, h)
        radius = int(min(h, w) * (0.15 + 0.35 * severity))
        cv2.circle(overlay, (cx, cy), radius, 1.0, -1)
    overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=radius * 0.5)
    overlay = overlay / (overlay.max() + 1e-6)
    strength = 120 * severity  # additive brightness in 0-255 space
    out = img.astype(np.float32) + overlay[..., None] * strength
    return np.clip(out, 0, 255).astype(np.uint8)


def angle(img: np.ndarray, severity: float) -> np.ndarray:
    """In-plane rotation + perspective warp, simulating an off-axis phone shot.

    Pads with edge replication so you do not introduce black borders that the
    model could cheat on.
    """
    if severity <= 0:
        return img
    h, w = img.shape[:2]
    rot_deg = random.uniform(-1, 1) * severity * 20  # up to +/-20 deg
    rot = cv2.getRotationMatrix2D((w / 2, h / 2), rot_deg, 1.0)
    rotated = cv2.warpAffine(img, rot, (w, h), borderMode=cv2.BORDER_REPLICATE)
    # perspective: nudge the four corners inward by a severity-scaled amount
    shift = severity * 0.12
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [w * random.uniform(0, shift), h * random.uniform(0, shift)],
        [w * (1 - random.uniform(0, shift)), h * random.uniform(0, shift)],
        [w * (1 - random.uniform(0, shift)), h * (1 - random.uniform(0, shift))],
        [w * random.uniform(0, shift), h * (1 - random.uniform(0, shift))],
    ])
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(rotated, m, (w, h), borderMode=cv2.BORDER_REPLICATE)


def low_light(img: np.ndarray, severity: float) -> np.ndarray:
    """Underexposure via gamma darkening plus Poisson-ish sensor noise."""
    if severity <= 0:
        return img
    gamma = 1.0 + severity * 2.5  # >1 darkens
    norm = img.astype(np.float32) / 255.0
    darkened = np.power(norm, gamma)
    # shot noise scales up as signal drops
    noise_sigma = severity * 0.06
    noise = np.random.normal(0, noise_sigma, img.shape).astype(np.float32)
    out = np.clip(darkened + noise, 0, 1) * 255.0
    return out.astype(np.uint8)


def jpeg(img: np.ndarray, severity: float) -> np.ndarray:
    """Re-encode at low JPEG quality -- the WhatsApp-forwarding artifact."""
    if severity <= 0:
        return img
    quality = int(round(90 - severity * 80))  # 90 down to ~10
    quality = max(5, quality)
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return img
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


# Registry so the composite applier and the confidence head share one name list.
DEGRADATIONS: dict[str, Callable[[np.ndarray, float], np.ndarray]] = {
    "blur": blur,
    "glare": glare,
    "angle": angle,
    "low_light": low_light,
    "jpeg": jpeg,
}

DEGRADATION_NAMES = list(DEGRADATIONS.keys())


# ---------------------------------------------------------------------------
# Composite applier
# ---------------------------------------------------------------------------


@dataclass
class DegradationResult:
    image: np.ndarray
    # label vector aligned to DEGRADATION_NAMES; 0.0 means "not applied"
    severities: dict[str, float] = field(default_factory=dict)

    def label_vector(self) -> np.ndarray:
        """Severity per degradation, in DEGRADATION_NAMES order. For the head."""
        return np.array([self.severities.get(n, 0.0) for n in DEGRADATION_NAMES],
                        dtype=np.float32)

    def any_severe(self, threshold: float = 0.6) -> bool:
        return any(v >= threshold for v in self.severities.values())


def apply_degradations(
    img: np.ndarray,
    which: list[str] | None = None,
    severity_range: tuple[float, float] = (0.3, 0.9),
    max_simultaneous: int = 3,
    seed: int | None = None,
) -> DegradationResult:
    """Apply a random subset of degradations at random severities.

    Args:
        img: BGR uint8 image.
        which: restrict to these degradation names; None means sample freely.
        severity_range: (lo, hi) each chosen degradation draws from.
        max_simultaneous: cap on how many stack on one image.
        seed: set for reproducibility in tests / dataset generation.

    Returns:
        DegradationResult with the degraded image and the per-type severities.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        # glare() in particular silently corrupts a 2D (H, W) grayscale input
        # into (H, W, H) via a numpy broadcasting accident ((H,W) + (H,W,1)
        # broadcasts on the wrong axis) instead of erroring -- fail loudly
        # here rather than let that propagate into a dataset unnoticed.
        raise ValueError(
            f"expected a 3-channel BGR uint8 image (H, W, 3), got shape {img.shape}"
        )
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    pool = which if which is not None else DEGRADATION_NAMES
    n = random.randint(1, min(max_simultaneous, len(pool)))
    chosen = random.sample(pool, n)

    out = img.copy()
    applied: dict[str, float] = {}
    # order matters a little: geometry -> lighting -> optics -> codec, which
    # roughly mirrors the real capture pipeline (angle at capture, then light,
    # then lens blur, then the phone's JPEG encoder last).
    for name in ["angle", "low_light", "glare", "blur", "jpeg"]:
        if name in chosen:
            sev = random.uniform(*severity_range)
            out = DEGRADATIONS[name](out, sev)
            applied[name] = round(sev, 3)
    return DegradationResult(image=out, severities=applied)


def make_burst(
    img: np.ndarray,
    n_shots: int = 3,
    severity_range: tuple[float, float] = (0.2, 0.6),
    seed: int | None = None,
) -> list[DegradationResult]:
    """Generate 2-3 quick 'shots' of the same film for the fusion approach.

    Each shot gets independent, mostly-mild glare/angle/noise, so no single
    frame is clean but the cross-frame consistency carries the signal. This is
    what feeds the fusion module (Phase 3) and the cross-photo confidence
    signal (Phase 4).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    shots = []
    for i in range(n_shots):
        # bursts are dominated by glare/angle jitter, not heavy blur/jpeg
        res = apply_degradations(
            img,
            which=["glare", "angle", "low_light"],
            severity_range=severity_range,
            max_simultaneous=2,
            seed=None,
        )
        shots.append(res)
    return shots


if __name__ == "__main__":
    # tiny smoke test on a synthetic gradient so you can eyeball output
    demo = np.tile(np.linspace(0, 255, 512, dtype=np.uint8), (512, 1))
    demo = cv2.cvtColor(demo, cv2.COLOR_GRAY2BGR)
    r = apply_degradations(demo, seed=0)
    print("applied:", r.severities)
    print("label vector:", r.label_vector())
    print("burst shots:", [s.severities for s in make_burst(demo, seed=1)])
