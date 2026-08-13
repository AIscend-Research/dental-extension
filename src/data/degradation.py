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

Bounding boxes: `angle` is the only degradation that moves image content, so
it is the only one that invalidates ground-truth boxes. Detector training
must therefore go through `apply_degradations(..., boxes=...)`, which remaps
them via `transform_boxes()`; degrading pixels without remapping boxes trains
the model against silently misaligned labels. The confidence head doesn't use
boxes, so it can ignore all of this.
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


def angle_with_matrix(img: np.ndarray, severity: float) -> tuple[np.ndarray, np.ndarray]:
    """`angle()`, but also returns the 3x3 homography it applied.

    Needed for detector training: `angle` is the one degradation that MOVES
    image content, so ground-truth boxes have to move with it. Anything that
    warps pixels without remapping the boxes trains the detector against
    silently misaligned labels. See transform_boxes().
    """
    if severity <= 0:
        return img, np.eye(3, dtype=np.float64)
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
    warped = cv2.warpPerspective(rotated, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    # compose: rotation (2x3 -> 3x3) first, then perspective
    rot3 = np.vstack([rot, [0.0, 0.0, 1.0]])
    return warped, m.astype(np.float64) @ rot3


def angle(img: np.ndarray, severity: float) -> np.ndarray:
    """In-plane rotation + perspective warp, simulating an off-axis phone shot.

    Pads with edge replication so you do not introduce black borders that the
    model could cheat on.

    NOTE: this moves image content. If the image has boxes attached, use
    apply_degradations(..., boxes=...) rather than calling this directly --
    that path remaps the boxes through the same homography.
    """
    return angle_with_matrix(img, severity)[0]


def transform_boxes(
    boxes: np.ndarray, matrix: np.ndarray, width: int, height: int
) -> np.ndarray:
    """Remap COCO [x, y, w, h] boxes through a 3x3 homography.

    Each box's four corners are projected and re-tightened into an
    axis-aligned box (the usual detection convention -- a rotated rectangle
    has no axis-aligned equivalent, so the result is the enclosing box and is
    slightly looser than the original). Results are clipped to the image.

    Args:
        boxes: (N, 4) array of [x, y, w, h] in pixels. An empty array is fine.
        matrix: 3x3 homography, e.g. from angle_with_matrix().
        width / height: image size, for clipping.

    Returns:
        (N, 4) array of [x, y, w, h]. Boxes warped entirely out of frame come
        back with zero width/height -- filter them with
        `keep = (out[:, 2] > 0) & (out[:, 3] > 0)` before use.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if len(boxes) == 0:
        return boxes.reshape(0, 4)

    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    corners = np.stack([
        np.stack([x, y], axis=-1),
        np.stack([x + w, y], axis=-1),
        np.stack([x + w, y + h], axis=-1),
        np.stack([x, y + h], axis=-1),
    ], axis=1)  # (N, 4, 2)

    warped = cv2.perspectiveTransform(corners.astype(np.float64), matrix)  # (N, 4, 2)
    x1 = np.clip(warped[:, :, 0].min(axis=1), 0, width)
    y1 = np.clip(warped[:, :, 1].min(axis=1), 0, height)
    x2 = np.clip(warped[:, :, 0].max(axis=1), 0, width)
    y2 = np.clip(warped[:, :, 1].max(axis=1), 0, height)
    return np.stack([x1, y1, np.maximum(x2 - x1, 0.0), np.maximum(y2 - y1, 0.0)], axis=-1)


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
    # (N, 4) COCO [x, y, w, h] boxes remapped through any geometric
    # degradation, or None if no boxes were passed in. Only "angle" moves
    # them; every other degradation leaves them untouched by construction.
    boxes: np.ndarray | None = None

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
    boxes: np.ndarray | None = None,
) -> DegradationResult:
    """Apply a random subset of degradations at random severities.

    Args:
        img: BGR uint8 image.
        which: restrict to these degradation names; None means sample freely.
        severity_range: (lo, hi) each chosen degradation draws from.
        max_simultaneous: cap on how many stack on one image.
        seed: set for reproducibility in tests / dataset generation.
        boxes: optional (N, 4) COCO [x, y, w, h] ground-truth boxes. Pass
            these whenever the output feeds detector training -- "angle"
            warps image content, so boxes must be remapped through the same
            homography or the detector trains against misaligned labels. The
            remapped boxes come back on DegradationResult.boxes.

    Returns:
        DegradationResult with the degraded image, the per-type severities,
        and (if boxes were given) the remapped boxes.
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
    out_boxes = None if boxes is None else np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    h, w = img.shape[:2]
    applied: dict[str, float] = {}
    # order matters a little: geometry -> lighting -> optics -> codec, which
    # roughly mirrors the real capture pipeline (angle at capture, then light,
    # then lens blur, then the phone's JPEG encoder last).
    for name in ["angle", "low_light", "glare", "blur", "jpeg"]:
        if name not in chosen:
            continue
        sev = random.uniform(*severity_range)
        if name == "angle":
            # the only geometric degradation -- take the homography back out
            # so the boxes can ride along (see transform_boxes)
            out, matrix = angle_with_matrix(out, sev)
            if out_boxes is not None:
                out_boxes = transform_boxes(out_boxes, matrix, w, h)
        else:
            out = DEGRADATIONS[name](out, sev)
        applied[name] = round(sev, 3)
    return DegradationResult(image=out, severities=applied, boxes=out_boxes)


def make_burst(
    img: np.ndarray,
    n_shots: int = 3,
    severity_range: tuple[float, float] = (0.2, 0.6),
    seed: int | None = None,
    boxes: np.ndarray | None = None,
) -> list[DegradationResult]:
    """Generate 2-3 quick 'shots' of the same film for the fusion approach.

    Each shot gets independent, mostly-mild glare/angle/noise, so no single
    frame is clean but the cross-frame consistency carries the signal. This is
    what feeds the fusion module (Phase 3) and the cross-photo confidence
    signal (Phase 4).

    Pass `boxes` if the burst feeds detection training/eval: each shot's angle
    jitter is independent, so each shot gets its OWN remapped boxes on
    DegradationResult.boxes -- they are not interchangeable across frames.
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
            boxes=boxes,
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
