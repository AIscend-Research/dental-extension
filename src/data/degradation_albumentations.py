"""
Albumentations-preset degradation pipeline -- the Phase 2 ablation arm.

TASKS.md's Stream 1 calls for comparing degradation strategies: "the current
hand-built OpenCV transforms vs albumentations presets vs (if you have time)
re-photographing a few printed X-rays with a phone." This module is the
albumentations arm -- same five degradation names and the same [0, 1]
monotonic severity contract as src/data/degradation.py, so the two can be
swapped in for a side-by-side comparison (see demo_degradation_compare.py).

Reuses DegradationResult from src/data/degradation.py rather than duplicating
it, so both arms produce the same label-vector shape for the confidence head.

Note on reproducibility: albumentations transforms do NOT respect Python's or
numpy's global random.seed()/np.random.seed() (confirmed empirically) -- each
transform needs to go through an A.Compose(..., seed=...) to be reproducible.
"""

from __future__ import annotations

import random
from typing import Callable

import albumentations as A
import cv2
import numpy as np

from src.data.degradation import DEGRADATION_NAMES, DegradationResult


def _run(transform: A.BasicTransform, img: np.ndarray, seed: int | None) -> np.ndarray:
    """Apply a single transform through a seeded Compose for reproducibility."""
    return A.Compose([transform], seed=seed)(image=img)["image"]


def blur(img: np.ndarray, severity: float, seed: int | None = None) -> np.ndarray:
    if severity <= 0:
        return img
    k = int(round(3 + severity * 22))
    k = k + 1 if k % 2 == 0 else k
    sigma = max(0.1, severity * 6)
    t = A.GaussianBlur(blur_limit=(k, k), sigma_limit=(sigma, sigma), p=1.0)
    return _run(t, img, seed)


def glare(img: np.ndarray, severity: float, seed: int | None = None) -> np.ndarray:
    if severity <= 0:
        return img
    h, w = img.shape[:2]
    radius = max(1, int(min(h, w) * (0.15 + 0.35 * severity)))
    t = A.RandomSunFlare(
        flare_roi=(0, 0, 1, 1),
        src_radius=radius,
        num_flare_circles_range=(1, 2),
        p=1.0,
    )
    return _run(t, img, seed)


def _angle_transform(severity: float) -> A.BasicTransform:
    rot = severity * 20
    shift = severity * 0.12
    return A.Affine(
        rotate=(-rot, rot),
        scale=(1 - shift, 1 + shift),
        shear=(-shift * 20, shift * 20),
        border_mode=cv2.BORDER_REPLICATE,
        p=1.0,
    )


def angle(img: np.ndarray, severity: float, seed: int | None = None) -> np.ndarray:
    if severity <= 0:
        return img
    return _run(_angle_transform(severity), img, seed)


def angle_with_boxes(
    img: np.ndarray, severity: float, boxes: np.ndarray, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`angle()` that also remaps COCO [x, y, w, h] boxes.

    Mirrors src/data/degradation.py's box contract for this arm: Affine is the
    only geometric transform of the five, so it is the only one that
    invalidates ground truth. Albumentations does the remapping itself via
    bbox_params, which is why this needs its own Compose rather than reusing
    _run().

    Returns:
        (image, boxes, kept) -- `kept` is the indices of the input boxes that
        survived. Albumentations DROPS boxes warped out of frame rather than
        zeroing them, so the box array can come back shorter than it went in;
        use `kept` to index any parallel per-box class labels, or they will
        silently desynchronize from the boxes.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if severity <= 0:
        return img, boxes, np.arange(len(boxes))
    if len(boxes) == 0:
        return _run(_angle_transform(severity), img, seed), boxes, np.arange(0)

    composed = A.Compose(
        [_angle_transform(severity)],
        bbox_params=A.BboxParams(format="coco", label_fields=["box_index"], clip=True),
        seed=seed,
    )
    result = composed(
        image=img,
        bboxes=boxes.tolist(),
        box_index=list(range(len(boxes))),
    )
    out_boxes = np.asarray(result["bboxes"], dtype=np.float64).reshape(-1, 4)
    return result["image"], out_boxes, np.asarray(result["box_index"], dtype=int)


def low_light(img: np.ndarray, severity: float, seed: int | None = None) -> np.ndarray:
    if severity <= 0:
        return img
    brightness = -0.8 * severity
    t1 = A.RandomBrightnessContrast(
        brightness_limit=(brightness, brightness), contrast_limit=0, p=1.0
    )
    out = _run(t1, img, seed)
    std = max(1e-4, severity * 0.06)
    t2 = A.GaussNoise(std_range=(std, std), p=1.0)
    # derive the second seed instead of using seed+1: apply_degradations hands
    # out per-step seeds as seed+step, so seed+1 here would collide with the
    # next degradation's seed and correlate two supposedly independent draws.
    return _run(t2, out, None if seed is None else seed + 1013)


def jpeg(img: np.ndarray, severity: float, seed: int | None = None) -> np.ndarray:
    if severity <= 0:
        return img
    quality = max(5, int(round(90 - severity * 80)))
    t = A.ImageCompression(compression_type="jpeg", quality_range=(quality, quality), p=1.0)
    return _run(t, img, seed)


DEGRADATIONS: dict[str, Callable[..., np.ndarray]] = {
    "blur": blur,
    "glare": glare,
    "angle": angle,
    "low_light": low_light,
    "jpeg": jpeg,
}


def apply_degradations(
    img: np.ndarray,
    which: list[str] | None = None,
    severity_range: tuple[float, float] = (0.3, 0.9),
    max_simultaneous: int = 3,
    seed: int | None = None,
    boxes: np.ndarray | None = None,
) -> DegradationResult:
    """Same interface/contract as src.data.degradation.apply_degradations.

    Args mirror the hand-built pipeline exactly so the two are drop-in
    alternatives for the same call site -- including `boxes`, which is
    remapped through the Affine (the only geometric transform here) so this
    arm is equally safe for detector training. See angle_with_boxes() for the
    one behavioral difference: albumentations drops out-of-frame boxes rather
    than zeroing them, so DegradationResult.boxes can be shorter than the
    input.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(
            f"expected a 3-channel BGR uint8 image (H, W, 3), got shape {img.shape}"
        )
    if seed is not None:
        random.seed(seed)

    pool = which if which is not None else DEGRADATION_NAMES
    n = random.randint(1, min(max_simultaneous, len(pool)))
    chosen = random.sample(pool, n)

    out = img.copy()
    out_boxes = None if boxes is None else np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    # angle is the only step that can drop a box (Affine crops out-of-frame
    # boxes instead of zeroing them); it runs at most once and always first,
    # so `kept_indices` maps 1:1 into the ORIGINAL `boxes` the whole time --
    # no later step touches it. Left unset (None) if angle never runs --
    # DegradationResult.__post_init__ fills in the identity default from
    # out_boxes itself in that case, same as the opencv arm.
    kept_indices = None
    applied: dict[str, float] = {}
    step = 0
    for name in ["angle", "low_light", "glare", "blur", "jpeg"]:
        if name not in chosen:
            continue
        sev = random.uniform(*severity_range)
        step_seed = None if seed is None else seed + step
        if name == "angle" and out_boxes is not None:
            out, out_boxes, kept_indices = angle_with_boxes(out, sev, out_boxes, seed=step_seed)
        else:
            out = DEGRADATIONS[name](out, sev, seed=step_seed)
        applied[name] = round(sev, 3)
        step += 1
    return DegradationResult(image=out, severities=applied, boxes=out_boxes, kept_indices=kept_indices)


if __name__ == "__main__":
    demo = np.tile(np.linspace(0, 255, 512, dtype=np.uint8), (512, 1))
    demo = cv2.cvtColor(demo, cv2.COLOR_GRAY2BGR)
    r = apply_degradations(demo, seed=0)
    print("applied:", r.severities)
    print("label vector:", r.label_vector())
