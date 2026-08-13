"""Render a latent scene state onto a real image.

`src/data/degradation.py` already implements the five artifact primitives and,
importantly, the box-remapping contract for the one geometric degradation
(`angle`). This module does not reimplement any of that -- it only supplies the
severities from a `SceneState` instead of drawing them at random, and applies
them in the same canonical capture order.

The severity contract is unchanged: every value is in [0, 1], 0.0 is a no-op.
So a rendered capture is a drop-in for anything that already consumes a
`DegradationResult`.

Note on randomness: the primitives in `degradation.py` use the module-level
`random` / `np.random` state internally (kernel angles, hotspot positions).
`render_severities` seeds both from the caller's Generator so a session
replays identically, without changing `degradation.py`'s public contract.
"""

from __future__ import annotations

import random as _stdlib_random

import cv2
import numpy as np

from src.data.degradation import (
    DEGRADATIONS,
    DegradationResult,
    angle_with_matrix,
    transform_boxes,
)
from src.sim.state import SceneState

# Same order as apply_degradations(): geometry at capture, then light, then
# optics, then the encoder last.
RENDER_ORDER = ("angle", "low_light", "glare", "blur", "jpeg")


def render_severities(
    img: np.ndarray,
    severities: dict[str, float],
    boxes: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> DegradationResult:
    """Apply named degradations at *given* severities (no sampling).

    Args:
        img: BGR uint8 (H, W, 3).
        severities: {degradation_name: severity in [0, 1]}. Names must be keys
            of `src.data.degradation.DEGRADATIONS`. Zero/absent means skip.
        boxes: optional (N, 4) COCO [x, y, w, h] ground truth. Required if the
            output feeds detection -- `angle` moves image content, so boxes
            must ride through the same homography.
        rng: seeds the primitives' internal randomness for replayability.

    Returns:
        DegradationResult, exactly as apply_degradations() would produce.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected BGR uint8 (H, W, 3), got shape {img.shape}")
    unknown = set(severities) - set(DEGRADATIONS)
    if unknown:
        raise ValueError(f"unknown degradation names: {sorted(unknown)}")

    if rng is not None:
        seed = int(rng.integers(0, 2**31 - 1))
        _stdlib_random.seed(seed)
        np.random.seed(seed)

    out = img.copy()
    h, w = img.shape[:2]
    out_boxes = None if boxes is None else np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    applied: dict[str, float] = {}

    for name in RENDER_ORDER:
        sev = float(severities.get(name, 0.0))
        if sev <= 0.0:
            continue
        if name == "angle":
            out, matrix = angle_with_matrix(out, sev)
            if out_boxes is not None:
                out_boxes = transform_boxes(out_boxes, matrix, w, h)
        else:
            out = DEGRADATIONS[name](out, sev)
        applied[name] = round(sev, 4)

    return DegradationResult(image=out, severities=applied, boxes=out_boxes)


def glare_at(
    img: np.ndarray,
    severity: float,
    center: tuple[float, float],
    radius_frac: float = 0.34,
) -> np.ndarray:
    """Specular highlight at a *chosen* position, rather than a random one.

    `src.data.degradation.glare` samples its hotspot position internally, which
    is right for building an augmented training set -- you want the highlight
    everywhere across the corpus. It is wrong whenever the position is the
    thing under study.

    `SceneState` carries a `glare_azimuth` and folds it into a single
    `effective_glare()` scalar before rendering, so the shipped renderer dims
    the wash as the hotspot "moves away" without ever moving anything in the
    pixels. That is a fair model of how much signal is lost and a poor model of
    what the photograph looks like. This function closes that gap for the cases
    that need it: same brightness, different place.

    Args:
        center: hotspot centre in normalised (x, y) coordinates, where (0.5,
            0.5) is the middle of the frame. Values outside [0, 1] are allowed
            and put the hotspot off-frame, leaving only its falloff.
        radius_frac: hotspot radius as a fraction of the shorter image side.
    """
    if severity <= 0:
        return img
    h, w = img.shape[:2]
    overlay = np.zeros((h, w), np.float32)
    cx, cy = int(round(center[0] * w)), int(round(center[1] * h))
    radius = max(1, int(min(h, w) * radius_frac))
    cv2.circle(overlay, (cx, cy), radius, 1.0, -1)
    overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=radius * 0.5)
    peak = overlay.max()
    if peak > 0:
        overlay /= peak
    out = img.astype(np.float32) + overlay[..., None] * (120.0 * float(severity))
    return np.clip(out, 0, 255).astype(np.uint8)


def render_capture(
    img: np.ndarray,
    state: SceneState,
    boxes: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    positioned_glare: bool = False,
) -> DegradationResult:
    """Render one shot of `img` under scene `state`.

    By default glare is passed through `SceneState.effective_glare()`, so a
    bright hotspot parked away from the tooth of interest costs little -- see
    `src.sim.state`.

    With `positioned_glare=True` the highlight is instead drawn at the position
    the azimuth implies, at full intensity, via `glare_at`. The two agree about
    how much diagnostic signal survives and disagree about the picture; use the
    positioned version when the picture is what matters. It is off by default so
    that every number already computed against the scalar path stays comparable.
    """
    severities = dict(state.severities())
    if not positioned_glare:
        return render_severities(img, severities, boxes=boxes, rng=rng)

    intensity = float(state.factors["glare"])
    severities.pop("glare", None)
    out = render_severities(img, severities, boxes=boxes, rng=rng)
    if intensity > 0:
        # azimuth 0 puts the hotspot on the tooth; 0.5 is diametrically away
        # The travel rate is tied to `effective_glare`'s falloff (sigma 0.22):
        # at the azimuth where occlusion has halved, the hotspot should be
        # leaving the frame, not already long gone. Too fast a rate makes the
        # picture disagree with the number printed beside it.
        d = min(state.glare_azimuth, 1.0 - state.glare_azimuth)
        offset = 1.8 * d
        out.image = glare_at(out.image, intensity, (0.5 + offset, 0.5 - 0.45 * offset))
        out.severities["glare"] = round(state.effective_glare(), 4)
    return out
