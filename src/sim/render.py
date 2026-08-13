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


def render_capture(
    img: np.ndarray,
    state: SceneState,
    boxes: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> DegradationResult:
    """Render one shot of `img` under scene `state`.

    Glare is passed through `SceneState.effective_glare()`, so a bright
    hotspot parked away from the tooth of interest costs little -- see
    `src.sim.state`.
    """
    return render_severities(img, state.severities(), boxes=boxes, rng=rng)
