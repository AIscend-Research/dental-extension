"""
Quick visual sanity check for the degradation pipeline.

Run it with no data and it makes a synthetic gradient; point it at a real image
and it degrades that. Saves a grid of before / each-degradation / composite so
you can eyeball whether the synthetic artifacts look like real phone shots --
that judgement call is a Phase 2 deliverable.

    python demo_degradation.py
    python demo_degradation.py --image path/to/xray.png
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from src.data.degradation import DEGRADATIONS, apply_degradations, make_burst


def load_or_synth(path: str | None) -> np.ndarray:
    if path:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        return img
    # synthetic stand-in: gradient + some circles so blur/glare are visible
    size = 512
    grad = np.tile(np.linspace(0, 255, size, dtype=np.uint8), (size, 1))
    img = cv2.cvtColor(grad, cv2.COLOR_GRAY2BGR)
    for _ in range(12):
        c = (np.random.randint(0, size), np.random.randint(0, size))
        cv2.circle(img, c, np.random.randint(10, 40), (60, 60, 60), -1)
    return img


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None)
    ap.add_argument("--out", default="degradation_demo.png")
    ap.add_argument("--severity", type=float, default=0.7)
    args = ap.parse_args()

    np.random.seed(0)
    base = load_or_synth(args.image)
    # Preserve aspect ratio -- real DENTEX panoramics are ~2:1 wide, and
    # squashing them to a square before judging "does this look like a real
    # phone shot" (the Phase 2 deliverable this script is for) defeats the
    # purpose.
    h, w = base.shape[:2]
    target_w = 384
    target_h = max(1, round(target_w * h / w))
    base = cv2.resize(base, (target_w, target_h))

    tiles = [label(base, "original")]
    for name, fn in DEGRADATIONS.items():
        tiles.append(label(fn(base, args.severity), f"{name} @ {args.severity}"))

    composite = apply_degradations(base, seed=3)
    tiles.append(label(composite.image,
                       "composite: " + ",".join(composite.severities)))

    # burst row
    for i, shot in enumerate(make_burst(base, seed=5)):
        tiles.append(label(shot.image, f"burst {i}"))

    # pad to a rectangle and tile into a grid
    cols = 4
    while len(tiles) % cols:
        tiles.append(np.zeros_like(base))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    grid = np.vstack(rows)

    cv2.imwrite(args.out, grid)
    print(f"wrote {args.out}  ({grid.shape[1]}x{grid.shape[0]})")
    print("composite severities:", composite.severities)


if __name__ == "__main__":
    main()
