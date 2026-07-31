"""
Side-by-side comparison of the two degradation strategies (Phase 2 ablation).

TASKS.md's Stream 1 ablation: "the current hand-built OpenCV transforms vs
albumentations presets vs (if you have time) re-photographing a few printed
X-rays with a phone." This covers the first two -- the third needs an actual
phone and printed film, which needs a person, not code.

    python demo_degradation_compare.py --image path/to/xray.png
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

import src.data.degradation as opencv_pipeline
import src.data.degradation_albumentations as albu_pipeline


def load_or_synth(path: str | None) -> np.ndarray:
    if path:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        return img
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
    ap.add_argument("--out", default="degradation_compare.png")
    ap.add_argument("--severity", type=float, default=0.7)
    args = ap.parse_args()

    np.random.seed(0)
    base = load_or_synth(args.image)
    h, w = base.shape[:2]
    target_w = 384
    target_h = max(1, round(target_w * h / w))
    base = cv2.resize(base, (target_w, target_h))

    rows = [[label(base, "original")]]
    for name in opencv_pipeline.DEGRADATION_NAMES:
        opencv_out = opencv_pipeline.DEGRADATIONS[name](base.copy(), args.severity)
        albu_out = albu_pipeline.DEGRADATIONS[name](base.copy(), args.severity, seed=0)
        rows.append([
            label(opencv_out, f"opencv: {name} @ {args.severity}"),
            label(albu_out, f"albumentations: {name} @ {args.severity}"),
        ])

    cols = 2
    tiles = [t for row in rows for t in row]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(base))
    grid_rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    grid = np.vstack(grid_rows)

    cv2.imwrite(args.out, grid)
    print(f"wrote {args.out}  ({grid.shape[1]}x{grid.shape[0]})")


if __name__ == "__main__":
    main()
