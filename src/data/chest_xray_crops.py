"""Chest X-ray pneumonia images -- the second-modality generalization check.

Real CheXphoto data (natural photos + synthetic transforms of CheXpert
x-rays) is gated behind a CheXpert data-use agreement, not obtainable in
this environment. This module loads a substitute that IS freely available:
the Kermany et al. pediatric pneumonia chest X-ray dataset
(CC BY 4.0, https://data.mendeley.com/datasets/rscbjbr9sj/2), mirrored on
Hugging Face as `hf-vision/chest-xray-pneumonia`, no data-use agreement
required.

This is a genuinely different real medical-imaging classification task
(pneumonia vs. normal on pediatric AP chest radiographs) from dental caries
detection, run through this project's OWN capture simulator
(`src/data/degradation.py`) and evidence machinery -- not a reproduction of
CheXphoto's specific photographic-corruption code or its natural-photo
capture rig, neither of which were obtained. See
`docs/experiments_results.md`'s E13 section for exactly what this can and
cannot claim; the honest label throughout is "second real modality, same
framework" not "CheXphoto reproduction".

Unlike DENTEX, each row here is one independent patient image with a single
label -- no multi-tooth-per-radiograph structure, so no group-aware split is
needed; a plain random split by image is the correct protocol here, not a
limitation relative to DENTEX's grouped split.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

DEFAULT_ROOT = Path("data/chest_xray_pneumonia/data")


@dataclass
class ChestCrop:
    """One chest radiograph, resized to a fixed size.

    Attributes:
        image: BGR uint8.
        label: 1 = pneumonia, 0 = normal.
        source_file: parquet shard + row index, for auditing only.
    """

    image: np.ndarray
    label: int
    source_file: str


def _load_parquet(path: Path, crop_size: int) -> list[ChestCrop]:
    table = pq.read_table(path)
    crops: list[ChestCrop] = []
    for i, row in enumerate(table.to_pylist()):
        buf = np.frombuffer(row["image"]["bytes"], dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.resize(img, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
        crops.append(ChestCrop(image=img, label=int(row["label"]), source_file=f"{path.name}:{i}"))
    return crops


def load_chest_crops(root: Path | str = DEFAULT_ROOT, crop_size: int = 96) -> dict[str, list[ChestCrop]]:
    """Load whatever train/test parquet shards are present under `root`.

    Only the shards actually downloaded (see docs/experiments_results.md
    E13 for which ones) are read -- this does not require the full ~1.2GB
    dataset, just whichever shards were fetched.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"{root} not found. Fetch a few parquet shards of "
            "hf-vision/chest-xray-pneumonia first, e.g.:\n"
            "  from huggingface_hub import snapshot_download\n"
            "  snapshot_download('hf-vision/chest-xray-pneumonia', repo_type='dataset',\n"
            "      local_dir='data/chest_xray_pneumonia',\n"
            "      allow_patterns=['data/test-*.parquet', 'data/train-00002-*.parquet', "
            "'data/train-00003-*.parquet'])"
        )
    train_files = sorted(root.glob("train-*.parquet"))
    test_files = sorted(root.glob("test-*.parquet"))
    train = [c for f in train_files for c in _load_parquet(f, crop_size)]
    test_all = [c for f in test_files for c in _load_parquet(f, crop_size)]
    return {"train": train, "test_all": test_all}


def split_test(
    test_all: list[ChestCrop], fraction: float = 0.5, seed: int = 0
) -> tuple[list[ChestCrop], list[ChestCrop]]:
    """Split the test pool into (calibration, held-out) by image, plain random."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(test_all))
    n_cal = int(round(fraction * len(idx)))
    cal = [test_all[i] for i in idx[:n_cal]]
    held = [test_all[i] for i in idx[n_cal:]]
    return cal, held


def describe(crops: list[ChestCrop], name: str = "") -> str:
    n = len(crops)
    pos = sum(c.label for c in crops)
    if not n:
        return f"{name:<12} empty"
    return f"{name:<12} {n:>5} images | pneumonia {pos:>5} ({pos / n:.2f}) | normal {n - pos:>5}"
