"""
DENTEX dataset access.

DENTEX ships COCO-format annotations for panoramic dental X-rays. Only the
fully-labelled third split (quadrant-enumeration-diagnosis) carries the
diagnosis classes we care about: caries, deep_caries, periapical_lesion,
impacted. For a caries-only detector you can either keep all four diagnosis
classes or collapse to {caries, deep_caries} vs background -- decide this
early in Phase 2, it changes the label maps everywhere downstream.

This module deliberately does NOT depend on detectron2. It reads the COCO json
into plain dicts so the degradation pipeline, the split logic, and quick
sanity checks all work before the detector stack is installed. When you wire
up training in Phase 3, register these into detectron2's DatasetCatalog from
here.

Get the data first:  python scripts/download_dentex.py
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

# The four DENTEX diagnosis classes, in a fixed order. Keep this canonical.
DIAGNOSIS_CLASSES = ["caries", "deep_caries", "periapical_lesion", "impacted"]

# If you go caries-only, this is the collapse map. Everything not listed -> bg.
CARIES_ONLY_MAP = {"caries": 0, "deep_caries": 0}


def load_coco(json_path: str | Path) -> dict:
    """Load a COCO-format annotation file into memory."""
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(
            f"{json_path} not found. Run scripts/download_dentex.py first."
        )
    with open(json_path) as f:
        return json.load(f)


def index_annotations(coco: dict) -> dict[int, list[dict]]:
    """Group annotations by image_id for O(1) lookup."""
    by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        by_image[ann["image_id"]].append(ann)
    return by_image


def _patient_key(file_name: str) -> str:
    """Best-effort patient id from a filename.

    DENTEX filenames do not always expose a clean patient id, so this is a
    heuristic: strip the extension and any trailing _<number> that looks like a
    per-shot index. TODO(phase2): confirm against the real metadata once the
    data is downloaded -- if there is a proper patient field, use that instead.
    A wrong key here leaks patients across train/val, which quietly inflates
    every number in the paper, so this one is worth getting right.
    """
    stem = Path(file_name).stem
    return re.sub(r"_\d+$", "", stem)


def patient_level_split(
    coco: dict,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
) -> dict[str, list[int]]:
    """Split image_ids into train/val/test with no patient in two splits.

    Returns a dict {"train": [...], "val": [...], "test": [...]} of image_ids.
    Splitting by patient (not by image) is the whole point -- see _patient_key.
    """
    import random

    rng = random.Random(seed)
    images = coco.get("images", [])
    # bucket image_ids by patient
    patients: dict[str, list[int]] = defaultdict(list)
    for img in images:
        patients[_patient_key(img["file_name"])].append(img["id"])

    keys = list(patients.keys())
    rng.shuffle(keys)
    n = len(keys)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test_keys = keys[:n_test]
    val_keys = keys[n_test:n_test + n_val]
    train_keys = keys[n_test + n_val:]

    def collect(ks: list[str]) -> list[int]:
        out: list[int] = []
        for k in ks:
            out.extend(patients[k])
        return out

    return {
        "train": collect(train_keys),
        "val": collect(val_keys),
        "test": collect(test_keys),
    }


def class_balance(coco: dict) -> dict[str, int]:
    """Count annotations per category name. Use this to check imbalance early.

    DENTEX is imbalanced (impacted and caries dominate; periapical is rare), so
    look at this before training and decide on RepeatFactorSampler weights or a
    class-balanced loss. HierarchicalDet already turns on USE_FED_LOSS for this
    reason.
    """
    id_to_name = {c["id"]: c["name"] for c in coco.get("categories", [])}
    counts: dict[str, int] = defaultdict(int)
    for ann in coco.get("annotations", []):
        counts[id_to_name.get(ann["category_id"], str(ann["category_id"]))] += 1
    return dict(counts)


# TODO(phase3): register_dentex_detectron2(split_ids, image_root) ->
# call DatasetCatalog.register / MetadataCatalog.set here so train_net.py can
# consume "custom_train_class" / "custom_validation_class" the configs expect.


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python -m src.data.dentex <path-to-coco.json>")
        raise SystemExit(1)
    coco = load_coco(sys.argv[1])
    print("images:", len(coco.get("images", [])))
    print("annotations:", len(coco.get("annotations", [])))
    print("class balance:", class_balance(coco))
    split = patient_level_split(coco)
    print({k: len(v) for k, v in split.items()})
