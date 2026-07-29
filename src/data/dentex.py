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

    CONFIRMED against the real download (Phase 2): DENTEX ships no patient
    identifier anywhere -- not in the filename, not as a json field. Filenames
    are just a per-split sequential index: train_673.png, val_15.png,
    test_66.png. The original heuristic here stripped a trailing "_<number>"
    assuming it was a multi-shot index (e.g. "patientA_1.png", "patientA_2.png"),
    but for DENTEX that trailing number IS the unique image id, so stripping it
    collapsed every image in a split into one fake patient bucket -- i.e.
    patient_level_split silently put 100% of images in a single split and
    0 in the other two. Confirmed by running it against the downloaded data.

    Since there is no recoverable patient id, this now returns one key per
    image (an image-level split). Whether that is also patient-safe depends on
    whether DENTEX's 1005 fully-labelled images are one-per-patient, which
    cannot be verified from the released data -- state this as a limitation in
    the paper rather than assuming it.
    """
    return Path(file_name).stem


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

    The real quadrant-enumeration-diagnosis json has no flat "categories" /
    "category_id" -- it's a multi-task schema with categories_1 (quadrant),
    categories_2 (enumeration), categories_3 (diagnosis) and matching
    category_id_1/2/3 per annotation. Diagnosis (what we care about) is task 3.
    Confirmed against the actual downloaded file; the flat schema is kept as a
    fallback for simple COCO-style fixtures.
    """
    if "categories" in coco:
        id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
        key = "category_id"
    else:
        id_to_name = {c["id"]: c["name"] for c in coco.get("categories_3", [])}
        key = "category_id_3"
    counts: dict[str, int] = defaultdict(int)
    for ann in coco.get("annotations", []):
        counts[id_to_name.get(ann.get(key), str(ann.get(key)))] += 1
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
