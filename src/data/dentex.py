"""
DENTEX dataset access.

DENTEX ships COCO-format annotations for panoramic dental X-rays. Only the
fully-labelled third split (quadrant-enumeration-diagnosis) carries the
diagnosis classes we care about: Caries, Deep Caries, Periapical Lesion,
Impacted (see DIAGNOSIS_CLASSES for the exact names/order the data uses). For
a caries-only detector you can either keep all four diagnosis classes or
collapse to {Caries, Deep Caries} vs background -- decide this early in
Phase 2, it changes the label maps everywhere downstream.

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

# The four DENTEX diagnosis classes, in the exact id order the real
# categories_3 list uses (confirmed against the downloaded annotation file:
# id 0=Impacted, 1=Caries, 2=Periapical Lesion, 3=Deep Caries). Names match
# the dataset's own capitalization -- do not lowercase/reorder these, or a
# name-based lookup against real annotations will silently miss.
DIAGNOSIS_CLASSES = ["Impacted", "Caries", "Periapical Lesion", "Deep Caries"]

# If you go caries-only, this is the collapse map, keyed on the real
# categories_3 names above. Everything not listed -> bg.
CARIES_ONLY_MAP = {"Caries": 0, "Deep Caries": 0}


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


def _category_schema(coco: dict) -> tuple[str, dict[int, str]]:
    """Figure out which category schema a coco dict uses.

    Real DENTEX diagnosis jsons have no flat "categories" / "category_id" --
    it's a multi-task schema with categories_1 (quadrant), categories_2
    (enumeration), categories_3 (diagnosis) and matching category_id_1/2/3 per
    annotation. Diagnosis (what we care about) is task 3. Confirmed against
    the actual downloaded file; the flat schema is kept as a fallback for
    simple COCO-style fixtures. class_balance, class_weights, and
    repeat_factors all share this so the schema only needs handling in one
    place.

    Returns: (annotation field name to read, {category_id: category_name}).
    """
    if "categories" in coco:
        return "category_id", {c["id"]: c["name"] for c in coco["categories"]}
    return "category_id_3", {c["id"]: c["name"] for c in coco.get("categories_3", [])}


def class_balance(coco: dict) -> dict[str, int]:
    """Count annotations per category name. Use this to check imbalance early.

    DENTEX is imbalanced (impacted and caries dominate; periapical is rare), so
    look at this before training and decide on RepeatFactorSampler weights or a
    class-balanced loss. HierarchicalDet already turns on USE_FED_LOSS for this
    reason. See class_weights() / repeat_factors() for the two actual
    handling strategies, built from these counts.
    """
    key, id_to_name = _category_schema(coco)
    counts: dict[str, int] = defaultdict(int)
    for ann in coco.get("annotations", []):
        counts[id_to_name.get(ann.get(key), str(ann.get(key)))] += 1
    return dict(counts)


def class_weights(coco: dict, scheme: str = "effective_num", beta: float = 0.999) -> dict[str, float]:
    """Per-class loss weights from class_balance() counts.

    For a class-balanced loss (the "class-balanced loss" option TASKS.md
    calls out, alongside RepeatFactorSampler-style oversampling in
    repeat_factors()). Weights are normalized to mean 1.0 so overall loss
    magnitude doesn't shift when you turn this on.

    Args:
        scheme: "effective_num" (Cui et al. 2019, "Class-Balanced Loss Based
            on Effective Number of Samples") -- softer than raw inverse
            frequency on very rare classes, which matters here since
            Periapical Lesion (4.5% of annotations) is rare enough that plain
            inverse-frequency weighting would dominate the loss with it.
            "inverse_freq" is the classic total / (n_classes * count) weighting.
        beta: effective_num's hyperparameter, closer to 1.0 = closer to
            inverse-frequency behavior. Cui et al.'s paper defaults (0.99-0.9999
            depending on dataset size) motivate the 0.999 default here.

    Returns:
        {category_name: weight}, mean 1.0.
    """
    counts = class_balance(coco)
    if not counts:
        return {}

    if scheme == "inverse_freq":
        total = sum(counts.values())
        n_classes = len(counts)
        raw = {name: total / (n_classes * count) for name, count in counts.items()}
    elif scheme == "effective_num":
        raw = {name: (1 - beta) / (1 - beta ** count) for name, count in counts.items()}
    else:
        raise ValueError(f"unknown scheme: {scheme!r} (use 'effective_num' or 'inverse_freq')")

    mean_w = sum(raw.values()) / len(raw)
    return {name: w / mean_w for name, w in raw.items()}


def repeat_factors(coco: dict, repeat_thresh: float = 0.5) -> dict[int, float]:
    """Per-image oversampling factor for rare categories (LVIS-style).

    Mirrors detectron2's RepeatFactorTrainingSampler algorithm (Gupta et al.,
    "LVIS: A Dataset for Large Vocabulary Instance Segmentation"): for each
    category c, category_repeat = max(1, sqrt(repeat_thresh / freq_c)), where
    freq_c is the fraction of images containing at least one instance of c.
    Each image's repeat factor is the max over the categories it contains, so
    an image with a rare Periapical Lesion gets oversampled even if it also has
    common Caries. Feed this straight into detectron2's
    RepeatFactorTrainingSampler in Phase 3, or use it as manual sample weights.

    Returns:
        {image_id: repeat_factor}. Images with no annotations get 1.0
        (never oversampled, never dropped).
    """
    key, _ = _category_schema(coco)
    by_image = index_annotations(coco)
    images = coco.get("images", [])
    n_images = len(images)
    if n_images == 0:
        return {}

    images_with_cat: dict[int, set] = defaultdict(set)
    for img in images:
        cats_here = {a.get(key) for a in by_image.get(img["id"], [])}
        for c in cats_here:
            images_with_cat[c].add(img["id"])

    cat_repeat = {
        c: max(1.0, (repeat_thresh / (len(imgs) / n_images)) ** 0.5)
        for c, imgs in images_with_cat.items()
    }

    factors = {}
    for img in images:
        cats_here = {a.get(key) for a in by_image.get(img["id"], [])}
        factors[img["id"]] = max((cat_repeat[c] for c in cats_here), default=1.0)
    return factors


def to_detectron2_dicts(
    coco: dict, image_root: str | Path, image_ids: list[int] | None = None
) -> list[dict]:
    """Convert our COCO dict into detectron2's standard "dataset dicts" list.

    Does NOT hard-depend on detectron2 at import time (lazy import of
    BoxMode, matching the pattern in src/eval/metrics.py:coco_map) -- only
    needs it installed when this is actually called, i.e. from the detector
    venv (see SETUP.md), not the core one.

    Each annotation keeps category_id_1/2/3 (not collapsed to a single
    "category_id") since DiffusionDet's hierarchical loss needs all three --
    a custom dataset mapper reads these into gt_classes_1/2/3 on the Instances
    object. Do NOT use detectron2's stock annotations_to_instances() here, it
    only knows about a single "category_id".

    Also includes a flat "category_id" (aliased to category_id_3, the
    diagnosis task) purely because detectron2's own
    get_detection_dataset_dicts() calls print_instances_class_histogram()
    internally, which unconditionally does `x["category_id"]` and raises
    KeyError without it -- confirmed by running build_detection_train_loader
    against this data. It's not used for anything but that summary printout;
    the real training path reads category_id_1/2/3.

    Args:
        coco: a COCO-format dict, e.g. from load_coco().
        image_root: directory the "xrays/" images live in.
        image_ids: restrict to these image ids (e.g. one arm of a split from
            patient_level_split); None means all images in coco.

    Returns:
        List of dicts: {"file_name", "image_id", "height", "width",
        "annotations": [{"bbox", "bbox_mode", "category_id_1/2/3", "iscrowd"}]}.
    """
    from detectron2.structures import BoxMode

    # Unlike class_balance/class_weights/repeat_factors, this function is
    # multi-task-schema-only by design (see docstring: it needs all three of
    # category_id_1/2/3 for the hierarchical loss). A flat-schema COCO dict
    # has no category_id_3 on its annotations, so `ann.get("category_id_3",
    # 0)` below would silently label every annotation class 0 instead of
    # erroring -- fail loudly here instead.
    if "categories_3" not in coco:
        raise ValueError(
            "to_detectron2_dicts() requires DENTEX's multi-task schema "
            "(categories_1/2/3 + category_id_1/2/3 per annotation), not a "
            "flat COCO categories/category_id schema -- got a coco dict with "
            "no 'categories_3' key. class_balance()/class_weights()/"
            "repeat_factors() support both schemas via _category_schema(); "
            "this function does not, because DiffusionDet's hierarchical "
            "loss needs all three tasks, not a single collapsed category_id."
        )

    image_root = Path(image_root)
    by_image = index_annotations(coco)
    wanted = set(image_ids) if image_ids is not None else None

    dicts = []
    for img in coco.get("images", []):
        if wanted is not None and img["id"] not in wanted:
            continue
        annos = []
        for ann in by_image.get(img["id"], []):
            annos.append({
                "bbox": ann["bbox"],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id_1": ann.get("category_id_1", 0),
                "category_id_2": ann.get("category_id_2", 0),
                "category_id_3": ann.get("category_id_3", 0),
                "category_id": ann.get("category_id_3", 0),  # see docstring
                "iscrowd": ann.get("iscrowd", 0),
            })
        dicts.append({
            "file_name": str(image_root / img["file_name"]),
            "image_id": img["id"],
            "height": img["height"],
            "width": img["width"],
            "annotations": annos,
        })
    return dicts


def register_dentex_detectron2(
    coco: dict,
    image_root: str | Path,
    split_ids: dict[str, list[int]],
    name_prefix: str = "custom",
) -> None:
    """Register DENTEX splits with detectron2's DatasetCatalog/MetadataCatalog.

    Registers "{name_prefix}_train_class" and "{name_prefix}_validation_class"
    (matching what HierarchicalDet's configs put in DATASETS.TRAIN/TEST) plus
    a "{name_prefix}_test_class" if a "test" key is present in split_ids.
    Needs detectron2 installed (lazy import) -- call this from the detector
    venv, not the core one. Idempotent: re-registering the same name is a
    detectron2 error, so this skips names already in DatasetCatalog.

    Args:
        coco: a COCO-format dict, e.g. from load_coco().
        image_root: directory the "xrays/" images live in.
        split_ids: {"train": [...], "val": [...], "test": [...]} of image_ids,
            e.g. from patient_level_split().
        name_prefix: dataset name prefix; "custom" matches the vendored
            HierarchicalDet configs' DATASETS.TRAIN/TEST values exactly.
    """
    from detectron2.data import DatasetCatalog, MetadataCatalog

    split_to_suffix = {"train": "train_class", "val": "validation_class", "test": "test_class"}
    for split, suffix in split_to_suffix.items():
        if split not in split_ids:
            continue
        name = f"{name_prefix}_{suffix}"
        if name in DatasetCatalog.list():
            continue
        ids = split_ids[split]
        DatasetCatalog.register(name, lambda ids=ids: to_detectron2_dicts(coco, image_root, ids))
        MetadataCatalog.get(name).set(
            thing_classes=DIAGNOSIS_CLASSES,
            thing_classes_1=[c["name"] for c in coco.get("categories_1", [])],
            thing_classes_2=[c["name"] for c in coco.get("categories_2", [])],
            thing_classes_3=DIAGNOSIS_CLASSES,
        )


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
