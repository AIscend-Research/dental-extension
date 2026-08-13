"""Tooth crops from DENTEX, for the real-image arm of the benchmark.

What this loader can and cannot give you, stated up front because it bounds
every claim the real-image experiment is allowed to make:

  * DENTEX annotates only *abnormal* teeth. There are no sound-tooth
    annotations anywhere in the dataset. So a "caries vs healthy" task cannot
    be built from it without inventing negatives -- e.g. by sampling
    unannotated regions, which would silently label undiagnosed disease as
    healthy. This module refuses to do that.
  * What it builds instead is one of two binary tasks over annotated teeth,
    both from real dentists' labels -- see TASKS below. Neither is "caries vs
    healthy", so "convicting" means something narrower here than it does in
    the simulated benchmark, and the experiment says so rather than quietly
    reusing the same words.
  * The publicly released split with diagnosis labels is small: 50 panoramic
    radiographs, 182 annotated teeth, of which 32 are Deep Caries. That is
    enough to measure how a real classifier degrades under rendered capture
    artifacts -- a within-model relative comparison -- and not enough to train
    a competitive detector, nor to support a well-populated conformal
    calibration set on the rarer classes.

Splitting is by SOURCE IMAGE, never by tooth. Several teeth come from one
panoramic, and letting teeth from the same radiograph straddle the split would
leak the patient's anatomy, exposure and machine calibration across it. With
182 crops that leak would be large enough to manufacture most of any result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# DENTEX's categories_3 ids, confirmed against the released validation json.
CARIES_IDS = {1, 3}  # Caries, Deep Caries
OTHER_IDS = {0, 2}  # Impacted, Periapical Lesion

# Two binary tasks are buildable from these labels, and they are not
# interchangeable. Which one is used changes what the real-image arm measures.
#
#   caries_vs_other  CARIES {Caries, Deep Caries} vs {Impacted, Periapical}.
#       Measured clean AUC ~0.997 with the linear reader here -- effectively
#       saturated. Impacted teeth are unerupted and often lying sideways, so
#       the class is separable on coarse morphology alone and never requires
#       reading the lesion. Useful as a sanity check that the pipeline runs
#       end to end; useless for asking which artifacts destroy diagnostic
#       signal, because a saturated classifier has no signal left to lose.
#
#   caries_vs_deep   Caries vs Deep Caries, i.e. how far the lesion has
#       progressed. Both classes are erupted teeth of ordinary shape, so
#       morphology carries almost nothing and the decision rests on
#       fine-grained density and depth -- the same kind of evidence real
#       caries detection depends on. This is the task the artifact-damage
#       analysis should use.
TASKS = {
    "caries_vs_other": ({1, 3}, {0, 2}),
    "caries_vs_deep": ({3}, {1}),  # positive = Deep Caries, negative = Caries
}

DEFAULT_ROOT = Path("data/dentex/DENTEX")
DEFAULT_ANNOTATIONS = DEFAULT_ROOT / "validation_triple.json"
DEFAULT_IMAGES = DEFAULT_ROOT / "validation_data/quadrant_enumeration_disease/xrays"


@dataclass
class ToothCrop:
    """One annotated tooth.

    Attributes:
        image: BGR uint8 crop, resized to a fixed size.
        label: 1 = caries (incl. deep caries), 0 = other pathology.
        source_image_id: the panoramic it came from. Splits group on this.
        category_id: the original DENTEX categories_3 id, kept for auditing.
    """

    image: np.ndarray
    label: int
    source_image_id: int
    category_id: int


def load_tooth_crops(
    annotations: Path | str = DEFAULT_ANNOTATIONS,
    image_root: Path | str = DEFAULT_IMAGES,
    crop_size: int = 96,
    context: float = 0.15,
    task: str = "caries_vs_deep",
) -> list[ToothCrop]:
    """Load the annotated teeth for one binary task as fixed-size crops.

    Args:
        task: which binary problem to build; see TASKS. Defaults to
            `caries_vs_deep`, the fine-grained one, because the coarse task
            saturates and cannot support the artifact-damage analysis.
        crop_size: output side length in pixels. Crops are resized to a square,
            which distorts aspect ratio -- teeth are taller than wide (median
            box here is roughly 170x270). That is deliberate and uniform: the
            classifier sees a consistent geometry, and no aspect information is
            available for it to exploit as a shortcut.
        context: fraction of the box size to pad around the tooth, so the
            crop includes some surrounding bone rather than cutting exactly at
            the annotation boundary.

    Returns:
        A list of ToothCrop. Annotations whose category is not in CARIES_IDS or
        OTHER_IDS are skipped; so are boxes that fall outside the image.
    """
    annotations = Path(annotations)
    image_root = Path(image_root)
    if not annotations.exists():
        raise FileNotFoundError(
            f"{annotations} not found. Fetch the DENTEX validation split first "
            "(see scripts/download_dentex.py -- the 150MB validation zip is "
            "enough for this arm; the 11GB training zip is not needed)."
        )

    if task not in TASKS:
        raise KeyError(f"unknown task {task!r}; have {sorted(TASKS)}")
    positive_ids, negative_ids = TASKS[task]

    data = json.loads(annotations.read_text())
    images = {im["id"]: im for im in data["images"]}
    cache: dict[int, np.ndarray] = {}
    crops: list[ToothCrop] = []

    for ann in data["annotations"]:
        cat = ann.get("category_id_3")
        if cat in positive_ids:
            label = 1
        elif cat in negative_ids:
            label = 0
        else:
            continue

        meta = images[ann["image_id"]]
        if ann["image_id"] not in cache:
            path = image_root / meta["file_name"]
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"could not read {path}")
            cache[ann["image_id"]] = img
        img = cache[ann["image_id"]]

        x, y, w, h = ann["bbox"]
        pad_x, pad_y = context * w, context * h
        x1 = int(max(0, x - pad_x))
        y1 = int(max(0, y - pad_y))
        x2 = int(min(img.shape[1], x + w + pad_x))
        y2 = int(min(img.shape[0], y + h + pad_y))
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue

        crop = cv2.resize(img[y1:y2, x1:x2], (crop_size, crop_size), interpolation=cv2.INTER_AREA)
        crops.append(
            ToothCrop(
                image=crop,
                label=label,
                source_image_id=int(ann["image_id"]),
                category_id=int(cat),
            )
        )
    return crops


def split_by_source_image(
    crops: list[ToothCrop],
    fractions: tuple[float, float, float] = (0.5, 0.25, 0.25),
    seed: int = 0,
) -> tuple[list[ToothCrop], list[ToothCrop], list[ToothCrop]]:
    """Group-split into (train, calibration, test) on the source radiograph.

    Grouping is on `source_image_id`, so no panoramic contributes teeth to more
    than one split. The three-way split is what the framework needs: one part
    to fit the reader, a *separate* part to calibrate the e-values on (reusing
    training data there would make the null distribution optimistic), and a
    third to evaluate.

    Class balance is not enforced within splits -- with 50 source images,
    forcing it would mean choosing which images land where, and that choice
    would be made using the labels. The realised balance is reported instead.
    """
    rng = np.random.default_rng(seed)
    image_ids = sorted({c.source_image_id for c in crops})
    rng.shuffle(image_ids)

    n = len(image_ids)
    n_train = int(round(fractions[0] * n))
    n_cal = int(round(fractions[1] * n))
    groups = {
        "train": set(image_ids[:n_train]),
        "cal": set(image_ids[n_train : n_train + n_cal]),
        "test": set(image_ids[n_train + n_cal :]),
    }
    return tuple(  # type: ignore[return-value]
        [c for c in crops if c.source_image_id in groups[k]] for k in ("train", "cal", "test")
    )


def describe(crops: list[ToothCrop], name: str = "") -> str:
    n = len(crops)
    pos = sum(c.label for c in crops)
    imgs = len({c.source_image_id for c in crops})
    return (
        f"{name:<12} {n:>4} teeth from {imgs:>3} radiographs | "
        f"caries {pos:>3} ({pos / n:.2f}) | other {n - pos:>3}"
        if n
        else f"{name:<12} empty"
    )
