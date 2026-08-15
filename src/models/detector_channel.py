"""DetectorChannel: the real trained detector, behind the DiagnosticChannel interface.

This is the adapter the roadmap's "write a DetectorChannel against the
DiagnosticChannel interface" item asks for. It is the mechanical piece that
lets E3-E5 (the leaderboard, ablations, sensitivity sweep) run against the
real HierarchicalDet-based reader instead of `SurrogateChannel` or
`RealImageChannel`'s linear stand-in -- every one of those experiments takes
a `DiagnosticChannel` and does not care which concrete class it is
(`experiments/common.py::build_world` is the one place that would change).

What this file can and cannot do in THIS environment, stated plainly rather
than silently: it is written and import-safe (torch optional, same guard as
`fusion.py`/`confidence_head.py`), but it cannot be *exercised* against a
real detector here, because two prerequisites do not exist yet:

  1. `CariesDetector` (`src/models/detector.py`) holding trained weights --
     currently a stub; needs `kaggle/01_train_baseline_detector.ipynb`'s GPU
     run to produce a checkpoint.
  2. `ConfidenceHead` (`src/models/confidence_head.py`) retrained against
     that detector's real FPN p5 features rather than the standalone
     `TinyTrunk` stand-in (`docs/phase3_confidence_head_training.md`) --
     needs `kaggle/02_train_confidence_head.ipynb` re-run with (1)'s weights
     attached.

`tests/test_detector_channel.py` verifies the adapter logic now, against
fake detector/head objects shaped like the real ones (same trick
`tests/test_models_torch.py` uses for the FPN feature-map shape) -- so the
plumbing is checked even though the real weights are not.

Design note: like `RealImageChannel`, this operates on tooth CROPS
(`Case.payload` is a crop image), not a full panoramic with a box index.
That keeps the same convention the real-image arm already uses, so a Docket
built with `make_image_docket` over a crop pool works unmodified once this
channel is swapped in -- no redesign of `Case`/`Docket` needed.

The `caries_class_ids` default is deliberately not filled in: it is
downstream of the label-map decision this roadmap item is bundled with (see
`configs/default.yaml`'s `model.task` note) -- the diagnosis task's
`category_id_3` values are `{1: Caries, 3: Deep Caries}` (see
`src/data/dentex_crops.py`), so once trained diagnosis-task weights exist,
`caries_class_ids=(1, 3)` is the value to pass; a caries-only collapse would
use a different mapping entirely. This module does not guess -- it takes the
value as a constructor argument.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.degradation import DEGRADATION_NAMES
from src.models.diagnostic import Case, DiagnosticChannel, Reading, predicted_usability
from src.sim.render import render_severities

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _caries_score(detections: list[dict], caries_class_ids: tuple[int, ...]) -> float:
    """Highest-confidence detection among the caries class ids, else 0.0.

    `detections` is whatever `CariesDetector.predict()` ends up returning --
    assumed here to be a list of dicts with at least `class_id` and `score`,
    matching detectron2's usual `Instances`-derived convention. `k=2` must be
    used at inference to get `pred_classes_3` (diagnosis-level); the default
    `k=0` silently returns quadrant-level classes instead, which would make
    every score here meaningless without erroring -- see kaggle/README.md.
    """
    best = 0.0
    for det in detections:
        if det.get("class_id") in caries_class_ids:
            best = max(best, float(det.get("score", 0.0)))
    return best


def _severity_tensor_to_dict(severity_pred: "torch.Tensor", names: list[str]) -> dict[str, float]:
    flat = severity_pred.reshape(-1).clamp(0.0, 1.0)
    return {name: float(flat[i].item()) for i, name in enumerate(names)}


if _HAS_TORCH:

    @dataclass
    class DetectorChannel(DiagnosticChannel):
        """Both channels read from one real trained detector + confidence head.

        Args:
            detector: a `CariesDetector` with weights loaded (kaggle/01 output).
                Must implement `.encode(image) -> features` and
                `.predict(image) -> list[dict]` (each dict has `class_id`,
                `score`). Duck-typed rather than isinstance-checked, so the
                tests can exercise this against fakes without the real
                detector stack installed.
            head: a `ConfidenceHead`-shaped object: callable on `features`,
                returning `(severity_pred, usability_score)` tensors, exactly
                `ConfidenceHead.forward`'s signature.
            caries_class_ids: which detector class ids count as "caries" --
                see the module docstring; there is no safe default.
        """

        detector: object
        head: object
        caries_class_ids: tuple[int, ...]
        name: str = "detector"

        def read(self, case: Case, severities: dict[str, float], rng: np.random.Generator) -> Reading:
            if case.payload is None:
                raise ValueError("DetectorChannel needs a tooth-crop image on Case.payload")
            rendered = render_severities(case.payload, severities, rng=rng)

            with torch.no_grad():
                features = self.detector.encode(rendered.image)
                detections = self.detector.predict(rendered.image)
                severity_pred, _ = self.head(features)

            score = _caries_score(detections, self.caries_class_ids)
            degradation = _severity_tensor_to_dict(severity_pred, DEGRADATION_NAMES)
            return Reading(
                score=score,
                degradation=degradation,
                usability=predicted_usability(degradation),
                true_quality=float(np.clip(1.0 - max(severities.values(), default=0.0), 0.0, 1.0)),
            )

else:

    class DetectorChannel:  # type: ignore[no-redef]
        """Interface stub used only when torch is not installed (core env)."""

        def __init__(self, *args, **kwargs):
            raise NotImplementedError(
                "torch is not installed in this environment. See SETUP.md "
                "Track B for the confirmed working install recipe."
            )
