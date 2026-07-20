"""
Detector wrapper around the HierarchicalDet baseline (Phase 3).

Thin adapter so the rest of the codebase talks to one interface while the heavy
lifting stays in external/HierarchicalDet (DiffusionDet + vendored Detectron2,
Swin-L backbone). Do NOT re-implement the detector here -- clone the baseline
(scripts/clone_baseline.sh) and call into it.

Adaptation plan (Phase 3):
  - baseline (clean): strip HierarchicalDet's 3 heads (quadrant/enumeration/
    diagnosis) down to a caries-only detector, or keep diagnosis-only. The
    configs use DiffusionDet.NUM_CLASSES = [4,8,4]; a caries-only run means
    changing the diagnosis head and dataset registration in src/data/dentex.py.
  - robustness: train the same detector on degradation-augmented data from
    src/data/degradation.py (the augmentation happens in the dataset mapper).
  - the confidence head (src/models/confidence_head.py) and, if pursued, the
    fusion module (src/models/fusion.py) attach to the image encoder output.

Before writing any of this, benchmark the baseline's size / latency / FLOPs so
the "lightweight, low-compute" claim in the proposal has real numbers behind it
(Phase 3 task). Swin-L is not small -- if latency is too high on target
hardware, that finding itself is worth reporting and may push toward a lighter
backbone.

Stub: needs the baseline cloned and the detectron2 stack installed.
"""

from __future__ import annotations


class CariesDetector:
    """Interface the training / eval code targets."""

    def __init__(self, config_path: str, weights: str | None = None):
        self.config_path = config_path
        self.weights = weights
        # TODO(phase3): build via detectron2 (build_model + DetectionCheckpointer),
        # importing from external/HierarchicalDet.

    def predict(self, image):
        """image -> list of boxes with scores + optional encoder features.

        Return the encoder feature map alongside detections so the confidence
        head and fusion module can consume it without a second forward pass.
        TODO(phase3).
        """
        raise NotImplementedError("Wire to HierarchicalDet in Phase 3.")

    def encode(self, image):
        """Run only the image encoder, for the confidence/fusion heads. TODO(phase3)."""
        raise NotImplementedError
