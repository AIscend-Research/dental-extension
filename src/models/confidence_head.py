"""
Confidence / quality head (Phase 3).

A small head that hangs off the detector's image-encoder features and predicts,
per image, how degraded it is -- and crucially *which* degradation dominates,
not just a scalar trust score. Predicting the type is what lets the system give
actionable feedback ("too much glare, retake by the window") instead of a bare
"image rejected". The weak-supervision targets are the severity label vectors
produced by src/data/degradation.py.

This is a stub: it needs torch and the detector's feature map, which arrive in
Phase 3. The interface and the loss shape are pinned down here so the eval code
(src/eval/metrics.py, already working) and the decision thresholds can be built
against it in parallel.
"""

from __future__ import annotations

from src.data.degradation import DEGRADATION_NAMES

# Decision outcomes the system can take on one image.
DECISIONS = ["predict", "retake", "refer_clinician"]


class ConfidenceHead:
    """Predicts per-degradation severity and an overall usability score.

    Design (implement in Phase 3 as an nn.Module):
      input : pooled feature map from the detector's image encoder
      output: severity_logits over DEGRADATION_NAMES (regression or ordinal),
              plus a scalar usability score in [0, 1].
      loss  : weak supervision against DegradationResult.label_vector().
              Start with smooth-L1 on severities; the usability scalar can be
              derived (1 - max severity) or learned against detector correctness.
    """

    def __init__(self, in_features: int, degradations: list[str] | None = None):
        self.in_features = in_features
        self.degradations = degradations or DEGRADATION_NAMES
        # TODO(phase3): define the actual layers once this is an nn.Module.

    def forward(self, features):
        """features -> (severity_pred, usability_score). TODO(phase3)."""
        raise NotImplementedError("Build the nn.Module in Phase 3.")

    @staticmethod
    def dominant_degradation(severity_pred) -> str:
        """Name of the worst degradation, for the retake message. TODO(phase3)."""
        raise NotImplementedError


def decide(
    usability_score: float,
    detection_score: float,
    retake_below: float = 0.4,
    refer_below: float = 0.7,
) -> str:
    """Map scores to an action.

    The two thresholds are the operating points you will tune in Phase 4 using
    the safe_deferral_rate curve -- pick them where the accuracy/deferral
    tradeoff matches what a clinic can tolerate. These defaults are placeholders,
    not tuned values.

    Returns one of DECISIONS.
    """
    if usability_score < retake_below:
        return "retake"
    if usability_score < refer_below or detection_score < refer_below:
        return "refer_clinician"
    return "predict"
