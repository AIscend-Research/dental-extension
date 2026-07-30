"""
Confidence / quality head (Phase 3).

A small head that hangs off the detector's image-encoder features and predicts,
per image, how degraded it is -- and crucially *which* degradation dominates,
not just a scalar trust score. Predicting the type is what lets the system give
actionable feedback ("too much glare, retake by the window") instead of a bare
"image rejected". The weak-supervision targets are the severity label vectors
produced by src/data/degradation.py.

Real nn.Module implementation, confirmed against the actual HierarchicalDet
backbone (2026-07-29, CPU/MPS via the recipe in SETUP.md): the FPN outputs
p2-p6, all 256 channels (see model.backbone.output_shape()) -- ConfidenceHead
defaults to in_features=256 to match p5 (the coarsest/most global level,
the usual choice for an image-level task like this). Verified: builds, runs a
forward pass on a real 256x25x25 feature map, and both outputs land in [0, 1].

This module intentionally does NOT hard-depend on torch at import time: it is
imported by src/eval/metrics.py (sweep_decision_thresholds -> decide(), which
is pure Python) in the core, no-GPU environment where torch is not installed.
If torch is unavailable, ConfidenceHead falls back to the pre-Phase-3
interface stub so the rest of the codebase still imports cleanly; decide()
and DECISIONS are unaffected either way.
"""

from __future__ import annotations

from src.data.degradation import DEGRADATION_NAMES

# Decision outcomes the system can take on one image.
DECISIONS = ["predict", "retake", "refer_clinician"]

try:
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


if _HAS_TORCH:

    class ConfidenceHead(nn.Module):
        """Predicts per-degradation severity and an overall usability score.

        input : pooled feature map from the detector's image encoder,
                (B, in_features, H, W) -- e.g. FPN level p5, (B, 256, H, W).
        output: (severity_pred, usability_score)
                severity_pred: (B, n_degradations) in [0, 1] per DEGRADATION_NAMES
                usability_score: (B,) in [0, 1], 1 = fully trust the detection
        loss  : weak supervision against DegradationResult.label_vector() with
                smooth-L1 (or BCE, since both are in [0, 1]) on severity_pred.
                usability_score can be learned against detector correctness,
                or derived as (1 - severity_pred.max(dim=-1)) as a cheap start.
        """

        def __init__(self, in_features: int = 256, degradations: list[str] | None = None):
            super().__init__()
            self.in_features = in_features
            self.degradations = degradations or DEGRADATION_NAMES
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.trunk = nn.Sequential(
                nn.Linear(in_features, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
            )
            self.severity_head = nn.Linear(64, len(self.degradations))
            self.usability_head = nn.Linear(64, 1)

        def forward(self, features: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
            pooled = self.pool(features).flatten(1)
            z = self.trunk(pooled)
            severity_pred = torch.sigmoid(self.severity_head(z))
            usability_score = torch.sigmoid(self.usability_head(z)).squeeze(-1)
            return severity_pred, usability_score

        def dominant_degradation(self, severity_pred: "torch.Tensor") -> str:
            """Name of the worst-predicted degradation, for the retake message.

            Args:
                severity_pred: (n_degradations,) or (1, n_degradations).
            """
            flat = severity_pred.reshape(-1)
            idx = int(torch.argmax(flat).item())
            return self.degradations[idx]

else:

    class ConfidenceHead:  # type: ignore[no-redef]
        """Interface stub used only when torch is not installed (core env)."""

        def __init__(self, in_features: int, degradations: list[str] | None = None):
            self.in_features = in_features
            self.degradations = degradations or DEGRADATION_NAMES

        def forward(self, features):
            raise NotImplementedError(
                "torch is not installed in this environment. See SETUP.md "
                "Track B for the confirmed working install recipe."
            )

        @staticmethod
        def dominant_degradation(severity_pred) -> str:
            raise NotImplementedError(
                "torch is not installed in this environment. See SETUP.md "
                "Track B for the confirmed working install recipe."
            )


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
