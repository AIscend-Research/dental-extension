"""
Multi-shot fusion module (Phase 3, optional novelty track).

Fuses 2-3 quick degraded phone shots of the same film into one cleaner latent
representation before detection. The bet: no single burst frame is clean, but
glare and angle jitter are independent across frames, so a learned fusion can
recover detail no single frame has -- and the cross-frame agreement doubles as
a confidence signal (frames that disagree = low trust).

Inspiration is burst photography (Night Sight-style multi-frame merge). The
bursts come from src/data/degradation.make_burst().

Stub: needs torch + the detector encoder (Phase 3). Interface pinned here so
the burst data generation (already working) and the cross-frame confidence
signal can be developed against a stable contract.
"""

from __future__ import annotations


class BurstFusion:
    """Merge N encoded frames into one feature map.

    Options to try in Phase 3, roughly in order of effort:
      1. attention-weighted average over frame features (simplest, strong
         baseline; weights double as a per-frame quality estimate).
      2. a tiny recurrent / transformer merge over the frame sequence.
      3. explicit alignment (homography) before merge, if angle jitter is large.

    input : list of per-frame feature maps from the shared image encoder
    output: one fused feature map + per-frame attention weights (the weights
            feed the cross-frame confidence signal).
    """

    def __init__(self, feature_dim: int, n_frames: int = 3):
        self.feature_dim = feature_dim
        self.n_frames = n_frames
        # TODO(phase3): define fusion layers as an nn.Module.

    def forward(self, frame_features: list):
        """frame_features -> (fused_features, attn_weights). TODO(phase3)."""
        raise NotImplementedError("Build the fusion nn.Module in Phase 3.")

    @staticmethod
    def cross_frame_agreement(attn_weights) -> float:
        """Turn frame weights / disagreement into a [0,1] confidence. TODO(phase3).

        Low agreement across frames -> low confidence -> feed into decide().
        """
        raise NotImplementedError
