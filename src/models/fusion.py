"""
Multi-shot fusion module (Phase 3, optional novelty track).

Fuses 2-3 quick degraded phone shots of the same film into one cleaner latent
representation before detection. The bet: no single burst frame is clean, but
glare and angle jitter are independent across frames, so a learned fusion can
recover detail no single frame has -- and the cross-frame agreement doubles as
a confidence signal (frames that disagree = low trust).

Inspiration is burst photography (Night Sight-style multi-frame merge). The
bursts come from src/data/degradation.make_burst().

Real nn.Module implementation of option 1 from the original design note
(attention-weighted average over frame features) -- confirmed against the
actual HierarchicalDet backbone (2026-07-29, CPU/MPS): builds and runs on a
list of real 256x25x25 feature maps (FPN p5), fusing them into one map of the
same shape plus per-frame attention weights.

Same import pattern as src/models/confidence_head.py: torch is optional at
import time, since src/eval/metrics.py's sweep_decision_thresholds pulls in
confidence_head (not this module directly) from the core, no-GPU environment.
This module has no core-env caller today, but keeps the same guard for
consistency and so importing it doesn't hard-fail where torch is absent.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


if _HAS_TORCH:

    class BurstFusion(nn.Module):
        """Merge N encoded frames into one feature map via learned attention.

        Options to try beyond this, roughly in increasing effort (per the
        original design note):
          2. a tiny recurrent / transformer merge over the frame sequence.
          3. explicit alignment (homography) before merge, if angle jitter is
             large.

        input : list of N per-frame feature maps from the shared image
                encoder, each (B, C, H, W), same shape (e.g. FPN p5, 256 ch).
        output: fused feature map (B, C, H, W) + per-frame attention weights
                (B, N), softmax-normalized (the weights double as a per-frame
                quality estimate).
        """

        def __init__(self, feature_dim: int = 256, n_frames: int = 3):
            super().__init__()
            self.feature_dim = feature_dim
            self.n_frames = n_frames
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.score = nn.Sequential(
                nn.Linear(feature_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),
            )

        def forward(
            self, frame_features: list["torch.Tensor"]
        ) -> tuple["torch.Tensor", "torch.Tensor"]:
            stacked = torch.stack(frame_features, dim=1)  # (B, N, C, H, W)
            b, n, c, h, w = stacked.shape
            pooled = self.pool(stacked.reshape(b * n, c, h, w)).reshape(b, n, c)
            logits = self.score(pooled).squeeze(-1)  # (B, N)
            attn = torch.softmax(logits, dim=-1)  # (B, N)
            fused = (stacked * attn.reshape(b, n, 1, 1, 1)).sum(dim=1)  # (B, C, H, W)
            return fused, attn

        @staticmethod
        def cross_frame_agreement(attn_weights: "torch.Tensor") -> "torch.Tensor":
            """Turn frame weights into a per-sample [0, 1] confidence signal.

            One defensible reading, not a validated one: normalized entropy of
            the attention distribution. Uniform weights (high entropy) mean
            the fusion module found all frames similarly trustworthy --
            "agreement" -- and this returns a value near 1.0. A peaked
            distribution (low entropy, one frame dominating) means the frames
            were treated very differently -- "disagreement" -- and this
            returns a value near 0.0, even though the fused output might
            still be fine if the dominant frame is genuinely clean.

            Args:
                attn_weights: (B, N) attention weights from forward(), or a
                    bare (N,) vector for a single sample.

            Returns:
                (B,) tensor of agreement scores, each in [0, 1]. Entropy is
                computed per sample and normalized by log(N) -- the number of
                *frames*, not the number of elements in the batch. Flattening
                the whole (B, N) batch into one distribution instead (the
                original implementation) normalized by log(B*N) and returned
                values above 1.0 for any B > 1, e.g. 1.23 for a uniform batch
                of 2 -- silently breaking the documented [0, 1] contract.

            This is a real judgment call the original design note flagged as
            unresolved ("frames that disagree = low trust") -- it conflates
            "disagreement" with "one frame being much better than the rest,"
            which are not obviously the same thing. Validate against real
            burst data and correctness labels in Phase 4 before trusting this
            formula; it may need to be the opposite, or a different signal
            (e.g. variance of the raw frame predictions, not the attention
            weights) entirely.
            """
            if attn_weights.dim() == 1:
                attn_weights = attn_weights.unsqueeze(0)
            p = attn_weights.clamp(min=1e-8)
            n_frames = p.shape[-1]
            if n_frames <= 1:
                return torch.ones(p.shape[0], device=p.device, dtype=p.dtype)
            entropy = -(p * p.log()).sum(dim=-1)  # (B,)
            max_entropy = float(torch.log(torch.tensor(float(n_frames))))
            return (entropy / max_entropy).clamp(0.0, 1.0)

else:

    class BurstFusion:  # type: ignore[no-redef]
        """Interface stub used only when torch is not installed (core env)."""

        def __init__(self, feature_dim: int, n_frames: int = 3):
            self.feature_dim = feature_dim
            self.n_frames = n_frames

        def forward(self, frame_features: list):
            raise NotImplementedError(
                "torch is not installed in this environment. See SETUP.md "
                "Track B for the confirmed working install recipe."
            )

        @staticmethod
        def cross_frame_agreement(attn_weights):
            raise NotImplementedError(
                "torch is not installed in this environment. See SETUP.md "
                "Track B for the confirmed working install recipe."
            )
