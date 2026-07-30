"""Checks for the real nn.Module implementations in confidence_head.py/fusion.py.

Skipped entirely if torch is not installed -- these need the separate
detector-stack environment (see SETUP.md Track B), not the core venv. Uses
synthetic feature maps shaped like the real backbone's output (confirmed
2026-07-29: FPN p5 is (B, 256, H, W)) rather than requiring the full detector
stack to be installed just to run these unit tests.

Run with: source .venv-detector/bin/activate && python -m pytest tests/test_models_torch.py
"""

import pytest

torch = pytest.importorskip("torch")

from src.data.degradation import DEGRADATION_NAMES
from src.models.confidence_head import ConfidenceHead
from src.models.fusion import BurstFusion


def _fake_feature_map(batch=1, channels=256, h=25, w=25):
    return torch.rand(batch, channels, h, w)


def test_confidence_head_output_shapes_and_ranges():
    head = ConfidenceHead(in_features=256)
    head.eval()
    feat = _fake_feature_map()
    with torch.no_grad():
        severity_pred, usability = head(feat)
    assert severity_pred.shape == (1, len(DEGRADATION_NAMES))
    assert usability.shape == (1,)
    assert torch.all(severity_pred >= 0) and torch.all(severity_pred <= 1)
    assert torch.all(usability >= 0) and torch.all(usability <= 1)


def test_confidence_head_dominant_degradation_matches_argmax():
    head = ConfidenceHead(in_features=256)
    head.eval()
    with torch.no_grad():
        severity_pred, _ = head(_fake_feature_map())
    name = head.dominant_degradation(severity_pred)
    assert name in DEGRADATION_NAMES
    assert name == DEGRADATION_NAMES[int(torch.argmax(severity_pred[0]).item())]


def test_confidence_head_batched_input():
    head = ConfidenceHead(in_features=256)
    head.eval()
    with torch.no_grad():
        severity_pred, usability = head(_fake_feature_map(batch=4))
    assert severity_pred.shape == (4, len(DEGRADATION_NAMES))
    assert usability.shape == (4,)


def test_burst_fusion_output_shape_and_attn_sums_to_one():
    fusion = BurstFusion(feature_dim=256, n_frames=3)
    fusion.eval()
    frames = [_fake_feature_map() for _ in range(3)]
    with torch.no_grad():
        fused, attn = fusion(frames)
    assert fused.shape == frames[0].shape
    assert attn.shape == (1, 3)
    assert torch.allclose(attn.sum(dim=-1), torch.ones(1), atol=1e-5)


def test_cross_frame_agreement_is_bounded():
    fusion = BurstFusion(feature_dim=256, n_frames=3)
    fusion.eval()
    frames = [_fake_feature_map() for _ in range(3)]
    with torch.no_grad():
        _, attn = fusion(frames)
    agreement = BurstFusion.cross_frame_agreement(attn)
    assert 0.0 <= agreement <= 1.0

    # a fully peaked distribution should read as near-zero agreement, per the
    # documented (unvalidated) entropy interpretation
    peaked = torch.tensor([[0.999, 0.0005, 0.0005]])
    assert BurstFusion.cross_frame_agreement(peaked) < 0.1

    # a uniform distribution should read as near-maximal agreement
    uniform = torch.tensor([[1 / 3, 1 / 3, 1 / 3]])
    assert BurstFusion.cross_frame_agreement(uniform) > 0.99


if __name__ == "__main__":
    for fn in [
        test_confidence_head_output_shapes_and_ranges,
        test_confidence_head_dominant_degradation_matches_argmax,
        test_confidence_head_batched_input,
        test_burst_fusion_output_shape_and_attn_sums_to_one,
        test_cross_frame_agreement_is_bounded,
    ]:
        fn()
        print("PASS", fn.__name__)
