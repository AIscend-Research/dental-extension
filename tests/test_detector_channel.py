"""Checks for the DetectorChannel adapter in src/models/detector_channel.py.

Skipped entirely if torch is not installed -- same convention as
tests/test_models_torch.py. The real CariesDetector/ConfidenceHead need the
full detector stack (SETUP.md Track B) and trained weights that do not exist
yet (see detector_channel.py's module docstring), so this exercises the
adapter logic against fakes shaped like the real interfaces: `.encode()`
returns a feature map, `.predict()` returns detectron2-style detection
dicts, and the head is a real `ConfidenceHead` (that part IS testable without
trained weights -- an untrained head still returns correctly-shaped output).

Run with: source .venv-detector/bin/activate && python -m pytest tests/test_detector_channel.py
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.data.degradation import DEGRADATION_NAMES
from src.models.confidence_head import ConfidenceHead
from src.models.detector_channel import DetectorChannel
from src.models.diagnostic import Case


class _FakeDetector:
    """Duck-typed CariesDetector: fixed detections, a real-shaped feature map."""

    def __init__(self, detections):
        self._detections = detections

    def encode(self, image):
        return torch.rand(1, 256, 25, 25)

    def predict(self, image):
        return self._detections


def _crop():
    return np.random.default_rng(0).integers(0, 255, (96, 96, 3), dtype=np.uint8)


def test_detector_channel_reads_caries_class_score():
    detector = _FakeDetector([{"class_id": 1, "score": 0.83}, {"class_id": 0, "score": 0.5}])
    head = ConfidenceHead(in_features=256)
    head.eval()
    channel = DetectorChannel(detector=detector, head=head, caries_class_ids=(1, 3))

    case = Case(label=1, difficulty=0.2, payload=_crop())
    reading = channel.read(case, severities={"blur": 0.3}, rng=np.random.default_rng(1))

    assert reading.score == pytest.approx(0.83)
    assert set(reading.degradation) == set(DEGRADATION_NAMES)
    assert all(0.0 <= v <= 1.0 for v in reading.degradation.values())
    assert 0.0 <= reading.usability <= 1.0


def test_detector_channel_score_is_zero_with_no_caries_detection():
    detector = _FakeDetector([{"class_id": 0, "score": 0.9}])  # impacted, not caries
    head = ConfidenceHead(in_features=256)
    head.eval()
    channel = DetectorChannel(detector=detector, head=head, caries_class_ids=(1, 3))

    case = Case(label=0, difficulty=0.2, payload=_crop())
    reading = channel.read(case, severities={}, rng=np.random.default_rng(2))

    assert reading.score == 0.0


def test_detector_channel_requires_payload():
    detector = _FakeDetector([])
    head = ConfidenceHead(in_features=256)
    channel = DetectorChannel(detector=detector, head=head, caries_class_ids=(1, 3))
    case = Case(label=0, difficulty=0.2, payload=None)
    with pytest.raises(ValueError):
        channel.read(case, severities={}, rng=np.random.default_rng(3))
