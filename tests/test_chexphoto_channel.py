"""The CheXphoto capture channel: swaps the renderer, and nothing else.

The whole value of E14 rests on that claim. If the channel quietly changed the
reader, the features, or the usability definition, the head-to-head would be
comparing two things at once and the result would be uninterpretable. These
tests pin the parts that would make it so.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.data.degradation import DEGRADATION_NAMES
from src.models.chexphoto_channel import (
    APPLY_ORDER,
    ARTIFACT_TO_PERTURBATION,
    CheXphotoChannel,
    corrupt_like_chexphoto,
)
from src.models.diagnostic import Case, predicted_usability
from src.models.real_channel import train_real_channel


class _Crop:
    """Minimal stand-in for ToothCrop/ChestCrop: an image and a label."""

    def __init__(self, image: np.ndarray, label: int):
        self.image = image
        self.label = label


def _image(seed: int, size: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    grey = 110 + 50 * np.sin(xx / 7.0 + seed) * np.cos(yy / 9.0)
    grey = np.clip(grey + rng.normal(0, 4, grey.shape), 0, 255).astype(np.uint8)
    return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)


@pytest.fixture(scope="module")
def fitted_channel():
    """A real (tiny) fitted reader, so nothing here runs against a mock."""
    crops = [_Crop(_image(i), i % 2) for i in range(12)]
    reader = train_real_channel(crops, np.random.default_rng(0), n_renders=2)
    return reader, CheXphotoChannel(reader=reader)


def test_mapping_covers_every_artifact_exactly_once():
    assert set(ARTIFACT_TO_PERTURBATION) == set(DEGRADATION_NAMES)
    assert set(APPLY_ORDER) == set(DEGRADATION_NAMES)
    assert len(set(ARTIFACT_TO_PERTURBATION.values())) == len(DEGRADATION_NAMES)


def test_zero_severities_leave_the_photograph_untouched():
    img = _image(1)
    out = corrupt_like_chexphoto(img, {n: 0.0 for n in DEGRADATION_NAMES}, np.random.default_rng(0))
    assert np.array_equal(out, img)


def test_severity_drives_how_corrupted_the_photograph_is():
    """More severity must mean less of the original image survives.

    Measured as correlation with the clean image, not as mean pixel distance:
    the composite is not monotone in pixel distance and should not be expected
    to be. At the top of the range a level-4 blur smooths the frame back
    *toward* the original mean while destroying every structure in it, so
    pixel distance falls even as the photograph becomes unreadable. Correlation
    tracks what actually matters here, which is whether the radiograph's
    content is still there.
    """
    img = _image(2)
    rng = np.random.default_rng(0)
    clean = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64).ravel()
    correlations = []
    for severity in (0.25, 0.5, 0.75, 1.0):
        draws = []
        for _ in range(4):
            out = corrupt_like_chexphoto(img, {n: severity for n in DEGRADATION_NAMES}, rng)
            g = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float64).ravel()
            draws.append(float(np.corrcoef(clean, g)[0, 1]) if g.std() > 1e-6 else 0.0)
        correlations.append(float(np.mean(draws)))
    assert correlations == sorted(correlations, reverse=True), (
        f"surviving signal not monotone in severity: {np.round(correlations, 3)}"
    )


def test_moire_substitution_can_be_switched_off():
    img = _image(3)
    severities = {"jpeg": 1.0}
    with_moire = corrupt_like_chexphoto(img, severities, np.random.default_rng(0))
    without = corrupt_like_chexphoto(img, severities, np.random.default_rng(0), substitute_moire_for_jpeg=False)
    assert not np.array_equal(with_moire, img), "jpeg slot should drive moire by default"
    assert np.array_equal(without, img), "with the substitution off, the jpeg slot must be a no-op"


def test_reading_shape_matches_the_simulator_channel(fitted_channel):
    reader, channel = fitted_channel
    case = Case(label=1, difficulty=0.0, payload=_image(4))
    severities = {"blur": 0.5, "glare": 0.3}

    reading = channel.read(case, severities, np.random.default_rng(0))
    assert 0.0 <= reading.score <= 1.0
    assert set(reading.degradation) == set(DEGRADATION_NAMES)
    # usability must come from the same aggregate as everywhere else, or the
    # calibrator's strata would mean something different in this arm
    assert reading.usability == pytest.approx(predicted_usability(reading.degradation))
    # true_quality is the worst artifact actually in the scene, as elsewhere
    assert reading.true_quality == pytest.approx(1.0 - max(severities.values()))


def test_channel_accepts_either_a_crop_or_a_bare_image(fitted_channel):
    _, channel = fitted_channel
    img = _image(5)
    rng_a, rng_b = np.random.default_rng(7), np.random.default_rng(7)
    from_image = channel.read(Case(label=0, difficulty=0.0, payload=img), {"blur": 0.4}, rng_a)
    from_crop = channel.read(Case(label=0, difficulty=0.0, payload=_Crop(img, 0)), {"blur": 0.4}, rng_b)
    assert from_image.score == pytest.approx(from_crop.score)


def test_channel_reuses_the_reader_rather_than_refitting(fitted_channel):
    reader, channel = fitted_channel
    assert channel.reader is reader
    assert channel.reader.diagnosis is reader.diagnosis
    assert channel.reader.degradation is reader.degradation


def test_missing_payload_is_an_error(fitted_channel):
    _, channel = fitted_channel
    with pytest.raises(ValueError):
        channel.read(Case(label=1, difficulty=0.0, payload=None), {"blur": 0.5}, np.random.default_rng(0))
