"""Read real radiographs corrupted by *CheXphoto's* capture model, not ours.

`RealImageChannel` (src/models/real_channel.py) renders each capture with this
project's own simulator. That is the arrangement every result so far has run
under, and it has a standing objection: the simulator and the method were
designed together, so "the guarantee holds under this capture process" could
in principle be a statement about the process rather than about the method.

This channel answers that objection by swapping the renderer and nothing else.
The reader is the *same fitted model* -- same features, same diagnosis head,
same degradation head, trained on this project's simulator exactly as before --
but at read time the photograph it is handed was corrupted by CheXphoto's
released transformations (`src/data/chexphoto_transforms.py`), written by the
authors of the closest prior work, for chest radiographs, before this project
existed. Everything downstream (calibrator, evidence ladder, policies, docket)
is untouched.

**The severity-to-level mapping.** The capture session still produces this
project's continuous [0, 1] severity per artifact; `severity_to_level` bins it
onto CheXphoto's ordinal 1-4. The artifact correspondence is:

    blur      -> blur              (defocus, both)
    glare     -> glare_matte       (soft specular highlight, both)
    low_light -> brightness_down   (underexposure, both)
    angle     -> tilt              (off-axis perspective warp, both)
    jpeg      -> moire             (SUBSTITUTION, not a correspondence)

The first four are genuine like-for-like. The last is not: CheXphoto has no
compression artifact, and this project has no moire -- CheXphoto photographs
monitors, where moire is the signature artifact, while this project's taxonomy
was grounded on photographing films on a lightbox, where it does not arise
(`docs/simulator_grounding.md`). Rather than drop a channel from each side, the
compression slot drives moire, so that the CheXphoto arm exercises the full
photographic recipe (glare_matte + moire + tilt) the paper actually released.
Set `substitute_moire_for_jpeg=False` to leave the slot empty instead; E14
reports the per-artifact damage curves separately either way, so the
substitution never has to be taken on faith.

**What the degradation head sees.** It was trained by weak supervision on this
project's simulator, and here it is asked to name artifacts produced by a
different corruption model. That mismatch is deliberate and is part of what E14
measures: a targeting instrument is only useful if it survives capture
conditions its training set did not contain.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.data.chexphoto_transforms import apply_perturbation, severity_to_level
from src.data.degradation import DEGRADATION_NAMES
from src.models.diagnostic import Case, DiagnosticChannel, Reading, predicted_usability
from src.models.real_channel import RealImageChannel, extract_features

# This project's artifact -> CheXphoto perturbation. See the module docstring
# for why `jpeg -> moire` is a substitution rather than a correspondence.
ARTIFACT_TO_PERTURBATION = {
    "blur": "blur",
    "glare": "glare_matte",
    "low_light": "brightness_down",
    "angle": "tilt",
    "jpeg": "moire",
}

# Applied in the same physical order the simulator uses: geometry at capture,
# then light, then optics, then the sensor-side artifact last.
APPLY_ORDER = ("angle", "low_light", "glare", "blur", "jpeg")


def corrupt_like_chexphoto(
    img: np.ndarray,
    severities: dict[str, float],
    rng: np.random.Generator,
    substitute_moire_for_jpeg: bool = True,
) -> np.ndarray:
    """Apply CheXphoto perturbations at levels binned from `severities`."""
    out = img
    for artifact in APPLY_ORDER:
        if artifact == "jpeg" and not substitute_moire_for_jpeg:
            continue
        level = severity_to_level(severities.get(artifact, 0.0))
        if level is None:
            continue
        out = apply_perturbation(out, ARTIFACT_TO_PERTURBATION[artifact], level, rng)
    return out


@dataclass
class CheXphotoChannel(DiagnosticChannel):
    """A fitted `RealImageChannel` reading CheXphoto-corrupted photographs.

    Takes the reader by composition rather than by subclassing, so it is
    impossible to accidentally retrain or re-fit anything: the arm differs from
    the simulator arm in the renderer and in nothing else.
    """

    reader: RealImageChannel
    substitute_moire_for_jpeg: bool = True
    name: str = "chexphoto_corruption"

    def read(self, case: Case, severities: dict[str, float], rng: np.random.Generator) -> Reading:
        if case.payload is None:
            raise ValueError("CheXphotoChannel needs an image on Case.payload")
        image = case.payload.image if hasattr(case.payload, "image") else case.payload
        corrupted = corrupt_like_chexphoto(
            image, severities, rng, substitute_moire_for_jpeg=self.substitute_moire_for_jpeg
        )
        feats = extract_features(corrupted)[None, :]

        score = float(self.reader.diagnosis.predict_proba(feats)[0])
        pred = np.clip(self.reader.degradation.predict(feats)[0], 0.0, 1.0)
        degradation = {name: float(pred[i]) for i, name in enumerate(DEGRADATION_NAMES)}
        return Reading(
            score=score,
            degradation=degradation,
            usability=predicted_usability(degradation),
            # Same definition as the simulator arm, so the two are on one
            # scale: the worst artifact the scene actually contains.
            true_quality=float(np.clip(1.0 - max(severities.values(), default=0.0), 0.0, 1.0)),
        )
