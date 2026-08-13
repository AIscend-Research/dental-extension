"""Latent state of a capture session: the scene, and the person holding the phone.

Two objects persist for the whole session:

`SceneState` -- the physical situation. Four continuous factors in [0, 1]
(glare, tremor, darkness, tilt), plus a glare *azimuth* that decides whether
the specular hotspot actually lands on the tooth we care about. A fifth
degradation, `jpeg`, is fixed per session and deliberately not a scene factor:
it is the messaging app, not the room (see src/sim/instructions.py).

`Operator` -- the person. Skill governs whether an instruction is followed and
how completely; haste governs how much collateral damage the correction does;
fatigue accumulates with every shot taken, so a long retake loop degrades the
operator even when it improves the scene.

Grounding: the factor taxonomy is not invented. The smartphone oral-imaging
literature repeatedly names lighting, camera angle, motion, and specular
reflection (from saliva or from the film/lightbox) as the failure modes that
break phone-captured dental images -- see docs/simulator_grounding.md for the
citations and for which numbers below are literature-anchored versus chosen.
The *dynamics* (persistence, partial compliance, coupling) are the modelling
contribution, and are stated as assumptions rather than measurements.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

# The instructable scene factors, in a fixed order. `jpeg` is NOT here: it is
# a channel property, constant within a session and not fixable by retaking.
FACTORS: tuple[str, ...] = ("glare", "tremor", "darkness", "tilt")

# Map each latent factor onto the degradation primitive it drives in
# src/data/degradation.py. This is the only place the two vocabularies meet.
FACTOR_TO_DEGRADATION: dict[str, str] = {
    "glare": "glare",
    "tremor": "blur",
    "darkness": "low_light",
    "tilt": "angle",
}

# Side effects of correcting one factor on the others.
#
# COUPLING[corrected][affected] = how much of the correction magnitude leaks
# into `affected`, before being scaled by the operator's haste. Read each row
# as a physical story:
#
#   glare    -- to kill a reflection you turn the film away from the lamp, so
#               you end up off-axis (tilt) and further from the light (darkness).
#   darkness -- to get more light you move toward the window or lamp, which is
#               exactly what puts a specular highlight back on the film (glare).
#   tilt     -- squaring up means re-gripping and re-aiming by hand (tremor).
#   tremor   -- bracing is cheap, but leaning in to steady the phone can shadow
#               the film slightly (darkness).
#
# The glare<->darkness pair is the interesting one: it is a genuine trade-off,
# so a greedy "always fix the worst factor" policy can oscillate between the
# two and burn its whole capture budget without converging. That failure mode
# is a *prediction* of this matrix, and E4 measures whether it happens.
COUPLING: dict[str, dict[str, float]] = {
    "glare": {"tilt": 0.35, "darkness": 0.30, "tremor": 0.10},
    "darkness": {"glare": 0.40, "tremor": 0.10, "tilt": 0.05},
    "tilt": {"tremor": 0.30, "glare": 0.10, "darkness": 0.05},
    "tremor": {"darkness": 0.15, "tilt": 0.05, "glare": 0.05},
}

# How strongly the scene remembers itself between shots. 1.0 = the scene never
# changes on its own; 0.0 = every shot is an independent draw (the i.i.d.
# corruption-filter assumption this package exists to reject).
DEFAULT_PERSISTENCE = 0.85

# Each shot tires the operator by this much (added to fatigue, capped at 1).
DEFAULT_FATIGUE_PER_SHOT = 0.08


@dataclass
class Operator:
    """The person taking the photo. Persists for the whole session.

    Attributes:
        skill: in [0, 1]. Drives both the probability an instruction is
            followed at all and how much of the problem gets fixed when it is.
        haste: in [0, 1]. Scales every side effect in COUPLING. A hasty
            operator fixes the named problem and creates two new ones.
        fatigue: in [0, 1]. Starts at 0 and rises with every shot taken;
            raises the tremor the scene relaxes toward, so retaking has a
            cost that is not just the capture budget.
    """

    skill: float = 0.6
    haste: float = 0.4
    fatigue: float = 0.0

    def compliance_probability(self) -> float:
        """Chance this operator acts on an instruction at all.

        Fatigue erodes compliance: by the fifth or sixth retake, a tired
        health worker starts just pressing the button again.
        """
        base = 0.45 + 0.5 * self.skill
        return float(np.clip(base * (1.0 - 0.4 * self.fatigue), 0.0, 1.0))

    def correction_strength(self, rng: np.random.Generator) -> float:
        """Fraction of the targeted factor removed, given compliance.

        Beta-distributed rather than fixed: even a skilled operator sometimes
        only half-fixes the problem. Mean rises with skill, variance falls.
        """
        mean = 0.25 + 0.55 * self.skill
        concentration = 3.0 + 7.0 * self.skill
        a = mean * concentration
        b = (1.0 - mean) * concentration
        return float(np.clip(rng.beta(a, b), 0.0, 1.0))

    def tire(self, amount: float = DEFAULT_FATIGUE_PER_SHOT) -> "Operator":
        return replace(self, fatigue=float(min(1.0, self.fatigue + amount)))


@dataclass
class SceneState:
    """Physical capture conditions, persisting across shots in one session.

    Attributes:
        factors: {factor_name: severity in [0, 1]} for every name in FACTORS.
        equilibrium: the value each factor relaxes toward between shots. This
            is what makes an untargeted retake nearly useless -- the room is
            still the room. A successful instruction moves the equilibrium,
            not just the current value, so a real fix persists.
        glare_azimuth: in [0, 1), where the specular hotspot sits relative to
            the tooth of interest (0 = directly on it). Glare only hides a
            lesion when the hotspot overlaps it, so intensity alone does not
            determine harm -- see `effective_glare`.
        jpeg: transport compression severity, constant for the session and not
            instructable.
    """

    factors: dict[str, float] = field(default_factory=dict)
    equilibrium: dict[str, float] = field(default_factory=dict)
    glare_azimuth: float = 0.5
    jpeg: float = 0.0

    def __post_init__(self) -> None:
        missing = [f for f in FACTORS if f not in self.factors]
        if missing:
            raise ValueError(f"SceneState is missing factors: {missing}")
        if not self.equilibrium:
            self.equilibrium = dict(self.factors)

    # -- derived quantities -------------------------------------------------

    def effective_glare(self) -> float:
        """Glare that actually lands on the tooth of interest.

        A bright reflection in the corner of the film is harmless; the same
        reflection over the lesion hides it. Overlap falls off with angular
        distance, so REDUCE_GLARE can help either by dimming the highlight or
        by moving it -- two physically different fixes with the same
        instruction, which is realistic and which a scalar-severity model
        cannot express.
        """
        # circular distance in [0, 0.5], rescaled so 0 -> full overlap
        d = abs(self.glare_azimuth)
        d = min(d, 1.0 - d)
        overlap = float(np.exp(-((d / 0.22) ** 2)))
        return float(self.factors["glare"] * overlap)

    def severities(self) -> dict[str, float]:
        """Per-degradation severities for the renderer, in degradation.py's names.

        Note glare is passed through `effective_glare` while the others map
        straight across.
        """
        out = {
            FACTOR_TO_DEGRADATION[f]: float(self.factors[f])
            for f in FACTORS
            if f != "glare"
        }
        out["glare"] = self.effective_glare()
        out["jpeg"] = float(self.jpeg)
        return out

    def usability(self) -> float:
        """Ground-truth "how usable is this shot", in [0, 1], 1 = pristine.

        Deliberately a soft-max-ish aggregate rather than a mean: one severe
        artifact ruins an image even when everything else is fine, which a
        mean would wash out.
        """
        sev = self.severities()
        worst = max(sev.values())
        mean = float(np.mean(list(sev.values())))
        return float(np.clip(1.0 - (0.75 * worst + 0.25 * mean), 0.0, 1.0))

    def copy(self) -> "SceneState":
        return SceneState(
            factors=dict(self.factors),
            equilibrium=dict(self.equilibrium),
            glare_azimuth=self.glare_azimuth,
            jpeg=self.jpeg,
        )


def sample_initial_scene(
    rng: np.random.Generator,
    difficulty: float = 0.5,
) -> SceneState:
    """Draw a starting scene.

    `difficulty` in [0, 1] shifts the whole clinic toward worse conditions --
    it is the knob that turns "well-lit district hospital" into "torch-lit
    rural outpost". Experiments sweep it to check conclusions are not an
    artifact of one operating regime.
    """
    # Beta shaped so difficulty moves the mode without ever pinning at 0 or 1.
    def draw(mode_lo: float, mode_hi: float) -> float:
        mean = mode_lo + (mode_hi - mode_lo) * difficulty
        conc = 6.0
        a = max(mean * conc, 0.15)
        b = max((1.0 - mean) * conc, 0.15)
        return float(np.clip(rng.beta(a, b), 0.0, 1.0))

    factors = {
        "glare": draw(0.10, 0.70),
        "tremor": draw(0.08, 0.55),
        "darkness": draw(0.10, 0.65),
        "tilt": draw(0.10, 0.50),
    }
    return SceneState(
        factors=factors,
        equilibrium=dict(factors),
        glare_azimuth=float(rng.random()),
        # transport compression: a fixed property of how the photo will be sent
        jpeg=float(np.clip(rng.beta(1.6, 5.0) * (0.4 + 0.6 * difficulty), 0.0, 1.0)),
    )


def sample_operator(rng: np.random.Generator, difficulty: float = 0.5) -> Operator:
    """Draw an operator. Harder settings mean less training and more hurry."""
    skill = float(np.clip(rng.beta(5.0, 3.0) * (1.0 - 0.45 * difficulty), 0.0, 1.0))
    haste = float(np.clip(rng.beta(3.0, 4.0) * (0.6 + 0.7 * difficulty), 0.0, 1.0))
    return Operator(skill=skill, haste=haste, fatigue=0.0)
