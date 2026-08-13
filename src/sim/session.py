"""The capture session: a stochastic process over shots, driven by instructions.

This is the object that replaces "apply a random corruption to each image".
One `CaptureSession` is one patient, one film, one person with a phone. Calling
`.capture()` takes a shot under the current latent state; calling
`.instruct(instruction)` asks the operator to change something, which updates
the latent state *causally* -- partially, noisily, and with side effects.

Why this matters for the theory: the sequence of captures in a session is not
exchangeable. Shot 2 exists only because shot 1 was bad, and shot 2's
conditions are correlated with shot 1's (same room, same hands) *and* shifted
by the instruction that shot 1 triggered. Any method that treats the shots as
i.i.d. draws -- or that assumes exchangeability, as standard conformal
prediction does -- is making a false assumption here. `src/evidence/` is built
to be valid without that assumption; `experiments/e1_simulator.py` measures
just how badly it is violated.

Dynamics, per step:

  1. Fatigue rises (every shot tires the operator).
  2. If an instruction was issued, roll compliance. On compliance, the target
     factor and its equilibrium both drop by a Beta-distributed fraction, and
     each side effect in COUPLING is added to the other factors, scaled by the
     operator's haste.
  3. All factors relax toward their equilibrium with persistence rho, plus
     Gaussian process noise. The tremor equilibrium is additionally pushed up
     by accumulated fatigue.
  4. Glare azimuth drifts; a complied REDUCE_GLARE may instead *move* the
     hotspot rather than dim it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.sim.instructions import Instruction, RETAKE_ANY
from src.sim.state import (
    COUPLING,
    DEFAULT_FATIGUE_PER_SHOT,
    DEFAULT_PERSISTENCE,
    FACTORS,
    Operator,
    SceneState,
    sample_initial_scene,
    sample_operator,
)


@dataclass
class SessionConfig:
    """Knobs for the capture process. Swept in the sensitivity experiment."""

    persistence: float = DEFAULT_PERSISTENCE
    process_noise: float = 0.05
    fatigue_per_shot: float = DEFAULT_FATIGUE_PER_SHOT
    # scales COUPLING globally; 0.0 turns off side effects entirely, which is
    # the "instructions are free" ablation
    coupling_scale: float = 1.0
    # chance a complied REDUCE_GLARE moves the hotspot instead of dimming it
    glare_reposition_prob: float = 0.45
    # passive drift of the glare hotspot per shot
    glare_azimuth_drift: float = 0.04
    #: Draw a completely fresh scene for every shot, discarding the latent
    #: state entirely. This is the i.i.d. corruption-filter control -- the
    #: world that ImageNet-C-style benchmarks implicitly assume, where every
    #: retake is an independent sample from the corruption distribution.
    #:
    #: Note that `persistence = 0.0` does NOT achieve this: the per-session
    #: *equilibrium* survives, so the scene still relaxes toward the same room
    #: every shot and the session stays correlated. Only resampling the scene
    #: outright removes the memory. Conflating the two silently turns the
    #: i.i.d. ablation into a no-op that appears to show persistence does not
    #: matter.
    iid_scene_per_shot: bool = False


@dataclass
class Capture:
    """One shot: what the world was, and what the shot cost.

    `state` is a snapshot taken at capture time, so a session's history is a
    list of independent states rather than aliases of one mutating object.
    """

    index: int
    state: SceneState
    operator_fatigue: float
    instruction: Instruction | None
    complied: bool
    # ground-truth per-degradation severities actually rendered
    severities: dict[str, float] = field(default_factory=dict)

    @property
    def usability(self) -> float:
        return self.state.usability()


class CaptureSession:
    """A single evidential-capture session over one patient's film.

    Usage:
        s = CaptureSession(rng=np.random.default_rng(0), difficulty=0.5)
        first = s.capture()                     # shot 1
        second = s.capture(REDUCE_GLARE)        # shot 2, after an instruction

    `capture(instruction)` advances the latent state in response to
    `instruction` and then takes the shot, so the instruction issued after
    shot t affects shot t+1 -- never shot t itself.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        difficulty: float = 0.5,
        config: SessionConfig | None = None,
        scene: SceneState | None = None,
        operator: Operator | None = None,
    ):
        self.rng = rng
        self.config = config or SessionConfig()
        self.difficulty = difficulty
        self.scene = scene.copy() if scene is not None else sample_initial_scene(rng, difficulty)
        self.operator = operator if operator is not None else sample_operator(rng, difficulty)
        self.history: list[Capture] = []

    # -- dynamics -----------------------------------------------------------

    def _apply_instruction(self, instruction: Instruction) -> bool:
        """Mutate the scene in response to `instruction`. Returns whether complied.

        An untargeted RETAKE_ANY never changes the scene: nothing was asked
        for, so nothing gets fixed. The only thing it buys is a fresh draw of
        process noise, which in a persistent process is very little. That is
        the control arm.
        """
        if instruction is None or instruction.target is None:
            return False
        if self.rng.random() > self.operator.compliance_probability():
            return False

        target = instruction.target
        strength = self.operator.correction_strength(self.rng)

        if target == "glare" and self.rng.random() < self.config.glare_reposition_prob:
            # the operator turns the film rather than dimming the light: the
            # highlight is just as bright, but it moves off the tooth
            shift = 0.15 + 0.35 * strength
            self.scene.glare_azimuth = float(
                (self.scene.glare_azimuth + shift * self.rng.choice([-1.0, 1.0])) % 1.0
            )
            magnitude = strength * 0.5  # side effects still apply, halved
        else:
            before = self.scene.factors[target]
            after = before * (1.0 - strength)
            self.scene.factors[target] = float(after)
            # a real fix moves the equilibrium too, so it does not simply
            # relax back next shot
            self.scene.equilibrium[target] = float(
                min(self.scene.equilibrium[target], after + 0.25 * (before - after))
            )
            magnitude = before - after

        # side effects: fixing one thing disturbs the others
        for affected, leak in COUPLING.get(target, {}).items():
            if affected == target:
                continue
            bump = leak * self.config.coupling_scale * self.operator.haste * magnitude
            self.scene.factors[affected] = float(
                np.clip(self.scene.factors[affected] + bump, 0.0, 1.0)
            )
            self.scene.equilibrium[affected] = float(
                np.clip(self.scene.equilibrium[affected] + 0.5 * bump, 0.0, 1.0)
            )
        return True

    def _relax(self) -> None:
        """Persistence step: factors drift toward equilibrium, plus noise."""
        cfg = self.config
        if cfg.iid_scene_per_shot:
            # the corruption-filter world: nothing is remembered, so an
            # instruction cannot help and a retake is simply another draw
            fresh = sample_initial_scene(self.rng, self.difficulty)
            self.scene = fresh
            return
        rho = cfg.persistence
        # fatigue raises the tremor the hand relaxes toward
        fatigue_push = 0.25 * self.operator.fatigue
        for f in FACTORS:
            eq = self.scene.equilibrium[f]
            if f == "tremor":
                eq = min(1.0, eq + fatigue_push)
            noise = self.rng.normal(0.0, cfg.process_noise)
            value = rho * self.scene.factors[f] + (1.0 - rho) * eq + noise
            self.scene.factors[f] = float(np.clip(value, 0.0, 1.0))
        self.scene.glare_azimuth = float(
            (self.scene.glare_azimuth + self.rng.normal(0.0, cfg.glare_azimuth_drift)) % 1.0
        )

    # -- public API ---------------------------------------------------------

    def capture(self, instruction: Instruction | None = None) -> Capture:
        """Advance the world in response to `instruction`, then take one shot."""
        complied = False
        if self.history:  # the first shot happens before anyone can be instructed
            self.operator = self.operator.tire(self.config.fatigue_per_shot)
            if instruction is not None and instruction is not RETAKE_ANY:
                complied = self._apply_instruction(instruction)
            self._relax()
        elif instruction is not None:
            raise ValueError(
                "the first capture cannot follow an instruction -- there is "
                "nothing yet to give feedback on"
            )

        shot = Capture(
            index=len(self.history),
            state=self.scene.copy(),
            operator_fatigue=self.operator.fatigue,
            instruction=instruction,
            complied=complied,
            severities=self.scene.severities(),
        )
        self.history.append(shot)
        return shot

    @property
    def n_captures(self) -> int:
        return len(self.history)
