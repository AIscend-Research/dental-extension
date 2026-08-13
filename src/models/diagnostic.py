"""The two channels a capture is read through, and the backends that implement them.

Every capture is read twice:

  * the DIAGNOSIS channel Y_t -- a caries score for the tooth in question;
  * the DEGRADATION channel D_t -- predicted per-artifact severities and a
    scalar usability, i.e. the confidence head of src/models/confidence_head.py.

Keeping them separate is not tidiness, it is the load-bearing structure of the
whole framework: the validity theorem holds precisely because betting and
stopping read only D_t, never Y_t (see src/evidence/view.py).

Two backends implement the pair behind one interface:

`SurrogateChannel` is an analytic noisy-observation model. It exists so the
validity experiments can run hundreds of thousands of sessions, and so the
ground truth about signal loss is known exactly rather than estimated. It is
not a stand-in for a detector that we failed to train -- a Monte Carlo study of
a theorem's coverage *wants* a channel whose null distribution is controlled.

`RealImageChannel` (src/models/real_channel.py) reads actual DENTEX tooth
crops rendered through the simulator, and is what checks the framework
survives contact with real images and a real, imperfect classifier.

The surrogate's two degradation effects are both deliberate:
  - signal attenuation: severity shrinks the true class separation;
  - noise inflation: severity widens the observation noise.
Attenuation alone would make bad photos merely uninformative (scores drift to
0.5, which is harmless and easy to detect). Inflation is what makes them
*confidently wrong*, which is the failure mode the entire paper is about. A
simulator with only the first effect would flatter every method under test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.data.degradation import DEGRADATION_NAMES

# How much each artifact destroys diagnostic signal, per unit severity.
#
# Ordering is the defensible part; the exact values are a modelling choice and
# are swept in the sensitivity experiment. Blur and compression are worst
# because caries on a radiograph is a fine-grained density change, and both
# destroy high-frequency detail outright. Glare occludes wherever it lands.
# Low light costs contrast, recoverable up to a point. Off-axis geometry is
# the mildest -- it distorts, but it does not erase.
SIGNAL_LOSS_WEIGHTS: dict[str, float] = {
    "blur": 0.90,
    "jpeg": 0.75,
    "glare": 0.80,
    "low_light": 0.55,
    "angle": 0.30,
}


@dataclass
class Reading:
    """One capture, read through both channels.

    Attributes:
        score: diagnosis channel Y_t, in [0, 1]. Higher = more suspicious.
        degradation: degradation channel D_t, predicted severity per artifact.
        usability: D_t's scalar summary, in [0, 1]. 1 = trust this photo.
        true_quality: the simulator's actual retained-signal fraction. Present
            for analysis only -- no policy may read it, and the benchmark
            never passes it into one.
    """

    score: float
    degradation: dict[str, float]
    usability: float
    true_quality: float = float("nan")


@dataclass
class Case:
    """The thing being adjudicated: one tooth, with a fixed ground truth.

    Attributes:
        label: 1 = caries present, 0 = sound. Constant for the whole session --
            retaking photographs changes the evidence, never the truth.
        difficulty: in [0, 1]. How subtle this lesion is even in a perfect
            photo; 1 means no amount of retaking will resolve it. This is what
            keeps the escalation rate from being an artifact of image quality
            alone -- some cases are genuinely undecidable from a photograph.
        payload: optional backend-specific data (e.g. a real image crop).
    """

    label: int
    difficulty: float = 0.3
    payload: object = None


class DiagnosticChannel:
    """Interface: read one capture into a Reading."""

    name = "base"

    def read(self, case: Case, severities: dict[str, float], rng: np.random.Generator) -> Reading:
        raise NotImplementedError


@dataclass
class SurrogateChannel(DiagnosticChannel):
    """Analytic noisy-observation model of a caries reader.

    The generative form, for a case with label y and difficulty d observed at
    retained-quality q:

        z ~ Normal( separation * (2y - 1) * (1 - d) * q**gamma,
                    sigma_base + sigma_degraded * (1 - q) )
        score = sigmoid(z)

    `separation` is fixed by `calibrate_separation` so that the clean-image
    AUC lands on a target, which is how the surrogate is tied to reality: the
    target is set from reported smartphone/photographic caries-detection
    performance rather than chosen to make the method look good. See
    docs/simulator_grounding.md.
    """

    separation: float = 2.6
    gamma: float = 1.0
    sigma_base: float = 1.0
    sigma_degraded: float = 0.7
    #: global multiplier on SIGNAL_LOSS_WEIGHTS, fitted by
    #: `calibrate_loss_scale` so that typical clinic conditions land on a
    #: stated degraded-AUC anchor. Without it the five artifacts compound
    #: multiplicatively into a reader that is barely better than chance at
    #: median conditions, which would make every method look equally bad.
    loss_scale: float = 1.0
    # noise on the confidence head's severity estimates; 0.0 = oracle head
    head_noise: float = 0.12
    head_bias: float = 0.0
    name: str = "surrogate"

    def quality(self, severities: dict[str, float]) -> float:
        """Fraction of diagnostic signal surviving this set of artifacts.

        Multiplicative across artifacts: two independent ways of destroying
        the same fine detail compound rather than add, and the product cannot
        leave [0, 1] however many artifacts stack.
        """
        q = 1.0
        for name, sev in severities.items():
            w = SIGNAL_LOSS_WEIGHTS.get(name, 0.5) * self.loss_scale
            q *= 1.0 - float(np.clip(w, 0.0, 1.0)) * float(np.clip(sev, 0.0, 1.0))
        return float(np.clip(q, 0.0, 1.0))

    def read(self, case: Case, severities: dict[str, float], rng: np.random.Generator) -> Reading:
        q = self.quality(severities)
        mu = self.separation * (2.0 * case.label - 1.0) * (1.0 - case.difficulty) * (q**self.gamma)
        sigma = self.sigma_base + self.sigma_degraded * (1.0 - q)
        z = rng.normal(mu, sigma)
        score = 1.0 / (1.0 + np.exp(-z))

        pred = self._read_degradation(severities, rng)
        return Reading(
            score=float(score),
            degradation=pred,
            usability=predicted_usability(pred),
            true_quality=q,
        )

    def _read_degradation(
        self, severities: dict[str, float], rng: np.random.Generator
    ) -> dict[str, float]:
        """The confidence head, modelled as a noisy but unbiased-ish estimator.

        A perfect head would make the framework look better than it is: the
        stratification, the stakes and the retake instructions all key off
        this channel, so its errors propagate everywhere. `head_noise` is
        therefore a first-class experimental knob, not a nuisance parameter.
        """
        out = {}
        for name in DEGRADATION_NAMES:
            true = float(severities.get(name, 0.0))
            est = true + self.head_bias + rng.normal(0.0, self.head_noise)
            out[name] = float(np.clip(est, 0.0, 1.0))
        return out


def predicted_usability(degradation: dict[str, float]) -> float:
    """Scalar usability from predicted severities.

    Same aggregate as `SceneState.usability` (worst-artifact dominated), so
    the predicted and true usabilities are on one scale and their gap is
    purely the head's error rather than a definitional mismatch.
    """
    if not degradation:
        return 1.0
    values = list(degradation.values())
    worst = max(values)
    mean = float(np.mean(values))
    return float(np.clip(1.0 - (0.75 * worst + 0.25 * mean), 0.0, 1.0))


def calibrate_separation(
    channel: SurrogateChannel,
    target_auc: float = 0.88,
    difficulty: float = 0.3,
    n: int = 20000,
    seed: int = 0,
    tol: float = 0.002,
) -> float:
    """Solve for the `separation` giving `target_auc` on clean, artifact-free images.

    Anchoring the surrogate to a stated clean-image AUC is what stops the
    experiments from being a story about an arbitrarily strong or weak model.
    Returns the fitted value and sets it on `channel` in place.

    Uses bisection on separation, which is monotone in AUC.
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, n)
    clean: dict[str, float] = {}

    def auc_for(sep: float) -> float:
        saved = channel.separation
        channel.separation = sep
        r = np.array([channel.read(Case(int(y), difficulty), clean, rng).score for y in labels])
        channel.separation = saved
        return _auc(labels, r)

    lo, hi = 0.01, 12.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        a = auc_for(mid)
        if abs(a - target_auc) < tol:
            lo = hi = mid
            break
        if a < target_auc:
            lo = mid
        else:
            hi = mid
    channel.separation = float(0.5 * (lo + hi))
    return channel.separation


def clinic_auc(
    channel: "SurrogateChannel",
    clinic_difficulty: float = 0.5,
    n: int = 6000,
    seed: int = 1,
    difficulty_alpha: float = 2.0,
    difficulty_beta: float = 4.0,
) -> float:
    """AUC of first shots taken under real clinic conditions.

    This is the second anchor. `calibrate_separation` fixes what the reader
    can do on a pristine image; this measures what survives the room. Both
    numbers are reported in every experiment's header so a reader can see the
    operating regime the results were produced in rather than inferring it.
    """
    from src.sim.session import CaptureSession

    rng = np.random.default_rng(seed)
    labels = np.empty(n, dtype=int)
    scores = np.empty(n)
    for i in range(n):
        y = int(rng.integers(0, 2))
        case = Case(y, float(rng.beta(difficulty_alpha, difficulty_beta)))
        session = CaptureSession(rng=rng, difficulty=clinic_difficulty)
        capture = session.capture()
        labels[i] = y
        scores[i] = channel.read(case, capture.severities, rng).score
    return _auc(labels, scores)


def calibrate_loss_scale(
    channel: "SurrogateChannel",
    target_clinic_auc: float = 0.78,
    clinic_difficulty: float = 0.5,
    n: int = 4000,
    seed: int = 1,
    tol: float = 0.004,
) -> float:
    """Solve for the `loss_scale` giving `target_clinic_auc` at median conditions.

    Anchoring the degraded end matters as much as anchoring the clean end. The
    five artifact primitives compose multiplicatively, so plausible per-artifact
    weights stack into an implausible whole -- unconstrained, the default
    weights drive a clean-AUC-0.88 reader down to roughly 0.57 under median
    conditions, i.e. near-useless. Any comparison run in that regime is
    measuring noise, not policy.

    The target is the modelling assumption that smartphone capture conditions
    cost a reader real accuracy without destroying it; see
    docs/simulator_grounding.md for what it is based on and how sensitive the
    conclusions are to it (the sensitivity experiment sweeps it directly).

    Monotone decreasing in loss_scale, so bisection is safe. Sets the value on
    `channel` in place and returns it.
    """
    lo, hi = 0.05, 1.5
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        channel.loss_scale = mid
        a = clinic_auc(channel, clinic_difficulty, n=n, seed=seed)
        if abs(a - target_clinic_auc) < tol:
            break
        if a < target_clinic_auc:
            hi = mid  # too much loss -> shrink the scale
        else:
            lo = mid
    channel.loss_scale = float(0.5 * (lo + hi))
    return channel.loss_scale


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney), ties averaged. No sklearn dependency."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks within ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    r_pos = ranks[labels == 1].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))
