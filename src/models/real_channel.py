"""Both channels, learned from real DENTEX tooth crops.

This is the same two-channel interface as `SurrogateChannel`, backed by real
images and two real fitted models:

  * the DIAGNOSIS channel is a logistic regression over hand-designed image
    features, trained on tooth crops rendered through the capture simulator;
  * the DEGRADATION channel is a ridge regression predicting the per-artifact
    severity vector from the same features -- the confidence head of
    src/models/confidence_head.py, trained by weak supervision on the
    simulator's own severity labels, exactly as the project's Phase 3 design
    intended.

Why hand-designed features and a linear model rather than a CNN. The public
DENTEX split with diagnosis labels contains 182 annotated teeth across 50
radiographs. A CNN on that would report whatever the split noise felt like
that day, and the point of this arm is not to set an accuracy record -- it is
to check that a *real* model on *real* radiographs degrades under rendered
capture artifacts the way the surrogate assumes it does. That is a relative
within-model comparison, and it is far better served by an estimator whose
variance is small enough for the comparison to mean something.

The features are chosen so that both channels are learnable from them:
intensity statistics and histogram (lighting and glare), frequency-band energy
and Laplacian variance (blur and compression), gradient statistics and
saturation fraction (glare and contrast), plus a coarse spatial intensity grid
carrying the actual radiographic content the diagnosis needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from src.data.degradation import DEGRADATION_NAMES
from src.models.diagnostic import Case, DiagnosticChannel, Reading, predicted_usability
from src.sim.render import render_severities


def extract_features(img_bgr: np.ndarray) -> np.ndarray:
    """Feature vector for one crop. Deterministic, no fitting involved."""
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    feats: list[float] = []

    # intensity statistics + histogram: lighting level, glare wash-out
    feats += [float(g.mean()), float(g.std()), float(np.median(g))]
    feats += [float(np.quantile(g, q)) for q in (0.05, 0.25, 0.75, 0.95)]
    hist, _ = np.histogram(g, bins=16, range=(0.0, 1.0), density=True)
    feats += hist.tolist()
    # fraction of near-saturated pixels: the direct specular-highlight cue
    feats += [float((g > 0.92).mean()), float((g < 0.08).mean())]

    # sharpness / frequency content: blur and JPEG both destroy this
    lap = cv2.Laplacian(g, cv2.CV_32F)
    feats += [float(lap.var()), float(np.abs(lap).mean())]
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    feats += [float(mag.mean()), float(mag.std()), float(np.quantile(mag, 0.9))]

    # radial frequency-band energy from the 2-D FFT: separates "blurred" from
    # "compressed" better than a single sharpness number, since JPEG removes
    # high frequencies in blocks while defocus removes them smoothly
    spec = np.abs(np.fft.fftshift(np.fft.fft2(g)))
    h, w = spec.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / max(cy, cx)
    total = float(spec.sum()) + 1e-8
    for lo, hi in ((0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.0)):
        band = spec[(r >= lo) & (r < hi)]
        feats.append(float(band.sum()) / total)

    # coarse spatial grid: the radiographic content itself
    small = cv2.resize(g, (6, 6), interpolation=cv2.INTER_AREA)
    feats += small.ravel().tolist()

    return np.asarray(feats, dtype=np.float64)


@dataclass
class _Ridge:
    """Least-squares with L2 penalty. Closed form, no sklearn dependency."""

    alpha: float = 1.0
    coef_: np.ndarray | None = None
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "_Ridge":
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0) + 1e-8
        Xs = np.hstack([(X - self.mean_) / self.scale_, np.ones((len(X), 1))])
        d = Xs.shape[1]
        penalty = self.alpha * np.eye(d)
        penalty[-1, -1] = 0.0  # never penalise the intercept
        self.coef_ = np.linalg.solve(Xs.T @ Xs + penalty, Xs.T @ Y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = np.hstack([(X - self.mean_) / self.scale_, np.ones((len(X), 1))])
        return Xs @ self.coef_


@dataclass
class _LogisticRegression:
    """Binary logistic regression by gradient descent with L2 penalty.

    Written out rather than imported so this module has the same dependency
    footprint as the rest of `src/` (numpy + opencv) and can run in the core
    environment with no torch and no sklearn.
    """

    alpha: float = 1.0
    lr: float = 0.5
    n_steps: int = 800
    coef_: np.ndarray | None = None
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_LogisticRegression":
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0) + 1e-8
        Xs = np.hstack([(X - self.mean_) / self.scale_, np.ones((len(X), 1))])
        y = np.asarray(y, dtype=np.float64)
        w = np.zeros(Xs.shape[1])
        n = len(Xs)
        # class weights, because the crop pool is roughly 73/27
        pos_w = 0.5 / max(y.mean(), 1e-6)
        neg_w = 0.5 / max(1.0 - y.mean(), 1e-6)
        sample_w = np.where(y > 0.5, pos_w, neg_w)
        for _ in range(self.n_steps):
            p = 1.0 / (1.0 + np.exp(-Xs @ w))
            grad = Xs.T @ (sample_w * (p - y)) / n
            grad[:-1] += self.alpha * w[:-1] / n
            w -= self.lr * grad
        self.coef_ = w
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = np.hstack([(X - self.mean_) / self.scale_, np.ones((len(X), 1))])
        return 1.0 / (1.0 + np.exp(-Xs @ self.coef_))


@dataclass
class RealImageChannel(DiagnosticChannel):
    """Reads real DENTEX crops rendered through the capture simulator.

    Fit with `train_real_channel`. `read` renders `case.payload` (a ToothCrop's
    image) at the shot's severities, extracts features once, and runs both
    heads on them -- so the two channels genuinely see the same photograph,
    which is the situation the framework is about.
    """

    diagnosis: _LogisticRegression = field(default_factory=_LogisticRegression)
    degradation: _Ridge = field(default_factory=_Ridge)
    name: str = "real_dentex"

    def read(self, case: Case, severities: dict[str, float], rng: np.random.Generator) -> Reading:
        if case.payload is None:
            raise ValueError("RealImageChannel needs a ToothCrop image on Case.payload")
        rendered = render_severities(case.payload, severities, rng=rng)
        feats = extract_features(rendered.image)[None, :]

        score = float(self.diagnosis.predict_proba(feats)[0])
        pred = np.clip(self.degradation.predict(feats)[0], 0.0, 1.0)
        degradation = {name: float(pred[i]) for i, name in enumerate(DEGRADATION_NAMES)}
        return Reading(
            score=score,
            degradation=degradation,
            usability=predicted_usability(degradation),
            true_quality=float(np.clip(1.0 - max(severities.values(), default=0.0), 0.0, 1.0)),
        )


def train_real_channel(
    crops,
    rng: np.random.Generator,
    n_renders: int = 12,
    clinic_difficulty: float = 0.5,
    diagnosis_alpha: float = 3.0,
    degradation_alpha: float = 3.0,
) -> RealImageChannel:
    """Fit both heads on rendered versions of the training crops.

    Each crop is rendered `n_renders` times under scene states drawn from the
    simulator, so the reader is trained on the same kind of degraded imagery it
    will be evaluated on, and the degradation head gets its weak-supervision
    targets for free from the renderer.

    Training on degraded renders rather than clean crops is the "robustness
    variant" the project roadmap calls for; the clean-only comparison is run in
    the experiment as a separate arm.
    """
    from src.sim.session import CaptureSession

    X, y, S = [], [], []
    for crop in crops:
        for _ in range(n_renders):
            session = CaptureSession(rng=rng, difficulty=clinic_difficulty)
            capture = session.capture()
            severities = capture.severities
            rendered = render_severities(crop.image, severities, rng=rng)
            X.append(extract_features(rendered.image))
            y.append(crop.label)
            S.append([severities.get(n, 0.0) for n in DEGRADATION_NAMES])

    X = np.asarray(X)
    y = np.asarray(y)
    S = np.asarray(S)
    return RealImageChannel(
        diagnosis=_LogisticRegression(alpha=diagnosis_alpha).fit(X, y),
        degradation=_Ridge(alpha=degradation_alpha).fit(X, S),
    )
