"""Executing a docket: calibration, then adjudication.

Calibration protocol, and why it is the realistic one: the calibration set is
built from **first shots only**. In deployment you calibrate on a labelled
archive of photographs that were taken once and kept -- nobody ships you a
corpus of retake loops. Test time then involves later shots, whose latent
conditions were selected by the retake loop itself.

That mismatch is not an oversight in the protocol, it *is* the phenomenon under
study: at test time the calibrator is applied to a population of shots the
retake loop has selected, not the population it was fitted on. Building the
calibration set the convenient way -- sampling shots from inside retake loops
-- would hide the effect entirely, so this module does not offer that option.

What the shift does to coverage is measured, not assumed. In this capture
process it runs conservative (E2), because retakes yield cleaner shots and a
sound tooth in a clean shot is compared against a pool of dirtier ones. So the
protocol is safe here; see src/evidence/calibration.py for why that is a
property of this process rather than something to rely on in general.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.bench.docket import Docket, to_model_case
from src.bench.policies import CapturePolicy, PolicyContext, SessionResult
from src.evidence.calibration import Calibrator, StratifiedCalibrator
from src.models.diagnostic import Case, DiagnosticChannel
from src.sim.session import CaptureSession, SessionConfig


@dataclass
class CalibrationData:
    """Raw calibration observations, kept so ablations can refit differently."""

    scores: np.ndarray
    labels: np.ndarray
    usabilities: np.ndarray
    # Set only by collect_calibration(). fit_calibrator() checks this on any
    # externally-supplied `data=` so a caller can't hand it session/retake-
    # loop data just because that happens to have the same three array
    # fields -- doing so would reintroduce exactly the selection-bias
    # mismatch stratification exists to prevent (see this module's
    # docstring), silently rather than by a caller's deliberate choice.
    _first_shots_only: bool = field(default=False, repr=False, compare=False)

    def __len__(self) -> int:
        return len(self.scores)


def collect_calibration(
    channel: DiagnosticChannel,
    n: int = 4000,
    prevalence: float = 0.35,
    clinic_difficulty: float = 0.5,
    n_strata: int = 4,
    seed: int = 12345,
    config: SessionConfig | None = None,
    difficulty_alpha: float = 2.0,
    difficulty_beta: float = 4.0,
    image_pool: list | None = None,
) -> CalibrationData:
    """Sample labelled first shots under the same clinic conditions.

    Uses a seed disjoint from any docket's, so calibration and evaluation
    never share a session.
    """
    rng = np.random.default_rng(seed)
    scores = np.empty(n)
    labels = np.empty(n, dtype=int)
    usabilities = np.empty(n)

    for i in range(n):
        label = int(rng.random() < prevalence)
        difficulty = float(rng.beta(difficulty_alpha, difficulty_beta))
        payload = None
        if image_pool:
            payload = image_pool[int(rng.integers(0, len(image_pool)))]
        case = Case(label=label, difficulty=difficulty, payload=payload)
        session = CaptureSession(rng=rng, difficulty=clinic_difficulty, config=config)
        capture = session.capture()  # first shot only -- see module docstring
        reading = channel.read(case, capture.severities, rng)
        scores[i] = reading.score
        labels[i] = label
        usabilities[i] = reading.usability

    return CalibrationData(
        scores=scores, labels=labels, usabilities=usabilities, _first_shots_only=True
    )


def fit_calibrator(
    channel: DiagnosticChannel,
    n_strata: int = 4,
    calibrator_cls: type[Calibrator] = StratifiedCalibrator,
    data: CalibrationData | None = None,
    **kwargs,
) -> tuple[Calibrator, CalibrationData]:
    """Collect calibration data (unless given) and fit a calibrator to it."""
    if data is None:
        data = collect_calibration(channel, n_strata=n_strata, **kwargs)
    elif not data._first_shots_only:
        raise ValueError(
            "fit_calibrator(data=...) requires CalibrationData produced by "
            "collect_calibration() (first shots only). Calibrating against "
            "anything else -- e.g. capture data pulled from inside a retake "
            "loop -- reintroduces exactly the selection-bias mismatch "
            "stratification exists to prevent; see this module's docstring. "
            "If this data really is first-shots-only, build it via "
            "collect_calibration() rather than constructing CalibrationData "
            "directly."
        )
    cal = calibrator_cls(n_strata=n_strata).fit(data.scores, data.labels, data.usabilities)
    return cal, data


def run_docket(
    docket: Docket,
    policy: CapturePolicy,
    channel: DiagnosticChannel,
    calibrator: Calibrator,
    n_strata: int = 4,
    config: SessionConfig | None = None,
    image_pool: list | None = None,
    progress_every: int | None = None,
) -> list[SessionResult]:
    """Run `policy` over every case on `docket`.

    Each case gets its own Generator seeded from the docket, so a case's
    session is reproducible independently of how many cases ran before it --
    and so two policies see the *same starting scene and operator* for the
    same case, which is what makes the arms comparable. They diverge only
    once they start issuing different instructions, which is the intended
    difference.
    """
    results: list[SessionResult] = []
    for i, dc in enumerate(docket.cases):
        rng = np.random.default_rng(dc.seed)
        payload = image_pool[dc.image_index] if (image_pool and dc.image_index is not None) else None
        ctx = PolicyContext(
            session=CaptureSession(rng=rng, difficulty=docket.clinic_difficulty, config=config),
            case=to_model_case(dc, payload=payload),
            channel=channel,
            calibrator=calibrator,
            burden=docket.burden,
            budget=docket.budget,
            rng=rng,
            n_strata=n_strata,
        )
        results.append(policy.run(ctx, dc.case_id))
        if progress_every and (i + 1) % progress_every == 0:
            print(f"    {policy.name}: {i + 1}/{len(docket)} cases", flush=True)
    return results
