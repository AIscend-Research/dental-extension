"""Shared setup for every experiment: one anchored world, built the same way twice.

Every experiment starts from `build_world()`, which fixes the surrogate reader
at two stated anchors -- clean-image AUC and clinic AUC -- and fits the
calibrator on first shots only. Centralising it means no experiment can
quietly run in a different regime from its neighbours, and the anchors get
printed in every result header so the operating point is never implicit.

Note on the default burden. `BurdenSpec()`'s own default is the strict
clinical pairing (convict at clear-and-convincing, discharge at beyond
reasonable doubt). At that standard almost every case escalates -- which is a
real finding, and E3 reports it -- but it flattens the comparison between
policies. So experiments that compare *policies* state a looser headline
burden explicitly (`HEADLINE_BURDEN`), and E3 sweeps the whole ladder to show
where the flattening sets in. Nothing here changes the library default; the
burden is always passed in explicitly, so what a result was produced under is
readable from the experiment rather than inherited.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

from src.evidence.ladder import (
    BurdenSpec,
    CLEAR_AND_CONVINCING,
    PREPONDERANCE,
)
from src.bench.runner import fit_calibrator
from src.models.diagnostic import (
    SurrogateChannel,
    calibrate_loss_scale,
    calibrate_separation,
    clinic_auc,
)

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
FIGURES = REPO / "figures"

# The regime every experiment is anchored to.
CLEAN_AUC_TARGET = 0.88  # a good reader on a pristine radiograph
CLINIC_AUC_TARGET = 0.78  # the same reader on a median-condition phone photo
CLINIC_DIFFICULTY = 0.5
PREVALENCE = 0.35
N_STRATA = 4
CALIBRATION_N = 8000

# Policy comparisons run here: convict on the balance of probabilities (a
# conviction only means "refer for treatment"), discharge at the higher bar
# (sending someone home untreated is the expensive error in screening).
HEADLINE_BURDEN = BurdenSpec(convict=PREPONDERANCE, discharge=CLEAR_AND_CONVINCING)


class World:
    """An anchored reader plus the calibrator fitted to it."""

    def __init__(self, channel, calibrator, calibration_data, clean_auc, clinic_auc_value):
        self.channel = channel
        self.calibrator = calibrator
        self.calibration_data = calibration_data
        self.clean_auc = clean_auc
        self.clinic_auc = clinic_auc_value

    def header(self) -> str:
        edges = np.round(self.calibrator.edges, 3) if self.calibrator.edges is not None else None
        return (
            f"reader: clean AUC {self.clean_auc:.3f} (target {CLEAN_AUC_TARGET}), "
            f"clinic AUC {self.clinic_auc:.3f} (target {CLINIC_AUC_TARGET})\n"
            f"calibration: n={len(self.calibration_data)} first shots, "
            f"{N_STRATA} strata, edges {edges}, "
            f"fallback strata {sorted(self.calibrator.fallback_strata) or 'none'}"
        )


def build_world(
    head_noise: float = 0.12,
    clinic_difficulty: float = CLINIC_DIFFICULTY,
    clinic_auc_target: float = CLINIC_AUC_TARGET,
    calibrator_cls=None,
    n_strata: int = N_STRATA,
    calibration_n: int = CALIBRATION_N,
    seed: int = 7,
    config=None,
) -> World:
    """Anchor a surrogate reader and fit its calibrator.

    The two calibrations are order-dependent and both are needed: separation
    fixes the clean end, loss_scale fixes the degraded end. Fitting only the
    first leaves the reader near chance under realistic conditions; fitting
    only the second leaves its ceiling arbitrary.
    """
    from src.evidence.calibration import StratifiedCalibrator

    channel = SurrogateChannel(head_noise=head_noise)
    calibrate_separation(channel, CLEAN_AUC_TARGET, n=6000, seed=0)
    calibrate_loss_scale(
        channel, clinic_auc_target, clinic_difficulty=clinic_difficulty, n=4000, seed=1
    )
    measured_clinic = clinic_auc(channel, clinic_difficulty, n=6000, seed=2)

    calibrator, data = fit_calibrator(
        channel,
        n_strata=n_strata,
        calibrator_cls=calibrator_cls or StratifiedCalibrator,
        n=calibration_n,
        prevalence=PREVALENCE,
        clinic_difficulty=clinic_difficulty,
        seed=seed,
        config=config,
    )
    return World(channel, calibrator, data, CLEAN_AUC_TARGET, measured_clinic)


# ---------------------------------------------------------------------------
# output plumbing
# ---------------------------------------------------------------------------


def _jsonable(obj):
    if is_dataclass(obj):
        return _jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def save_results(name: str, payload: dict) -> Path:
    """Write one experiment's results to results/<name>.json."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    payload = dict(payload)
    payload.setdefault("generated_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    path.write_text(json.dumps(_jsonable(payload), indent=2))
    return path


def save_table(name: str, rows: list[dict]) -> Path:
    """Write a CSV alongside the JSON, for pasting into the paper."""
    import csv

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.csv"
    if not rows:
        path.write_text("")
        return path
    keys = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: _jsonable(r.get(k)) for k in keys})
    return path


def banner(title: str, world: World | None = None) -> None:
    print("=" * 78)
    print(title)
    if world is not None:
        print(world.header())
    print("=" * 78, flush=True)


def figure_path(name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    return FIGURES / name
