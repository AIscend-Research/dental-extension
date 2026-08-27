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
import math
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
# real tooth crops, for figures that want a photograph rather than a bar
# ---------------------------------------------------------------------------

_TOOTH_CROPS_CACHE: list | None = None


def sample_tooth_crops(n: int = 1, label: int | None = None, seed: int = 0) -> list:
    """A few real DENTEX tooth crops, for dressing a results figure with an
    actual photograph instead of only bars. Loads the (small) validation
    split once and caches it -- this is image I/O, not model fitting, so it
    is cheap enough to call from inside a `make_figure`.

    Args:
        label: 1 = caries, 0 = sound, None = either.
    """
    global _TOOTH_CROPS_CACHE
    from src.data.dentex_crops import load_tooth_crops

    if _TOOTH_CROPS_CACHE is None:
        _TOOTH_CROPS_CACHE = load_tooth_crops(task="caries_vs_other", crop_size=220, context=0.75)

    pool = _TOOTH_CROPS_CACHE if label is None else [c for c in _TOOTH_CROPS_CACHE if c.label == label]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    return [pool[i] for i in idx]


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


_ANNOTATE_CANDIDATE_OFFSETS = [
    (r * math.cos(math.radians(a)), r * math.sin(math.radians(a)))
    for r in (10, 20, 34, 50)
    for a in (35, -35, 145, -145, 90, -90, 0, 180)
]


def annotate_no_overlap(ax, xs, ys, labels, fontsize=7, avoid=()) -> None:
    """Label each (x, y) scatter point without the text smearing together.

    A fixed offset (or a short cycle of fixed offsets) only separates labels
    when the points themselves are already spread out. Several of the
    leaderboard scatters here have policy arms that land almost on top of
    each other (e.g. evidential_capture/one_step_lookahead/oracle_instruction
    are within a few thousandths of a VPC of one another), and a fixed offset
    there just moves the collision from the markers to the text. This tries
    an expanding ring of candidate offsets per label and keeps the first one
    whose *rendered* bounding box doesn't overlap anything already placed
    (other labels, the axes title, plus anything extra in `avoid`), drawing a
    thin leader line back to the point so a label is never ambiguous about
    which marker it names, even for the smallest offset.

    The axes title is always in the collision set (not just when a caller
    remembers to pass it): any panel here can end up with a data point near
    the top of its axes on some future rerun (new arm, different seed), and
    checking one extra bbox is free -- e13's two-line title collided with a
    label before this was made unconditional; nothing about that scenario is
    specific to e13's panel.

    `avoid` is for anything ELSE a caller wants labels to clear (e.g. a
    legend box) beyond the title, which is covered automatically.
    """
    fig = ax.figure
    fig.canvas.draw()  # populate a renderer so text extents are real, not guessed
    renderer = fig.canvas.get_renderer()

    placed = [ax.title.get_window_extent(renderer=renderer)]
    placed += [a.get_window_extent(renderer=renderer) for a in avoid]

    for x, y, label in zip(xs, ys, labels):
        t = bbox = dx = dy = None
        for i, (dx, dy) in enumerate(_ANNOTATE_CANDIDATE_OFFSETS):
            t = ax.annotate(
                label, (x, y), xytext=(dx, dy), textcoords="offset points",
                fontsize=fontsize, zorder=5,
            )
            bbox = t.get_window_extent(renderer=renderer)
            collides = any(bbox.overlaps(b) for b in placed)
            is_last = i == len(_ANNOTATE_CANDIDATE_OFFSETS) - 1
            if not collides or is_last:
                # last candidate is kept even if it still collides, rather
                # than silently dropping the label.
                break
            t.remove()
        placed.append(bbox)
        line = ax.annotate(
            "", xy=(x, y), xytext=(dx, dy), textcoords="offset points", xycoords="data",
            arrowprops=dict(arrowstyle="-", lw=0.5, color="grey", shrinkA=0, shrinkB=3),
        )
        # The leader line's own extent goes in `placed` too, not just the
        # label text -- otherwise a later label in this same call can be
        # placed right across an earlier point's line, since the line was
        # invisible to the collision check that chose it. Call this on the
        # Annotation itself, not `line.arrow_patch` directly: the patch is
        # constructed with placeholder (0,0)-(1,1) coordinates and only gets
        # its real position from `Annotation.update_positions()`, which runs
        # inside `Annotation.get_window_extent()` -- calling it on the patch
        # before that measures nothing real (verified against this repo's
        # matplotlib: Annotation.get_window_extent() already unions in the
        # arrow's bbox once positions are updated, so this one call is both
        # necessary and sufficient).
        placed.append(line.get_window_extent(renderer=renderer))
