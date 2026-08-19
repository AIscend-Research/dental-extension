"""The shot list: decide what to photograph before anyone picks up a phone.

An unplanned pilot produces forty photographs of whatever the room happened to
do that afternoon, and that set can tell you the simulator is wrong but never
*which part* is wrong. The design here is one film photographed repeatedly
under conditions that each stress a single artifact axis, plus an unconstrained
condition that shows what an operator does when nobody is naming the axis.

Two things in the ordering are deliberate and easy to get wrong:

- **Condition order is randomised within each film.** Print brightness drifts,
  daylight moves, and `src/sim/state.py` asserts the operator gets *worse*
  with each retake (fatigue). Shooting every film's conditions in the same
  order confounds all three with the condition itself.
- **The control shot is repeated, not shot once.** Two clean captures of the
  same film bound the measurement noise of the whole pilot: any real-vs-
  synthetic gap smaller than the control-to-control gap is not a finding.

Severity is *not* assigned here. The photographer cannot dial a severity, and
pretending they can would put a fabricated number in the manifest. Conditions
say what to do physically; `src/pilot/realism.py:fit_severity` recovers the
severity afterwards from the photograph.
"""

from __future__ import annotations

import csv
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.data.degradation import DEGRADATION_NAMES

# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Condition:
    """One physical way to hold the phone, and what it is supposed to produce.

    `stresses` names the degradation in `DEGRADATION_NAMES` this condition is
    meant to dominate, or None for the control and free-form conditions. It is
    a *prediction*, not a label: if the recovered statistics say a "glare"
    photograph is mostly blurred, that is a finding about the protocol (or the
    photographer), and `src/pilot/realism.py` reports it rather than assuming
    the intent held.
    """

    name: str
    stresses: str | None
    instruction: str
    replicates: int = 1

    def __post_init__(self) -> None:
        if self.stresses is not None and self.stresses not in DEGRADATION_NAMES:
            raise ValueError(
                f"condition {self.name!r} stresses {self.stresses!r}, which is not "
                f"a known degradation: {DEGRADATION_NAMES}"
            )


#: The default grid. Every degradation the simulator models gets a mild and a
#: severe condition, except `jpeg` -- transport compression is not something
#: you can do with your hands, so it is applied afterwards (see the protocol's
#: transport step) to shots taken under the control condition.
CONDITIONS: tuple[Condition, ...] = (
    Condition(
        "control",
        None,
        "Film flat on a lightbox or against a window, phone parallel to the film "
        "and centred, both hands, room lights on, no flash. The best photograph "
        "you can take. Shoot this twice.",
        replicates=2,
    ),
    Condition(
        "glare_mild",
        "glare",
        "As control, but with a lamp or window reflection visible at the edge of "
        "the film, away from the teeth.",
    ),
    Condition(
        "glare_severe",
        "glare",
        "Move until the reflection sits over the dentition and you can no longer "
        "read the crowns through it.",
    ),
    Condition(
        "angle_mild",
        "angle",
        "Step about 20 degrees off-axis, keeping the whole film in frame.",
    ),
    Condition(
        "angle_severe",
        "angle",
        "Shoot from roughly 45 degrees off-axis and slightly above, so the film "
        "is visibly a trapezoid in the frame.",
    ),
    Condition(
        "blur_mild",
        "blur",
        "Handheld, one hand, tap to focus somewhere other than the film.",
    ),
    Condition(
        "blur_severe",
        "blur",
        "Handheld while moving the phone slowly sideways during the exposure.",
    ),
    Condition(
        "low_light_mild",
        "low_light",
        "As control, with the room lights off and only ambient daylight.",
    ),
    Condition(
        "low_light_severe",
        "low_light",
        "As control, at night with a single distant lamp, no flash. Dark enough "
        "that the phone's preview is visibly noisy.",
    ),
    Condition(
        "freeform",
        None,
        "Photograph the film the way you would if you were sending it to a "
        "colleague and nobody had given you any of the instructions above.",
    ),
)


# ---------------------------------------------------------------------------
# Shots and manifests
# ---------------------------------------------------------------------------

#: Columns the photographer fills in on site. Kept out of the generated
#: manifest's *values* (they are written empty) so that nothing in the file is
#: a guess: an empty cell is honestly unknown, a filled one was observed.
OPERATOR_FIELDS = ("filename", "device", "distance_cm", "notes")

MANIFEST_FIELDS = (
    "shot_id",
    "order",
    "source_image",
    "condition",
    "stresses",
    "replicate",
    "instruction",
    *OPERATOR_FIELDS,
)


@dataclass
class Shot:
    """One planned photograph. `filename` and friends stay empty until taken."""

    shot_id: str
    order: int
    source_image: str
    condition: str
    stresses: str | None
    replicate: int
    instruction: str
    filename: str = ""
    device: str = ""
    distance_cm: str = ""
    notes: str = ""

    @property
    def taken(self) -> bool:
        return bool(self.filename.strip())


def build_shot_list(
    source_images: list[str],
    conditions: tuple[Condition, ...] = CONDITIONS,
    seed: int | None = None,
) -> list[Shot]:
    """Expand films x conditions x replicates into a randomised capture order.

    Args:
        source_images: the printed films, named by their source file (for the
            cheap IRB path these are DENTEX filenames, so every photograph
            keeps a pointer back to the digital image it must be compared to).
        conditions: defaults to `CONDITIONS`.
        seed: fixes the shuffle. Set it, and record it in the protocol -- a
            pilot whose capture order cannot be reconstructed cannot be
            re-run.

    Returns:
        Shots ordered as they should be taken. `shot_id` is stable under the
        seed and independent of the shuffle, so re-generating the list with a
        different seed renames nothing.
    """
    if not source_images:
        raise ValueError("no source images: nothing to photograph")

    rng = random.Random(seed)
    shots: list[Shot] = []
    for image in source_images:
        stem = Path(image).stem
        per_film: list[Shot] = []
        for cond in conditions:
            for rep in range(1, cond.replicates + 1):
                suffix = f"_{rep}" if cond.replicates > 1 else ""
                per_film.append(
                    Shot(
                        shot_id=f"{stem}__{cond.name}{suffix}",
                        order=0,  # assigned after the shuffle
                        source_image=image,
                        condition=cond.name,
                        stresses=cond.stresses,
                        replicate=rep,
                        instruction=cond.instruction,
                    )
                )
        rng.shuffle(per_film)
        shots.extend(per_film)

    for i, shot in enumerate(shots, start=1):
        shot.order = i
    return shots


def write_manifest(shots: list[Shot], path: str | Path) -> Path:
    """Write the shot list as a CSV for the photographer to fill in."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(MANIFEST_FIELDS))
        writer.writeheader()
        for shot in shots:
            row = asdict(shot)
            row["stresses"] = row["stresses"] or ""
            writer.writerow({k: row[k] for k in MANIFEST_FIELDS})
    return path


def read_manifest(path: str | Path) -> list[Shot]:
    """Read a manifest back, including whatever the photographer filled in."""
    with Path(path).open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    missing = set(MANIFEST_FIELDS) - set(rows[0] if rows else MANIFEST_FIELDS)
    if missing:
        raise ValueError(f"manifest {path} is missing columns: {sorted(missing)}")
    return [
        Shot(
            shot_id=r["shot_id"],
            order=int(r["order"]),
            source_image=r["source_image"],
            condition=r["condition"],
            stresses=r["stresses"] or None,
            replicate=int(r["replicate"]),
            instruction=r["instruction"],
            filename=r.get("filename", ""),
            device=r.get("device", ""),
            distance_cm=r.get("distance_cm", ""),
            notes=r.get("notes", ""),
        )
        for r in rows
    ]


@dataclass
class Coverage:
    """What a (possibly half-shot) manifest actually covers."""

    films: int
    planned: int
    taken: int
    per_condition: dict[str, int] = field(default_factory=dict)
    per_factor: dict[str, int] = field(default_factory=dict)
    uncovered_factors: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return self.taken == self.planned and not self.uncovered_factors


def coverage(shots: list[Shot]) -> Coverage:
    """Count what has actually been photographed, per condition and per factor.

    The `uncovered_factors` list is the one that matters mid-pilot: a factor
    with no photographs is a factor whose simulator weights the pilot cannot
    speak to, and it is much cheaper to notice that while the films are still
    printed.
    """
    taken = [s for s in shots if s.taken]
    per_condition: dict[str, int] = {}
    per_factor: dict[str, int] = {}
    for shot in taken:
        per_condition[shot.condition] = per_condition.get(shot.condition, 0) + 1
        if shot.stresses:
            per_factor[shot.stresses] = per_factor.get(shot.stresses, 0) + 1
    planned_factors = {s.stresses for s in shots if s.stresses}
    return Coverage(
        films=len({s.source_image for s in shots}),
        planned=len(shots),
        taken=len(taken),
        per_condition=per_condition,
        per_factor=per_factor,
        uncovered_factors=sorted(planned_factors - set(per_factor)),
    )
