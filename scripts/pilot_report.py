#!/usr/bin/env python
"""Plan and analyse the real phone pilot (step 7 of docs/kaggle_instructions.md).

    # 1. before shooting -- writes the shot list the photographer fills in
    python scripts/pilot_report.py plan --films data/pilot/films.txt \
        --out data/pilot/manifest.csv

    # 2. today, with no photographs at all -- translates one arm's severity
    #    scale onto the other's, on real radiographs
    python scripts/pilot_report.py calibrate --references data/pilot/references

    # 3. after shooting -- registration, severity fits, arm comparison
    python scripts/pilot_report.py report --manifest data/pilot/manifest.csv \
        --photos data/pilot/photos --references data/pilot/references

`report` refuses to run on an empty manifest rather than emitting a table of
zeros: a pilot report with no photographs in it is the one output that could
be mistaken for a result. The protocol, the acceptance criteria, and the IRB
gate that has to clear before step 3 exists at all are in
`docs/phone_pilot_protocol.md`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.pilot.plan import (  # noqa: E402
    build_shot_list,
    coverage,
    read_manifest,
    write_manifest,
)
from src.pilot.realism import (  # noqa: E402
    SENSITIVE_STATS,
    artifact_stats,
    compare_arms,
    cross_arm_calibration,
    fit_angle_severity,
)
from src.pilot.registration import register_photo  # noqa: E402

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"could not read image: {path}")
    return img


def _list_images(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------


def cmd_plan(args: argparse.Namespace) -> int:
    films_arg = Path(args.films)
    if films_arg.is_dir():
        films = [p.name for p in _list_images(films_arg)]
    else:
        films = [line.strip() for line in films_arg.read_text().splitlines() if line.strip()]
    if not films:
        raise SystemExit(f"no source films found in {films_arg}")

    shots = build_shot_list(films, seed=args.seed)
    out = write_manifest(shots, args.out)
    cov = coverage(shots)
    print(f"{len(shots)} shots across {cov.films} films -> {out}")
    print(f"seed {args.seed} (record this; the capture order is not reconstructable without it)")
    print("conditions: " + ", ".join(sorted({s.condition for s in shots})))
    print("\nFill in the filename/device/distance_cm/notes columns as you shoot. "
          "An empty filename means 'not taken', which is what the report counts.")
    return 0


# ---------------------------------------------------------------------------
# calibrate (needs no photographs)
# ---------------------------------------------------------------------------


def cmd_calibrate(args: argparse.Namespace) -> int:
    refs = _list_images(Path(args.references))
    if not refs:
        raise SystemExit(f"no reference radiographs in {args.references}")

    results: dict[str, list[dict]] = {}
    for name in sorted(SENSITIVE_STATS):
        rows = []
        for ref_path in refs[: args.max_images]:
            ref = _read_image(ref_path)
            for src_sev, tgt_sev, residual in cross_arm_calibration(ref, name, seed=args.seed):
                rows.append({
                    "image": ref_path.name,
                    "opencv_severity": src_sev,
                    "albumentations_severity": tgt_sev,
                    "residual": round(residual, 4),
                })
        results[name] = rows
        matched = {}
        for row in rows:
            matched.setdefault(row["opencv_severity"], []).append(row["albumentations_severity"])
        summary = ", ".join(f"{k:.1f}->{np.median(v):.1f}" for k, v in sorted(matched.items()))
        print(f"{name:>10}: {summary}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"n_images": min(len(refs), args.max_images), "seed": args.seed, "calibration": results},
        indent=2))
    print(f"\nwrote {out}")
    print("Read this as: an OpenCV severity of X is as severe as an albumentations "
          "severity of Y on these statistics. Identity means the two scales agree.")
    return 0


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def cmd_report(args: argparse.Namespace) -> int:
    shots = read_manifest(args.manifest)
    cov = coverage(shots)
    if cov.taken == 0:
        print(f"{cov.planned} shots planned, 0 taken -- nothing to report yet.")
        print("Fill the `filename` column in the manifest as photographs are taken.")
        print("See docs/phone_pilot_protocol.md; the IRB determination "
              "(docs/irb_determination_request.md B) gates this step.")
        return 1

    photos_dir, refs_dir = Path(args.photos), Path(args.references)
    registered: list[dict] = []
    failures: list[dict] = []
    pairs_by_factor: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    angle_fits = []

    for shot in shots:
        if not shot.taken:
            continue
        photo_path, ref_path = photos_dir / shot.filename, refs_dir / shot.source_image
        if not photo_path.exists() or not ref_path.exists():
            failures.append({"shot_id": shot.shot_id, "reason": "missing photo or reference file"})
            continue
        photo, ref = _read_image(photo_path), _read_image(ref_path)
        reg = register_photo(photo, ref)
        if not reg.ok:
            failures.append({"shot_id": shot.shot_id, "reason": reg.reason,
                             "inliers": reg.n_inliers})
            continue
        registered.append({
            "shot_id": shot.shot_id,
            "condition": shot.condition,
            "inliers": reg.n_inliers,
            "reprojection_error": round(reg.reprojection_error, 3),
            "stats": {k: round(v, 5) for k, v in artifact_stats(reg.warped).items()},
        })
        angle_fits.append({
            "shot_id": shot.shot_id,
            "condition": shot.condition,
            "angle_severity": fit_angle_severity(reg.homography, ref.shape[:2]).severity,
        })
        if shot.stresses and shot.stresses in SENSITIVE_STATS:
            pairs_by_factor.setdefault(shot.stresses, []).append((reg.warped, ref))

    comparisons = {}
    for factor, pairs in sorted(pairs_by_factor.items()):
        cmp_ = compare_arms(pairs, factor, arms=tuple(args.arms))
        comparisons[factor] = {
            "n_pairs": cmp_.n_pairs,
            "median_residual": {k: round(v, 4) for k, v in cmp_.median_residual.items()},
            "median_severity": cmp_.median_severity,
            "winner": cmp_.winner,
        }
        print(f"{factor:>10}: n={cmp_.n_pairs} " +
              " ".join(f"{a}=res {r:.3f}@sev {cmp_.median_severity[a]:.1f}"
                       for a, r in cmp_.median_residual.items()) +
              f" -> {cmp_.winner or 'tie'}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "coverage": {"films": cov.films, "planned": cov.planned, "taken": cov.taken,
                     "per_condition": cov.per_condition, "per_factor": cov.per_factor,
                     "uncovered_factors": cov.uncovered_factors},
        "registration_failures": failures,
        "registered": registered,
        "angle_fits": angle_fits,
        "arm_comparison": comparisons,
    }, indent=2))
    print(f"\n{len(registered)}/{cov.taken} photographs registered, "
          f"{len(failures)} failed. wrote {out}")
    if cov.uncovered_factors:
        print("factors with no photographs (the pilot cannot speak to these): "
              + ", ".join(cov.uncovered_factors))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="write the shot list to fill in on site")
    plan.add_argument("--films", required=True,
                      help="directory of source radiographs, or a text file of filenames")
    plan.add_argument("--out", default="data/pilot/manifest.csv")
    plan.add_argument("--seed", type=int, default=0)
    plan.set_defaults(func=cmd_plan)

    cal = sub.add_parser("calibrate", help="translate one arm's severity scale onto the other's")
    cal.add_argument("--references", required=True, help="directory of clean radiographs")
    cal.add_argument("--max-images", type=int, default=5)
    cal.add_argument("--seed", type=int, default=0)
    cal.add_argument("--out", default="results/pilot_arm_calibration.json")
    cal.set_defaults(func=cmd_calibrate)

    rep = sub.add_parser("report", help="register the photographs and fit the arms")
    rep.add_argument("--manifest", required=True)
    rep.add_argument("--photos", required=True)
    rep.add_argument("--references", required=True)
    rep.add_argument("--arms", nargs="+", default=["opencv", "albumentations"])
    rep.add_argument("--out", default="results/pilot_realism.json")
    rep.set_defaults(func=cmd_report)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
