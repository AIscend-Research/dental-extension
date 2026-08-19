"""The real phone pilot: photographs of printed films, and what to do with them.

Everything in `src/data/degradation.py` and `src/sim/` is a *model* of what a
phone does to a radiograph. Nothing in this repo has ever compared that model
to a photograph. This package is the machinery for that comparison, so that
when the IRB determination clears (`docs/irb_determination_request.md` B) the
only missing input is the photographs themselves.

Three pieces, in the order the pilot uses them:

    plan          -- build the shot list before anyone picks up a phone, so
                     the captures span the artifact axes the simulator claims
                     to model instead of whatever the room happened to do
    registration  -- warp each photograph back onto its source radiograph, so
                     real and synthetic images can be compared pixel-for-pixel
                     (also the precondition for a *paired* GAN arm)
    realism       -- artifact statistics, and the fits built on them: which
                     synthetic arm matches reality, at what severity, and
                     where both arms miss

The protocol the shot list implements, the acceptance criteria, and what this
pilot can and cannot settle are in `docs/phone_pilot_protocol.md`.

Nothing here fabricates a result. With no photographs on disk the report
script says so and exits; the one analysis that runs today is the cross-arm
severity calibration, which needs no real data because it compares two
simulators to each other.
"""

from src.pilot.plan import (
    CONDITIONS,
    Condition,
    Shot,
    build_shot_list,
    read_manifest,
    write_manifest,
)
from src.pilot.realism import (
    STAT_NAMES,
    ArmComparison,
    SeverityFit,
    artifact_stats,
    compare_arms,
    cross_arm_calibration,
    fit_severity,
    stat_distance,
    stat_vector,
)
from src.pilot.registration import RegistrationResult, register_photo

__all__ = [
    "ArmComparison",
    "CONDITIONS",
    "Condition",
    "RegistrationResult",
    "STAT_NAMES",
    "SeverityFit",
    "Shot",
    "artifact_stats",
    "build_shot_list",
    "compare_arms",
    "cross_arm_calibration",
    "fit_severity",
    "read_manifest",
    "register_photo",
    "stat_distance",
    "stat_vector",
    "write_manifest",
]
