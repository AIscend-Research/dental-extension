"""Run every experiment in order, writing results/ and figures/.

    .venv/bin/python -m experiments.run_all           # everything
    .venv/bin/python -m experiments.run_all e2 e3     # a subset

Total runtime is roughly 13 minutes on a laptop CPU. E6 and E7 are the only
ones that need data on disk; they are skipped with a clear message if the
DENTEX validation split has not been fetched, so the rest still completes.
"""

from __future__ import annotations

import sys
import time
import traceback

from experiments import (
    e1_simulator,
    e2_validity,
    e3_leaderboard,
    e4_ablations,
    e5_sensitivity,
    e6_real_images,
    e7_qualitative,
)

EXPERIMENTS = {
    "e1": ("simulator validation", e1_simulator.main),
    "e2": ("anytime-validity", e2_validity.main),
    "e3": ("The Docket leaderboard", e3_leaderboard.main),
    "e4": ("ablations", e4_ablations.main),
    "e5": ("sensitivity", e5_sensitivity.main),
    "e6": ("real DENTEX images", e6_real_images.main),
    "e7": ("qualitative figures", e7_qualitative.main),
}


def main(argv: list[str]) -> int:
    selected = [a.lower() for a in argv] or list(EXPERIMENTS)
    unknown = [s for s in selected if s not in EXPERIMENTS]
    if unknown:
        print(f"unknown experiment(s): {unknown}; have {list(EXPERIMENTS)}")
        return 2

    failures = []
    for key in selected:
        label, fn = EXPERIMENTS[key]
        print(f"\n\n{'#' * 78}\n# {key.upper()} -- {label}\n{'#' * 78}", flush=True)
        started = time.time()
        try:
            fn()
        except FileNotFoundError as exc:
            # E6 needs the DENTEX download; missing data is a skip, not a failure
            print(f"\nSKIPPED {key}: {exc}")
            continue
        except Exception:
            traceback.print_exc()
            failures.append(key)
            continue
        print(f"\n{key} finished in {time.time() - started:.1f}s", flush=True)

    if failures:
        print(f"\nFAILED: {failures}")
        return 1
    print("\nall requested experiments completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
