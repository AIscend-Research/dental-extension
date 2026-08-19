"""Run every experiment in order, writing results/ and figures/.

    .venv/bin/python -m experiments.run_all           # everything
    .venv/bin/python -m experiments.run_all e2 e3     # a subset

Total runtime is roughly 25-30 minutes on a laptop CPU (E6 grew from ~1 to
~15 min once pointed at the full ~700-radiograph training split instead of
the 50-radiograph validation split; E13 adds another ~10 min). E6, E7 and
E13 are the ones that need data on disk (DENTEX for E6/E7,
`data/chest_xray_pneumonia` for E13 and E14 -- see
`src/data/chest_xray_crops.py`'s docstring for how to fetch it); each is
skipped with a clear message if its data is missing, so the rest still
completes. E14 adds another ~15 min: its captures run through CheXphoto's
corruption model, which costs roughly 3x a simulated one.
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
    e9_burst_vs_sequential,
    e10_instrument_ablation,
    e11_capture_budgets,
    e12_conformal_risk_control,
    e13_second_modality,
    e14_chexphoto_headtohead,
)

EXPERIMENTS = {
    "e1": ("simulator validation", e1_simulator.main),
    "e2": ("anytime-validity", e2_validity.main),
    "e3": ("The Docket leaderboard", e3_leaderboard.main),
    "e4": ("ablations", e4_ablations.main),
    "e5": ("sensitivity", e5_sensitivity.main),
    "e6": ("real DENTEX images", e6_real_images.main),
    "e7": ("qualitative figures", e7_qualitative.main),
    "e9": ("burst fusion vs sequential accumulation", e9_burst_vs_sequential.main),
    "e10": ("instrument ablation (blur-variance heuristic)", e10_instrument_ablation.main),
    "e11": ("per-case capture budgets under a shared cost", e11_capture_budgets.main),
    "e12": ("conformal-risk-control-style baseline", e12_conformal_risk_control.main),
    "e13": ("second real modality: chest radiographs", e13_second_modality.main),
    "e14": ("head-to-head vs CheXphoto's corruption model", e14_chexphoto_headtohead.main),
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
