# The Docket: a frozen, seeded benchmark — how to use it if you are not us

This is the release note the roadmap's "prep the release" item asks for: the
Docket is described in the README and implemented in `src/bench/`, but
nothing until now has told an outside user *how to point it at their own
reader, add their own policy, or reproduce a specific number* without first
reading every module. That is what this file is for.

## What you get by installing this repo

Two independent things, and it matters which one you are borrowing:

1. **The theory and machinery** (`src/evidence/`): a generic anytime-valid
   testing library — e-values, a wealth process, a standards-of-proof ladder,
   a measurability-enforcing type (`EvidenceView`/`PeekingView`). Nothing
   here is dental-specific. If you have any sequential screening problem with
   a quality-vs-content channel split, this is reusable as-is.
2. **The Docket** (`src/bench/`): a benchmark harness built on (1) — a case
   distribution, a capture-session simulator, and a leaderboard scorer. This
   part *is* about photographing things (currently dental radiographs), but
   the interfaces are model-agnostic by construction (`DiagnosticChannel`),
   so plugging in a different reader is the intended way to use it, not a
   fork.

## Running the frozen benchmark as shipped

```bash
bash setup_env.sh                        # core deps, no GPU needed
.venv/bin/python -m experiments.run_all  # ~20-25 min CPU, writes results/ + figures/
```

Every number in `docs/experiments_results.md` is reproduced by this command.
`results/*.json` and `results/*.csv` are the machine-readable outputs;
`figures/*.png` are the corresponding plots. Individual experiments run
standalone too: `.venv/bin/python -m experiments.e3_leaderboard`.

## Plugging in your own reader

Everything in `src/bench/` takes a `DiagnosticChannel`
(`src/models/diagnostic.py`) — an object with one method:

```python
class DiagnosticChannel:
    def read(self, case: Case, severities: dict[str, float], rng) -> Reading:
        ...
```

`severities` is what the capture simulator rendered for this shot (per-
artifact severity in `[0, 1]` — see `src/data/degradation.py:DEGRADATION_NAMES`
for the vocabulary). Your `read()` must return a `Reading` with:

- `score`: your diagnosis output, higher = more suspicious of the positive
  class. Any monotone score works; it does not need to be a calibrated
  probability, because the Docket calibrates it itself (see below).
- `degradation`: your *own* estimate of the per-artifact severities — this
  is the degradation channel D_t the validity theorem is built on. If you
  have no such estimate, you can derive a crude one (E10 in this repo shows
  even a single-scalar heuristic broadcast across all five slots retains
  most of the targeting benefit) but you cannot skip it: the stratification,
  the stakes, and the retake instructions all key off it.
- `usability`: a scalar summary in `[0, 1]`; `predicted_usability()` in
  `src/models/diagnostic.py` gives you the reference aggregate (worst-artifact
  dominated) if you don't have your own.
- `true_quality`: for analysis only, never read by any policy. Set to `nan`
  if you don't have ground truth.

Three concrete channels already implement this interface and are worth
reading as examples, in increasing order of realism:
`SurrogateChannel` (analytic, `src/models/diagnostic.py`),
`RealImageChannel` (real image features + linear models,
`src/models/real_channel.py`), `DetectorChannel` (adapter for a real trained
detector, `src/models/detector_channel.py` — the shape a deep model should
take, currently unexercised because no trained weights exist yet in this
repo; see the Kaggle instructions).

## Calibrating against your own population

```python
from src.bench.runner import fit_calibrator
from src.evidence.calibration import LikelihoodRatioCalibrator  # or StratifiedCalibrator

calibrator, calibration_data = fit_calibrator(
    your_channel, n_strata=4, calibrator_cls=LikelihoodRatioCalibrator,
    n=8000, prevalence=0.35, clinic_difficulty=0.5, seed=12345,
)
```

Read `src/evidence/calibration.py`'s module docstring before changing this:
the calibration set MUST be built from first shots only (never from inside a
retake loop), because that mismatch — the calibrator sees an unselected
population, the retake loop tests a selected one — is the entire reason
stratification exists. `src/bench/runner.py::collect_calibration` enforces
this; there is deliberately no convenience path that samples from inside a
session.

Use `StratifiedCalibrator` if you need the provable guarantee and can accept
somewhat less power; `LikelihoodRatioCalibrator` if power matters more and
you are willing to rest on an empirically-checked (not proved) density
estimate. `MarginalCalibrator` is the "ignore the degradation channel"
control most people would build by default — keep it as a comparison arm,
not a replacement, per `docs/theory_anytime_validity.md` §6.

## Defining a docket and running the leaderboard

```python
from src.bench.docket import make_docket
from src.bench.policies import policy_by_name
from src.bench.runner import run_docket
from src.bench.metrics import score_results, format_leaderboard
from src.evidence.ladder import BurdenSpec, PREPONDERANCE, CLEAR_AND_CONVINCING

docket = make_docket(
    "my_docket", n_cases=4000, prevalence=0.35, budget=4,
    burden=BurdenSpec(convict=PREPONDERANCE, discharge=CLEAR_AND_CONVINCING),
    clinic_difficulty=0.5, seed=11,
)
rows = [
    score_results(name, run_docket(docket, policy_by_name(name), your_channel, calibrator), docket.burden)
    for name in ["single_shot", "fixed_retake", "evidential_capture", "one_step_lookahead"]
]
print(format_leaderboard(rows))
```

`LeaderboardRow.guaranteed` and the `false_conviction_ci`/
`false_discharge_ci` columns are not decoration — any arm reporting
`guaranteed=False` (peeked at the diagnosis channel to decide when to stop)
should be read as a measurement, not a promise, and `format_leaderboard`
marks it `UNGUARANTEED` for exactly this reason. If you add a new policy,
set `peeks = True` honestly if it reads `PeekingView` rather than
`EvidenceView` — the whole benchmark's honesty depends on this being
self-reported correctly, and `tests/test_evidence.py::
test_stake_is_independent_of_the_diagnosis_score` is the check that keeps
the sound arms sound.

## Adding a policy

Subclass `CapturePolicy` (`src/bench/policies.py`) and implement `run(ctx,
case_id) -> SessionResult`. If your policy's stopping/betting/instruction
logic only ever touches `EvidenceView` fields, it inherits the validity
guarantee automatically — nothing needs to be proved per-policy, because the
theorem is about the *class* of G_t-measurable policies, not any one of
them. `OneStepLookahead` (`src/sim/lookahead.py` +
`src/bench/policies.py`) is a worked example of a nontrivial sound policy:
it reads only `degradation` and `shot_index` (both G_t-measurable) and
projects expected outcomes through the simulator's coupling structure,
picking a smarter subpoena without touching anything the theorem forbids.

## Adding a case source: real images, not the analytic simulator

`make_image_docket` (`src/bench/docket.py`) builds a docket over a fixed
pool of real labeled images instead of drawing synthetic labels — the
ground truth is whatever the pooled image's label is, so prevalence is
whatever the pool's prevalence is (not independently settable). Pass
`image_pool` to `run_docket` and give each `Case.payload` an image; your
`DiagnosticChannel.read()` is responsible for rendering it at the given
severities (see `RealImageChannel.read` for the pattern — it calls
`src/sim/render.py::render_severities`). `src/data/dentex_crops.py` is the
loader this repo uses for DENTEX; if you're using a different dataset, only
that one file needs replacing — everything downstream is dataset-agnostic
by construction.

## What "frozen" actually means here

The Docket is a *specification*, not a directory of files (see
`src/bench/docket.py`'s module docstring for why: the capture process
responds to the policy under test, so two policies genuinely see different
photographs, and there is no fixed corrupted-image set to ship the way
ImageNet-C ships one). "Frozen" means: given the same `seed`, the same
`n_cases`, `budget`, and `burden`, `make_docket` generates the identical
sequence of case labels, difficulties, and per-case session seeds on any
machine, with no download required. Reproducing a specific leaderboard
number means matching those four arguments plus the reader/calibrator setup
(`experiments/common.py::build_world`'s `CLEAN_AUC_TARGET`,
`CLINIC_AUC_TARGET`, etc., if you are using the analytic surrogate) — not
matching a checked-in dataset.

## Known rough edges for a new user

- `SIGNAL_LOSS_WEIGHTS` (`src/models/diagnostic.py`) is a stated assumption
  about which artifacts destroy the most signal, not a measurement — if you
  plug in a real reader, its actual damage ordering may differ, and E5
  already shows the framework's *policy ordering* conclusions are insensitive
  to this, but your own absolute numbers will reflect your reader's real
  behavior, not the surrogate's.
- The two calibrators (conformal vs. likelihood-ratio) trade validity
  strength for power — always report which one a number came from, the way
  every experiment here does in its header (`world.header()`).
- `n_strata=4` is the default and E4 shows results are flat from 1 to 16
  strata *for this reader*; do not assume that transfers to yours without
  checking, since a much noisier or much cleaner reader could behave
  differently at the edges.
