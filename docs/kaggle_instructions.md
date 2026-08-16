# What's left, and how to run it on Kaggle

Everything that doesn't need a GPU, a Kaggle account, or IRB sign-off has
been done in this pass — see `docs/decisions.md`, `docs/paper_draft.md`,
`docs/docket_release.md`, and the new experiments (E9–E11) in
`docs/experiments_results.md`. What's left is genuinely gated on things this
environment cannot do: a Kaggle account action, real GPU time, or a human
with an institution's IRB. This is the exact sequence to run them in, with
what to check at each step and where the results feed back into this repo.

## 0. Before you start

- The Kaggle steps upload **local** data — `data/dentex/` is gitignored (the
  DENTEX license is non-commercial and gated), so it exists only on the
  machine that downloaded it. On the machine where this was last run it holds
  the full training split unzipped (11 GB:
  `training_data/quadrant-enumeration-disease/` + `validation_data/`) and
  nothing needs re-fetching. On a fresh clone it does not exist, and step 1
  will say so and print the download commands; run
  `python scripts/download_dentex.py` (needs `huggingface-cli login` and
  accepting DENTEX's terms) and unzip before uploading.
- The label-map and "lightweight" decisions are locked (`docs/decisions.md`)
  — you do not need to make a call before training; `configs/default.yaml`
  already says what to train (`task: diagnosis`, all four classes).

## 1. Upload DENTEX as a Kaggle Dataset (~15–30 min, account action)

This is the one step that needs *your* account, but the fiddly parts are
scripted (`scripts/upload_dentex_to_kaggle.py`, logic in
`src/utils/kaggle_upload.py`, tested in `tests/test_kaggle_upload.py`):

```bash
pip install kaggle
# https://www.kaggle.com/settings -> API -> "Create New Token" downloads kaggle.json
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

python scripts/upload_dentex_to_kaggle.py --dry-run   # checks everything, uploads nothing
python scripts/upload_dentex_to_kaggle.py             # the training split (~2-3 GB)
```

The dry run is worth the ten seconds: every check it runs would otherwise
fail *after* a multi-gigabyte upload has finished streaming. It resolves your
username, validates the slug against Kaggle's rules (6–50 chars, lowercase,
no underscores), confirms the split is complete, and prints the exact command
it would run. `--update -m "what changed"` versions a dataset you already own
instead of failing with a 409.

It uploads just the `quadrant-enumeration-disease` subfolder (705 images +
annotations, ~2–3 GB), not the full 11 GB archive — it's the only split with
diagnosis labels and is what `configs/default.yaml` and every notebook below
actually point at. For a genuinely held-out eval, publish the validation
split as a second dataset with `--split validation`; that path also copies
`validation_triple.json` in from `DENTEX/`, one level above the uploaded
folder, which a hand-rolled `kaggle datasets create -p validation_data` would
silently leave behind — giving you images with no labels.

Nothing else in the repo needs editing afterwards: `find_dentex_root()`
(`src/utils/kaggle_env.py`) locates the mount under `/kaggle/input/` by
looking for `train_quadrant_enumeration_disease.json`, not by dataset name,
so the slug is yours to choose. You still need the slug to *attach* the
dataset to each notebook (Add Data → Your Datasets).

## 2. `kaggle/00_setup_and_sanity_check.ipynb` (~10 min GPU)

Attach the dataset from step 1. Run top to bottom. This installs
detectron2 + HierarchicalDet, downloads and correctly converts the
Swin-Large backbone weights (the config's own filename points at Swin-Base
by mistake — step 8 fixes this, already discovered and documented in
`kaggle/README.md`), and confirms a real forward + training step on 2 real
DENTEX images. If anything is going to break on Kaggle's actual
torch+CUDA build (the one thing not verified locally, since this repo was
built on CPU/MPS), it breaks here, at step 5 (`from detectron2 import _C`),
cheaply — not 30 minutes into notebook 01.

Before opening a session, run `pytest tests/test_kaggle_notebooks.py` (fast,
no GPU, no data). It is the preflight for everything in these notebooks that
is checkable without hardware: that each cell parses, that the names they
import from `src/` still exist, that no dataset mount path is hardcoded, and
— the one that actually bit — that the bootstrap clones this repo rather
than a fork. Until 2026-08-16 all four notebooks cloned
`christopherh-88/dental-extension`, so a Kaggle session silently ran that
fork's `main` no matter what was merged here. The clone URL now has a single
definition (`src/utils/kaggle_env.REPO_URL`) and the test keeps every
notebook in sync with it.

Two external dependencies this notebook fetches were re-checked on
2026-08-16 and are live: the Swin-Large-22k checkpoint (step 7) and
`ibrahimethemhamamci/HierarchicalDet` with the
`diffdet.custom.swinbase.nonpretrain.yaml` config step 9 merges. What
remains unverifiable outside a session is unchanged and is the whole point of
running it: the detectron2 build against Kaggle's torch+CUDA.

## 3. `kaggle/01_train_baseline_detector.ipynb`, BOTH arms (11–28h GPU each, unverified)

Run twice, in separate sessions, with `TRAIN_ARM` set to `"baseline"` then
`"robustness"` — the notebook writes to separate `OUTPUT_DIR`s
(`checkpoints_baseline`, `checkpoints_robustness`) automatically so the two
can't collide. **Before committing to either full run**, the notebook's
section 5 times 100 real training steps on the actual Kaggle GPU and prints
an iteration-rate estimate for the full 40k-iteration run — do this and
read the number before walking away. The only measurement on record right
now is 35.1s/step on CPU (~16.3 days), with a GPU estimate of 11–28h that is
an *unverified extrapolation* (`docs/phase3_model_benchmarks.md`) — the real
GPU number could reasonably fall outside that range, and knowing which end
determines how many Kaggle sessions to plan for (checkpoint/resume is
built in and tested — killing and resuming a session picks up from the last
checkpoint's iteration, not from 0).

The robustness arm is what makes "training on synthetic degradation helps"
a checkable claim rather than an assertion — running only one arm produces
a checkpoint, not a result. Both are needed before notebook 03 means
anything.

## 4. `kaggle/02_train_confidence_head.ipynb` (~1–2h GPU)

Can run in parallel with step 3 — it doesn't need step 3's checkpoint, only
detectron2 installed (for the real Swin-L FPN p5 features it trains
against, replacing the `TinyTrunk` stand-in
`docs/phase3_confidence_head_training.md` used to validate the architecture
at small scale). This is the single highest-value GPU run on this list: E4
already shows verdicts-per-capture falls from 0.230 to 0.125 as confidence-
head noise rises from 0 to 0.40 — this channel is the most load-bearing
thing measured in the entire framework, and every number in
`docs/experiments_results.md` currently uses either the analytic surrogate's
noise-injected stand-in or the small linear head trained on 2,767 real
crops (E6, this session), not this.

## 5. `kaggle/03_evaluate.ipynb` (~1h GPU, run once per arm)

Point `CHECKPOINT_PATH` at step 3's output for whichever arm you're
evaluating (`EVAL_ARM = "baseline"` or `"robustness"`). Produces the real
mAP/per-class-F1 numbers `src/eval/metrics.py`'s `coco_map()`/
`per_class_f1()` have been sitting ready for since Phase 4 — currently only
validated via a perfect-self-prediction sanity check, never against actual
model predictions. The confidence/deferral section of this notebook is an
explicit **template, not working code** (see the notebook itself) — it
needs step 6 below to exist first.

## 6. Wire up `DetectorChannel` (code exists now, needs steps 3+4's weights)

`src/models/detector_channel.py` is written and unit-tested against fakes
(`tests/test_detector_channel.py`) — the adapter logic is checked, but
nothing exercises it against a real detector yet, because nothing in this
environment has trained one. Once steps 3 and 4 produce real weights:

1. Load them into a `CariesDetector` (`src/models/detector.py` — still a
   stub; this is the one piece of real engineering left, wiring
   `build_model()`/`DetectionCheckpointer` from `external/HierarchicalDet`
   into `.encode()`/`.predict()`, per that file's existing TODO comments).
2. Load step 4's trained `ConfidenceHead` weights.
3. `DetectorChannel(detector=..., head=..., caries_class_ids=(1, 3))` — the
   class ids are already resolved by the label-map decision
   (`docs/decisions.md` #1), not something to redecide.
4. Point `experiments/common.py::build_world` at it instead of
   `SurrogateChannel`, and re-run `experiments.run_all e3 e4 e5` — the
   roadmap's own estimate for this is ~2–4h GPU (12k forward passes, 8+
   hours on CPU alone), so do this on Kaggle, not locally.

This is the step that upgrades every E3/E4/E5 number in
`docs/experiments_results.md` from "an analytic surrogate proves the
theorem's coverage" to "a real trained detector produces these verdicts per
photograph" — the single biggest remaining gap between what's proved and
what's demonstrated on a real model.

## 7. Real phone pilot (gated on IRB, not compute)

Photographing printed films to validate the synthetic degradation arms
against reality, and to unblock the deferred GAN-based degradation arm
(`docs/phase2_data_notes.md`). This needs a human with a phone and printed
radiographs, and — per `docs/phase1_background.md`'s IRB note — likely needs
an actual institutional determination before any photos are taken, not
after. Nothing about this is compute-bound; it's the one item on this whole
list that isn't waiting on Kaggle at all.

## What to do with the results

Once steps 3–6 produce real numbers, the sections in `docs/paper_draft.md`
marked "[pending re-run]" or "[numbers pending]" are the ones to fill in —
the intro, theory, and limitations sections do not depend on any of this
and are already current. `docs/experiments_results.md` is the canonical
results record; update it first, `docs/paper_draft.md` second (it explicitly
says it compresses that file, not the other way around).
