# The Burden-of-Proof Camera

**Evidential capture**: a formal task, an anytime-valid theory, and a benchmark
for diagnostic systems that must *meet a stated burden of proof* before
rendering a verdict — and may subpoena better evidence when they cannot.

A screening model looking at a smartphone photo of a dental radiograph has
three options, not one: convict (caries), discharge (sound), or demand another
photograph naming what to fix. The question this repo formalises is not "how
accurate is the model" but **how many justified verdicts it can produce per
photograph taken**, subject to an error guarantee that actually holds when the
system is allowed to keep retaking.

Three contributions, no new photography required:

1. **Theory** (`src/evidence/`, [docs/theory_anytime_validity.md](docs/theory_anytime_validity.md)) —
   evidence as wealth. A retake loop is optional stopping, which breaks
   fixed-level testing by multiplicity and breaks split-conformal by
   non-exchangeability. Betting against the null with stakes that read only the
   *degradation* channel, never the diagnosis, restores an anytime-valid
   guarantee under arbitrarily adaptive retaking. The measurability condition
   is enforced by the type system, not by discipline.
2. **The world** (`src/sim/`) — a capture-session *process*, not a corruption
   filter. A latent scene (glare, tremor, darkness, tilt) that persists across
   shots and responds causally and imperfectly to instructions, with side
   effects, operator fatigue, and partial compliance. Benchmarking on i.i.d.
   corruptions understates targeted retaking by ~7× (E4).
3. **The benchmark** (`src/bench/`) — *The Docket*: a frozen, seeded challenge.
   Given K captures and a mandated standard of proof, maximise justified
   verdicts per capture. Verdict rate cannot be gamed, because the
   false-conviction rate sits in the next column with its confidence interval.

Results, including five refuted predictions of our own design, are in
[docs/experiments_results.md](docs/experiments_results.md).

```bash
.venv/bin/python -m experiments.run_all      # ~9 min, CPU only
```

---

## The original application track

The framework above grew out of, and still contains, the applied project: a
lightweight caries detector for smartphone photos of dental X-rays, built for
offline use in low-resource clinics. Two things make it different from a plain
detector: it stays accurate when the input is a bad phone shot (blur, glare, off
angle, poor light), and it knows when an image is too degraded to trust, so it
asks for a retake or flags for a clinician instead of quietly guessing wrong.

Built on top of [HierarchicalDet](https://github.com/ibrahimethemhamamci/HierarchicalDet)
(the MICCAI 2023 DENTEX baseline: DiffusionDet + a modified Detectron2), using
the [DENTEX](https://huggingface.co/datasets/ibrahimhamamci/DENTEX) panoramic
X-ray dataset.

## Quickstart

```bash
git clone <this-repo> caries-confidence && cd caries-confidence
bash setup_env.sh
```

That installs the core dependencies (no GPU needed), clones the baseline into
`external/`, runs the tests, and generates `degradation_demo.png` so you can see
the degradation pipeline working. It takes a couple of minutes.

The detector stack (Detectron2) is a separate, heavier install and only one
person needs it at first. See [SETUP.md](SETUP.md).

## The idea in one paragraph

Real clinics photograph X-ray films off a lightbox with whatever phone they
have. Those photos are blurry, glary, tilted, and compressed by whatever
messaging app moved them around. A model trained on clean scans falls apart on
them, and worse, it fails silently. So we (1) train the detector on
synthetically degraded images so it holds up, and (2) add a small head that
predicts how degraded an image is and which artifact dominates, which drives a
predict / retake / refer decision. There is an optional third piece: fuse two or
three quick shots of the same film into one cleaner reading, using how much the
shots agree as an extra confidence signal.

## What runs today vs what needs the detector

You do not have to wait for the Detectron2 install to start working. Most of the
codebase is decoupled from it on purpose.

The evidential-capture framework (everything in the section above) runs
entirely without the detector stack: numpy + opencv + matplotlib, CPU only.

- `src/sim/` — capture-session simulator: latent scene state, instruction
  response with side effects, operator fatigue, renderer over the existing
  degradation primitives
- `src/evidence/` — e-processes, conformal and likelihood-ratio e-values,
  degradation-stratified calibration, the standards-of-proof ladder, and the
  two-sided verdict machine with its subpoena power
- `src/bench/` — The Docket: benchmark spec, seven capture policies (four
  sound, two deliberately unsound, one oracle), leaderboard metrics with a
  binomial guarantee audit
- `src/models/diagnostic.py` — the analytic reader, anchored at clean and
  clinic AUC
- `src/models/real_channel.py`, `src/data/dentex_crops.py` — the real-image
  arm: DENTEX tooth crops, a learned reader, and a learned degradation head
- `experiments/e1..e6` — the six experiments; `experiments/run_all.py` runs them

Also runs right now, no GPU, no detector:
- `src/data/degradation.py` — the synthetic phone-artifact pipeline (Phase 2
  core), including ground-truth box remapping for detector training
  (`apply_degradations(..., boxes=...)`)
- `src/data/degradation_albumentations.py` — the albumentations-preset
  alternative, same interface, for the Phase 2 degradation-realism ablation
- `src/data/dentex.py` — DENTEX COCO loading, image-level split (no patient id
  in the data -- see the module), class balance, class weights, repeat factors
- `src/eval/metrics.py` — safe-deferral/risk-coverage metrics, detection
  metrics (`coco_map`, `per_class_f1`), threshold sweep, ablation table,
  calibration (`expected_calibration_error`, `reliability_diagram_bins`)
  (Phase 4 core -- all implemented and tested, but need Phase 3's trained
  detector for real numbers to plug in)
- `src/eval/plots.py` — risk-coverage curve and reliability-diagram plotting
  (Phase 4)
- `docs/` — Phase 1 background/delta/workshop pick, Phase 2 data notes
  (inter-rater disagreement, Kaggle usability), Phase 3 model benchmarks,
  Phase 4 adjacent-field inspiration
- `src/utils/config.py` — loads `configs/default.yaml` (paths, degradation
  params, decision thresholds), so those live in one place instead of being
  copy-pasted into each script
- `src/utils/kaggle_env.py` — the Kaggle session bootstrap the notebooks share
  (idempotent clone, dependency install that can't move numpy/torch, DENTEX
  lookup, detectron2 import ordering)
- `demo_degradation.py` — visual before/after grid
- `demo_degradation_compare.py` — hand-built vs albumentations side by side
  (see `figures/example_degradation_compare.png`)
- `tests/` — all suites pass

Needs the detector stack (Phase 3), see SETUP.md for the confirmed-working
install recipe (a separate `.venv-detector`, not the core venv):
- `src/models/detector.py` — wrapper around HierarchicalDet, still a stub
- `src/models/confidence_head.py` — **real, tested `nn.Module`** now (not a
  stub): pools a feature map, predicts per-degradation severity + a usability
  score, both in `[0,1]`. Confirmed against the actual backbone's real
  feature-map shapes. Not yet trained.
- `src/models/fusion.py` — **real, tested `nn.Module`** now: attention-weighted
  burst fusion + a cross-frame-agreement signal (explicitly flagged as an
  unvalidated design choice — see the module docstring). Not yet trained.

`detector.py` is still a stub (the caries-only vs all-four label-map decision
needs to land first). The other two guard `import torch` so they still import
cleanly with no torch installed; test them via `tests/test_models_torch.py`
in the detector venv, not the core one.

## Layout

```
caries-confidence/
  setup_env.sh              bootstrap the startable half
  SETUP.md                  detector stack install -- confirmed working recipe
  TASKS.md                  roadmap phases -> parallel workstreams, live status
  configs/default.yaml      paths, degradation params, decision thresholds
  docs/                     Phase 1-4 writeups: background/delta/workshop,
                            data notes, model benchmarks, adjacent-field
                            inspiration, confidence-head training results
  kaggle/                   4 notebooks for the GPU training run -- setup,
                            train detector (baseline AND robustness arms),
                            train confidence head, evaluate
                            (see kaggle/README.md)
  scripts/
    clone_baseline.sh       clone HierarchicalDet into external/
    download_dentex.py      pull DENTEX from Hugging Face
    train_confidence_head_standalone.py  trains the confidence head without
                            waiting on the full detector -- already run, see
                            docs/phase3_confidence_head_training.md
  src/
    data/degradation.py     WORKING degradation pipeline + burst generator
                            + box remapping for the geometric degradation
    data/degradation_albumentations.py  the albumentations ablation arm
    data/dentex.py          DENTEX loading, image-level split (no patient id
                            in the data), class balance/weights/repeat
                            factors, detectron2 dataset registration
    models/detector.py      HierarchicalDet wrapper (stub -- label-map
                            decision still needed)
    models/confidence_head.py  real, tested nn.Module + decide()
    models/fusion.py        real, tested nn.Module (burst fusion)
    eval/metrics.py         selective-prediction + detection + calibration
                            metrics, all real and tested
    eval/plots.py           risk-coverage + reliability-diagram plotting
    utils/config.py         loads configs/default.yaml
    utils/kaggle_env.py     Kaggle session bootstrap shared by the notebooks
    utils/seed.py           reproducibility
  tests/                    runnable checks for the working pieces
  external/                 HierarchicalDet lands here (gitignored)
  data/                     DENTEX lands here (gitignored)
  models_weights/           backbone weights land here (gitignored)
```

## Data and weights

Neither is committed (both are large, DENTEX is licensed). Get them with:

```bash
python scripts/download_dentex.py      # needs `huggingface-cli login` first
```

Backbone weights (`swin_large_patch4_window7_224_22k.pkl`) are separate again —
see SETUP.md for the download-and-convert recipe. Note it is Swin-**Large**:
the upstream config's `MODEL.WEIGHTS` names a Swin-*Base* file while building a
Large model, and loading the Base file raises no error — it just leaves most of
the backbone randomly initialized.

## Convention notes

- Degradation severity is always in `[0, 1]`, and the eval code assumes it is
  monotonic. Keep that contract if you add new degradations.
- Any degradation that **moves image content** must also remap ground-truth
  boxes. Today `angle` is the only one, and
  `apply_degradations(..., boxes=...)` handles it. Degrading pixels while
  leaving boxes in place doesn't crash — it just trains the detector on
  misaligned labels, so add box handling alongside any new geometric
  degradation.
- `patient_level_split` in `src/data/dentex.py` is actually an **image-level**
  split: confirmed against the real download that DENTEX ships no patient id
  anywhere (filenames are just a sequential per-split index). State this as a
  limitation in the paper rather than assuming patient safety.
- Call `src.utils.seed.set_seed()` at the top of every experiment.

## Credit

HierarchicalDet and DENTEX are by Hamamci et al. (MICCAI 2023). Cite both if you
use them — bibtex is in `external/HierarchicalDet/README.md` after you clone.
