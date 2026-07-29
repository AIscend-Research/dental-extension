# Confidence-Aware Caries Detection

A lightweight caries detector for smartphone photos of dental X-rays, built for
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

Runs right now, no GPU, no detector:
- `src/data/degradation.py` — the synthetic phone-artifact pipeline (Phase 2 core)
- `src/data/dentex.py` — DENTEX COCO loading + patient-level split
- `src/eval/metrics.py` — the safe-deferral / risk-coverage metrics (Phase 4 core)
- `demo_degradation.py` — visual before/after grid
- `tests/` — both suites pass

Needs the detector stack (Phase 3):
- `src/models/detector.py` — wrapper around HierarchicalDet
- `src/models/confidence_head.py` — the quality/confidence head
- `src/models/fusion.py` — the multi-shot fusion module

The three model files are interface stubs with real signatures and `TODO(phase3)`
markers, so the metric and decision code can be built against them in parallel.

## Layout

```
caries-confidence/
  setup_env.sh              bootstrap the startable half
  SETUP.md                  detector stack install + the Detectron2/Kaggle gotcha
  TASKS.md                  roadmap phases -> parallel workstreams
  configs/default.yaml      paths, degradation params, decision thresholds
  scripts/
    clone_baseline.sh       clone HierarchicalDet into external/
    download_dentex.py      pull DENTEX from Hugging Face
  src/
    data/degradation.py     WORKING degradation pipeline + burst generator
    data/dentex.py          DENTEX loading, patient-level split, class balance
    models/detector.py      HierarchicalDet wrapper (stub)
    models/confidence_head.py  quality head + decide() (stub + working decision logic)
    models/fusion.py        burst fusion (stub)
    eval/metrics.py         WORKING selective-prediction metrics + detection stubs
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

Backbone weights (`swin_base_patch4_window7_224_22k.pkl`) are separate again —
see SETUP.md.

## Convention notes

- Degradation severity is always in `[0, 1]`, and the eval code assumes it is
  monotonic. Keep that contract if you add new degradations.
- `patient_level_split` in `src/data/dentex.py` is actually an **image-level**
  split: confirmed against the real download that DENTEX ships no patient id
  anywhere (filenames are just a sequential per-split index). State this as a
  limitation in the paper rather than assuming patient safety.
- Call `src.utils.seed.set_seed()` at the top of every experiment.

## Credit

HierarchicalDet and DENTEX are by Hamamci et al. (MICCAI 2023). Cite both if you
use them — bibtex is in `external/HierarchicalDet/README.md` after you clone.
