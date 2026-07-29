# Tasks and who can work in parallel

The point of the scaffold is that people are not all blocked on the same thing.
Below, each workstream says what it depends on. Anything marked "no detector
needed" can start the moment `setup_env.sh` finishes.

## Day one, in parallel

**Stream 1 — Degradation realism (no detector needed).**
File: `src/data/degradation.py`, `demo_degradation.py`.
The pipeline already runs. The real Phase 2 work is judging whether the
synthetic artifacts match actual phone shots, and this is the paper's main
ablation. Compare degradation strategies: the current hand-built OpenCV
transforms vs albumentations presets vs (if you have time) re-photographing a few
printed X-rays with a phone. Tune the parameter ranges so the synthetic images
look like the real ones. Add degradations if real photos show something we
missed, keeping the `[0,1]` severity contract and the label vector accurate.

**Stream 2 — Data and splits (no detector needed).**
File: `src/data/dentex.py`.
Download DENTEX, confirm the layout, point the config at the real paths (done —
see `configs/default.yaml`; the real layout is three zips + a top-level
`validation_triple.json`, and the diagnosis json uses a `category_id_3` /
`categories_3` multi-task schema, not flat COCO). Patient-id logic resolved:
DENTEX ships no patient identifier at all, so `patient_level_split` is an
image-level split — document this as a limitation rather than trying to fix
it further. `class_balance()` now works against the real schema (Caries 62%,
Impacted 17%, Deep Caries 16%, Periapical 4.5% on the training set) — still
need to decide sampler weights vs class-balanced loss. Decide caries-only vs
all-four-diagnosis and lock the label maps.

**Stream 3 — Metrics and decision policy (no detector needed).**
Files: `src/eval/metrics.py`, `src/models/confidence_head.py:decide`.
The selective-prediction metrics work now. Build the analysis around them:
plotting the risk-coverage curve, sweeping thresholds, and defining the
predict/retake/refer decision. You can validate all of it on simulated
(correct, confidence) arrays before the detector exists — the metrics test
already does this.

**Stream 4 — Detector stack (one person, needs GPU).**
File: SETUP.md, then `src/models/detector.py`.
Get HierarchicalDet installed and running on Kaggle (this is the risky path;
start it first). Confirm a baseline forward pass. Benchmark size / latency /
FLOPs so the "lightweight" claim has numbers. Once it runs, everyone else's work
plugs into it.

## Phase-by-phase mapping

### Phase 1 — Background and framing
Not code. Read the HierarchicalDet paper, survey related work, write the delta
statement, check IRB/ethics if any real photos of real films get collected, pick
the target workshop and confirm the deadline/format. Keep notes in a `docs/` or
a shared doc.

### Phase 2 — Data and preprocessing
- Degradation pipeline → Stream 1 (`src/data/degradation.py`, done, needs tuning)
- Base dataset usable on Kaggle → Stream 2 + 4
- Burst simulation for fusion → `make_burst()` in `degradation.py` (done)
- Small real pilot set to validate realism → Stream 1
- Patient-level split → Stream 2 (`patient_level_split`, done, verify the id logic)
- Class imbalance → Stream 2 (`class_balance`, done, act on it)
- Degradation type/severity labels → produced automatically by `DegradationResult`

### Phase 3 — Model development
All of this needs Stream 4 done first.
- Adapt HierarchicalDet → caries-only baseline: `src/models/detector.py`
- Robustness variant on degraded data: feed `degradation.py` through the dataset
  mapper
- Fusion module: `src/models/fusion.py`
- Confidence head with degradation labels as weak supervision:
  `src/models/confidence_head.py`
- Predict the degradation *type*, not just trust/don't: already the head's design
- Decision thresholds: `decide()` (logic done, tune the operating points)
- Size/latency/FLOPs benchmark: part of Stream 4

### Phase 4 — Evaluation and extensions
- Accuracy/F1/mAP across severities: `coco_map()` / `per_class_f1()` in
  `metrics.py` are now real implementations (plain pycocotools + a pure-Python
  IoU matcher, not `coco_3class_eval.py` -- that vendored evaluator hard-imports
  torch/detectron2, so it can't run until Track B is installed anyway).
  Validated against real DENTEX ground truth (705 train images, 3529 boxes) via
  a perfect-self-prediction sanity check. **Still blocked**: real numbers need
  Phase 3's trained detector to produce actual predictions -- nothing here can
  substitute for that.
- Safe deferral rate: `safe_deferral_rate()` (done)
- Risk-coverage plotting: `src/eval/plots.py` (done, see
  `figures/example_risk_coverage.png` -- simulated data, not real results)
- Threshold sweep: `sweep_decision_thresholds()` grid-sweeps
  `confidence_head.decide()`'s two thresholds against outcome arrays (done,
  ready for real usability/detection-score/correctness arrays once they exist)
- Ablations (robustness-only / confidence-only / combined / +fusion):
  `ablation_table()` compares named arms on base accuracy / AURC / safe
  deferral rate (done; needs each arm's real (correct, confidence) arrays from
  Phase 3 runs to produce actual ablation numbers)
- Clinician gut-check on deferred cases: qualitative, no code -- not started,
  needs a human
- Adjacent-field inspiration sweep (computational photography, speech
  rejection, astronomy multi-exposure, triage tiers): not started, needs a
  human literature pass

### Phase 5 — Writing
Draft, internal review, limitations (label noise, synthetic-vs-real gap, dataset
size, no real clinical test), clean the repo, submit on OpenReview.

## A note on sequencing

The one hard dependency is the detector stack. If it slips, Streams 1–3 still
produce a working degradation pipeline, a clean dataset with honest splits, and a
validated metric — which is most of the method section — while Stream 4 gets
unstuck. Do not let four people idle waiting on one Detectron2 install.
