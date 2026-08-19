# Tasks and who can work in parallel

> **Status update — the project pivoted to evidential capture.** The roadmap
> below is the original application plan (train HierarchicalDet, make it robust,
> add a confidence head). It is still accurate about that track, and the GPU
> training run is still not done. But the paper's contribution is now the
> framework/theory/benchmark described in the README, which is complete and
> fully evaluated: see [docs/experiments_results.md](docs/experiments_results.md).
>
> What that changes about the roadmap:
>
> - **Phase 2's degradation pipeline is now a dependency, not the deliverable.**
>   `src/sim/` wraps it in a latent capture *process*. The primitives, severity
>   contract and box remapping are used unchanged.
> - **Phase 3's confidence head is now load-bearing infrastructure.** E4 shows
>   verdicts per capture fall 0.230 → 0.125 as head noise rises, so improving
>   it is the highest-value remaining modelling work.
> - **Phase 4's selective-prediction metrics are superseded, not discarded.**
>   Risk-coverage and safe-deferral answer "how good is the confidence
>   ranking"; the Docket answers "how many verdicts per photograph, at a stated
>   error rate". Both are computed.
> - **The detector is no longer on the critical path for the paper.** The
>   framework is model-agnostic and is demonstrated on an anchored analytic
>   reader and on a real learned reader over DENTEX crops. A trained
>   HierarchicalDet would slot in as another `DiagnosticChannel`.
> - **The one thing that still genuinely needs more data**: validating
>   `SIGNAL_LOSS_WEIGHTS` (which artifacts destroy the most diagnostic signal).
>   E6 attempted it and was inconclusive — the public 50-radiograph split is
>   too small for the fine-grained task (AUC 0.481) and too easy on the coarse
>   one (0.986). This needs the full 11GB DENTEX training split.

The point of the scaffold is that people are not all blocked on the same thing.
Below, each workstream says what it depends on. Anything marked "no detector
needed" can start the moment `setup_env.sh` finishes.

## Day one, in parallel

**Stream 1 — Degradation realism (no detector needed).**
File: `src/data/degradation.py`, `demo_degradation.py`.
Note before adding anything: degradations that move image content must remap
ground-truth boxes too (`apply_degradations(..., boxes=...)`). `angle` is
currently the only one. Getting this wrong is silent -- no crash, just a
detector trained on misaligned labels.
The pipeline already runs. The real Phase 2 work is judging whether the
synthetic artifacts match actual phone shots, and this is the paper's main
ablation. Compare degradation strategies: the current hand-built OpenCV
transforms vs albumentations presets vs (if you have time) re-photographing a few
printed X-rays with a phone. Tune the parameter ranges so the synthetic images
look like the real ones. Add degradations if real photos show something we
missed, keeping the `[0,1]` severity contract and the label vector accurate.

**Albumentations arm done** — `src/data/degradation_albumentations.py` (same
five names, same `[0,1]` severity contract, same `DegradationResult` type, so
either arm is a drop-in for the other) and `demo_degradation_compare.py`
(side-by-side grid, see `figures/example_degradation_compare.png`). Ran on a
real DENTEX panoramic at severity=0.7. Findings, reported honestly rather than
tuned to agree:
- **blur, jpeg**: the two arms look comparable.
- **glare**: OpenCV's hand-built version is a diffuse brightness wash, closer
  to a real lightbox/print reflection. Albumentations' `RandomSunFlare`
  produces a distinct circular halo -- reads more like a camera lens flare
  than the target artifact. Prefer the OpenCV arm for glare unless a real
  phone photo shows the halo pattern is actually more accurate.
- **angle**: albumentations' `Affine` can leave a bright replicated-edge wedge
  in a corner under shear -- correct `BORDER_REPLICATE` behavior (it's
  stretching real edge pixels, not a fill bug), but it looks artificial.
- **low_light**: at the *same* nominal severity (0.7), albumentations is
  dramatically darker than OpenCV's gamma-darkening. The two arms' severity
  scales are not currently calibrated to mean the same perceptual thing --
  do not average/compare severity-conditioned results across arms without
  first recalibrating one to match the other, or note this as a caveat.
**Cross-arm severity calibration done** -- the "not calibrated to mean the
same perceptual thing" caveat above is now a measurement, on five real DENTEX
radiographs, and it is more specific than the caveat: `blur` and `jpeg` map
identity between the arms (pool them freely), `low_light` agrees only up to
~0.4 and then albumentations *saturates* (no albumentations severity
reproduces OpenCV above 0.4, so no translation table can fix it), and the two
`glare` models never match at any severity -- they are different artifacts,
not different scales. See `docs/phone_pilot_protocol.md` 8.1,
`results/pilot_arm_calibration.json`, and
`python scripts/pilot_report.py calibrate --references <dir>`.

Still needed (needs a human + a phone, not code): re-photographing a few
printed X-rays to check which arm's artifacts actually match reality. The
analysis that photographs feed into is built and tested -- shot list,
registration onto the source radiograph, and per-artifact severity fits for
both arms (`src/pilot/`, `tests/test_pilot.py`). Protocol, sample size, and
acceptance criteria: `docs/phone_pilot_protocol.md`. The remaining input is
the photographs, and the IRB determination that gates taking them.

**Stream 2 — Data and splits (no detector needed).**
File: `src/data/dentex.py`.
Download DENTEX, confirm the layout, point the config at the real paths (done —
see `configs/default.yaml`; the real layout is three zips + a top-level
`validation_triple.json`, and the diagnosis json uses a `category_id_3` /
`categories_3` multi-task schema, not flat COCO). Patient-id logic resolved:
DENTEX ships no patient identifier at all, so `patient_level_split` is an
image-level split — document this as a limitation rather than trying to fix
it further. `class_balance()` now works against the real schema (Caries 62%,
Impacted 17%, Deep Caries 16%, Periapical 4.5% on the training set). Both
imbalance strategies are implemented: `class_weights()` (effective-number and
inverse-frequency class-balanced loss weights) and `repeat_factors()`
(LVIS-style per-image oversampling, matching detectron2's
RepeatFactorTrainingSampler algorithm) — pick one when wiring up training in
Phase 3, both are ready. Decide caries-only vs all-four-diagnosis and lock the
label maps.

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
Not code. First pass done — see `docs/phase1_background.md`: related-work
survey (closest prior work is CheXphoto, ML4H 2020 -- same smartphone-photo
robustness problem for chest X-rays), a draft delta statement, an IRB note
(DENTEX itself needs no new review; a real phone-photo pilot set would),
candidate workshops (recommend ML4H 2026 Findings track, Sep 10 2026 AoE --
later and more achievable than the roadmap's original Aug 22 placeholder), and
baseline models to benchmark against. This was a search-engine-summary pass,
not a full paper read -- actually read CheXphoto and SelectiveNet in full
before writing the real related-work section, and get the IRB note confirmed
by your institution rather than treating it as settled. The confirmation
request is drafted and ready to submit -- see
`docs/irb_determination_request.md` (three separate asks: secondary use of
DENTEX, the phone-photo pilot, clinician feedback). Sending it and recording
the written determination is a human action, still outstanding.

### Phase 2 — Data and preprocessing
- Degradation pipeline → Stream 1 (`src/data/degradation.py`, done; albumentations
  ablation arm also done, see `src/data/degradation_albumentations.py`; GAN-based
  arm scoped but deliberately deferred, see `docs/phase2_data_notes.md`)
- Base dataset usable on Kaggle → **checked, see `docs/phase2_data_notes.md`**:
  DENTEX (~11GB) is well within Kaggle's 200GB dataset limit, no size blocker.
  Actual upload/attach as a Kaggle Dataset not yet done (needs a Kaggle account
  action, not something buildable from this environment).
- Burst simulation for fusion → `make_burst()` in `degradation.py` (done)
- Small real pilot set to validate realism → **protocol and analysis built,
  see `docs/phone_pilot_protocol.md` and `src/pilot/`**; still blocked (needs a
  human + a phone + IRB clearance, see `docs/phase1_background.md`; the
  clearance request is drafted at `docs/irb_determination_request.md` §B,
  which also notes a lower-governance fallback: re-photographing printed
  DENTEX images rather than clinic-sourced films)
- Patient-level split → resolved as an image-level split, DENTEX has no
  patient id at all (see `src/data/dentex.py:_patient_key`)
- Class imbalance → Stream 2 (`class_balance`/`class_weights`/`repeat_factors`, done)
- Degradation type/severity labels → produced automatically by `DegradationResult`
- Inter-rater label disagreement → **checked, see `docs/phase2_data_notes.md`**:
  DENTEX reports no inter-annotator agreement statistic (single
  student-annotator + one of three expert-dentist reviewers per image); the
  DENTEX paper itself acknowledges this as a limitation -- cite it directly in
  Phase 5's limitations section rather than trying to measure it ourselves.

### Phase 3 — Model development
**Stream 4 (the detector stack) is no longer a hard blocker for the other
three items below** -- confirmed 2026-07-29 that official detectron2 installs
and runs on CPU/MPS via the recipe now in SETUP.md (`--no-build-isolation`,
plus an import-order trick so HierarchicalDet's vendored `detectron2/`/
`pycocotools/` copies don't shadow the real installed ones). `build_model()`
on the real Swin-L + DiffusionDet config (281.8M params) builds and runs a
forward pass on CPU in ~2.7s. This confirms the architecture wires up; it does
NOT confirm training time or quality on real hardware -- the actual GPU/Kaggle
training run is still needed and still the risky, time-consuming part.

**Kaggle notebooks are ready**, see `kaggle/` (00 setup/sanity check, 01 train
the detector with checkpoint/resume, 02 train the confidence head, 03
evaluate). All four were built by actually running the code end to end
locally (not guessed), and found + fixed several real bugs in the upstream
HierarchicalDet repo and its config along the way -- see `kaggle/README.md`
for the full list, the two most important being: the config's backbone
weights filename is wrong (points at Swin-Base, needs Swin-Large -- silently
loads with zero errors, just leaves ~90% of the backbone randomly
initialized), and `hierarchialdet.dataset_mapper.DiffusionDetDatasetMapper`
is broken for anyone but the original author (hardcoded personal file
paths) -- the notebooks use a custom mapper instead.

- Adapt HierarchicalDet → caries-only baseline: `src/models/detector.py` --
  still a stub; the label-map decision (caries-only vs all-four, see
  `src/data/dentex.py`'s `DIAGNOSIS_CLASSES`/`CARIES_ONLY_MAP`) needs to be
  locked before wiring `ROI_HEADS.NUM_CLASSES` and dataset registration here.
  Until it is, `configs/default.yaml` says `task: diagnosis` -- all four
  classes, which is what `kaggle/01` actually trains. It previously said
  `caries_only`, describing a collapse that exists nowhere in the code; the
  config now states what runs, and flipping it is part of landing the
  decision, not a substitute for it.
  Dataset registration itself is done, though -- see
  `src/data/dentex.py:register_dentex_detectron2`/`to_detectron2_dicts`,
  exercised end to end in `kaggle/01_train_baseline_detector.ipynb`. Also
  confirmed: `DiffusionDet.forward()` defaults to `k=0` at inference
  (quadrant-level `pred_classes_1` only) -- diagnosis/caries predictions need
  `model(batch, k=2)` for `pred_classes_3`, easy to miss since `k=0` doesn't
  error, it just silently returns the wrong task's output.
- Robustness variant on degraded data: **wired up** --
  `kaggle/01_train_baseline_detector.ipynb`'s `CariesDatasetMapper` now takes
  `degrade_prob`, and the notebook's `TRAIN_ARM` switch selects `baseline`
  (clean, `degrade_prob=0.0`) or `robustness` (`0.7`), writing to separate
  `OUTPUT_DIR`s so the two arms can't overwrite each other. Verified locally
  against real DENTEX images: the degraded path produces in-frame,
  label-aligned boxes. **The GPU run itself is still not done** -- this is the
  training *path*, not trained weights, and the arms only mean something
  compared against each other.
  - The non-obvious part, now handled: `angle` applies a rotation +
    perspective warp, so it moves image content. Degrading pixels while
    leaving ground truth in place doesn't error -- it silently trains the
    detector on misaligned boxes. `apply_degradations(..., boxes=...)` remaps
    them through the same homography (`transform_boxes()`), and the mapper
    drops boxes warped out of frame along with their class labels so the two
    can't desynchronize. Same contract implemented in the albumentations arm
    via `bbox_params`. See `tests/test_degradation.py`.
- Fusion module: `src/models/fusion.py` -- **real nn.Module now**, not a stub.
  Attention-weighted average over frame features, confirmed against the real
  backbone's FPN p5 output (256 channels): fuses N frame feature maps into one
  same-shape map + softmax attention weights. `cross_frame_agreement()` is
  implemented but its entropy-based interpretation is an explicit, flagged
  judgment call -- validate it against real burst data + correctness labels in
  Phase 4, don't assume it's right. See `tests/test_models_torch.py`.
  Fixed since first written: it flattened the whole `(B, N)` batch into one
  distribution and normalized by `log(B*N)`, so it returned values **above
  1.0** for any batch (1.23 at B=2, 1.77 at B=4) despite documenting a `[0, 1]`
  signal, and mixed unrelated samples together. Now per-sample, returning a
  `(B,)` tensor normalized by `log(n_frames)`. The old B=1-only test couldn't
  see it; there's a batched regression test now.
- Confidence head with degradation labels as weak supervision:
  `src/models/confidence_head.py` -- real `nn.Module`, and **now actually
  trained**, standalone (didn't need to wait for the full detector): see
  `docs/phase3_confidence_head_training.md` and
  `scripts/train_confidence_head_standalone.py`. Trained against a small
  stand-in CNN trunk (not the real, still-untrained Swin-L backbone) on real
  DENTEX images + synthetic degradation labels. Real results at the
  best-validation-loss epoch: **66.7%** dominant-degradation accuracy at 200
  training images and **77.0%** at 495 (vs 20% chance), with usability
  correlation 0.894/0.933 and roughly calibrated usability scores. Once the
  real detector is trained, retrain this head against its actual FPN p5
  features instead of the stand-in trunk.
  - **Correction**: this previously reported 69.3% and "measurably
    overconfident on degraded images (0.696 predicted vs 0.321 true)". That
    overconfidence was an artifact of reporting the *last* epoch when
    validation loss is unstable (0.015-0.18 across epochs), not a property
    of the model. Both the script and notebook 02 now select the best epoch
    by val loss and print which one they used. See
    `docs/phase3_confidence_head_training.md`.
- Predict the degradation *type*, not just trust/don't: already the head's design
- Decision thresholds: `decide()` (logic done, tune the operating points)
- Size/latency/FLOPs benchmark: **done**, see `docs/phase3_model_benchmarks.md`.
  281.9M params (~1075 MB fp32), backbone-only 508.1 GFLOPs @ 800x800, CPU
  inference 2.52s/image. Important finding, not just a number: Swin-L is not
  lightweight by any normal edge/mobile standard -- this needs an honest
  framing decision (report the size as-is vs. swap to a smaller backbone)
  before the paper can claim "lightweight, low-compute."
- Training-time audit (explicit Phase 3 task): **done**, see the same doc.
  Measured a real training step (forward+backward+optimizer, batch=2,
  800x800, all three hierarchical loss heads) at 35.1s on CPU -> ~390h/16.3
  days for the full 40k-iteration run, confirming GPU is mandatory. A rough,
  UNVERIFIED extrapolation to Kaggle GPU hardware suggests ~11-28h, but this
  needs to be re-measured on the actual Kaggle GPU before planning around it.

**Note on both new nn.Module implementations**: they guard `import torch` so
`src/models/*.py` still import cleanly with no torch installed (the core env
that `src/eval/metrics.py` runs in) -- see the top of each file. Test them via
a *separate* venv following SETUP.md's Track B recipe
(`source .venv-detector/bin/activate && python -m pytest tests/test_models_torch.py`),
not the core venv.

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
  rejection, astronomy multi-exposure, triage tiers): **done**, see
  `docs/phase4_adjacent_fields.md`. Concrete follow-ups it surfaced: benchmark
  `BurstFusion` against plain frame averaging (both computational photography
  and astronomy treat averaging as the baseline to beat, not a strawman); try
  a content-agreement alternative to `cross_frame_agreement`'s entropy
  formula; and — implemented, not just noted — a **calibration metric** was
  a real gap the triage-tier analogy surfaced (existing metrics check
  ranking, not whether confidence values are numerically calibrated), so
  `expected_calibration_error()` and `reliability_diagram_bins()` are now in
  `metrics.py`, with a plot in `plots.py` (see
  `figures/example_reliability_diagram.png`).

### Phase 5 — Writing
Draft, internal review, limitations (label noise, synthetic-vs-real gap, dataset
size, no real clinical test), clean the repo, submit on OpenReview.

## A note on sequencing

The one hard dependency is the detector stack. If it slips, Streams 1–3 still
produce a working degradation pipeline, a clean dataset with honest splits, and a
validated metric — which is most of the method section — while Stream 4 gets
unstuck. Do not let four people idle waiting on one Detectron2 install.
