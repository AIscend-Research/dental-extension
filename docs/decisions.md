# Two decisions the roadmap flagged as blocking, resolved

Both were explicitly costed as "no compute needed for the decision itself" --
they were waiting on a call, not a GPU run. Made 2026-08-14.

## 1. Label map: diagnosis (all four classes), not a caries-only collapse

**Decision: keep `configs/default.yaml`'s `model.task: diagnosis`.** Do not
build a caries-only collapse in the detector (`ROI_HEADS.NUM_CLASSES`
reduced to 1, dataset re-registered with a 2-class map). Train
`kaggle/01_train_baseline_detector.ipynb` on all four DENTEX diagnosis
classes (Caries, Deep Caries, Periapical Lesion, Impacted), exactly as it
already does, and collapse to the binary caries/other decision the Docket
actually needs at the evaluation boundary instead: `src/models/
detector_channel.py::_caries_score` takes the max detection score over
`caries_class_ids=(1, 3)` (Caries, Deep Caries), the same collapse
`src/data/dentex_crops.py`'s `CARIES_IDS`/`OTHER_IDS` already uses for the
real-image arm (E6).

**Why.** Every `Case` in the benchmark (`src/models/diagnostic.py`) is
already a binary label -- the Docket has never needed the detector itself to
be binary, only the *reading* of it to be. Given that, training on all four
classes strictly dominates a caries-only collapse:

- More supervisory signal per image (the auxiliary Impacted/Periapical
  labels are free -- DENTEX ships them regardless -- and standard multi-task
  intuition says they should help, not hurt, the shared backbone's caries
  sub-task, not compete with it).
- Zero additional engineering or GPU time: dataset registration
  (`src/data/dentex.py:register_dentex_detectron2`) and the training
  notebook already run this configuration end to end; a caries-only variant
  would need new `ROI_HEADS.NUM_CLASSES`, a re-registered dataset, and a
  second full training run to compare against -- exactly the "another full
  detector cycle" cost the roadmap flagged this decision as trying to avoid.
- Stays comparable to HierarchicalDet's own reported numbers, which are
  against all four classes -- useful for the baseline-comparison table
  `docs/phase1_background.md` calls for.
- The collapse-at-the-boundary pattern is already implemented and tested
  for the real-image arm (`dentex_crops.py`), so `DetectorChannel` reuses a
  pattern that already exists rather than inventing a second one.

**What this resolves.** `configs/default.yaml` previously said `task:
caries_only` while describing a collapse that existed nowhere in the code;
it was corrected to `diagnosis` (matching what actually runs) as a
stopgap, with the real decision deferred to this note. Nothing in
`configs/default.yaml` needs to change again -- this document is the record
of why `diagnosis` is now a locked choice, not a placeholder.

## 2. The "lightweight" claim: dropped, not fixed by a backbone swap

**Decision: stop calling the detector "lightweight."** Reframe the
project's applied-track description around what is actually true: a
diagnostic reader wrapped in a capture process that stays accurate under
phone-shot degradation and knows when to ask for a retake, evaluated here on
Swin-L -- a strong, not small, backbone. Do not spend a GPU cycle swapping
to a smaller backbone (Swin-T, MobileNet-family) to make the word true.

**Why.** `docs/phase3_model_benchmarks.md` measured the actual numbers:
281.9M parameters (~1075 MB fp32), 508.1 GFLOPs backbone-only at 800x800,
2.52s/image on CPU. None of that is lightweight by mobile/edge standards
(tens of MB, sub-100ms is the usual bar), and asserting otherwise in a
submission would be a false claim sitting right next to the actual number
in the same paper.

The other option -- swap to a genuinely small backbone and re-measure -- was
rejected on cost/benefit: it is "another full detector cycle" (an 11-28h GPU
training run, per `docs/phase3_model_benchmarks.md`'s extrapolation, itself
unverified without a GPU), and it would not touch the paper's actual
contribution. The README's own framing already states this plainly: "The
detector is no longer on the critical path for the paper. The framework is
model-agnostic," demonstrated on an analytic surrogate reader and a real
learned linear reader (`RealImageChannel`) with zero GPU time, and a trained
HierarchicalDet is one more pluggable `DiagnosticChannel` among those, not
the thing being claimed lightweight. Spending a training cycle on backbone
size would improve a number the paper does not depend on.

**What this resolves.** `README.md`'s "original application track" section
described a "lightweight caries detector... built for offline use in
low-resource clinics" -- updated to drop "lightweight" and state the actual
size honestly, with "offline, once captured" (no claim about model size)
as the low-resource-relevant property instead. A genuinely small backbone
remains a legitimate follow-up if the applied track is ever prioritized
again, but it is explicitly out of scope for this submission.
