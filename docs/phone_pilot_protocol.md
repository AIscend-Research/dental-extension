# The real phone pilot — protocol, analysis, and what it settles

**Status: not started. Gated on the IRB determination, not on compute.**
`docs/irb_determination_request.md` §B is this step, and it is the one section
of that packet that blocks *collecting* rather than *publishing*: no
photograph should be taken before a determination naming the photography step
comes back. Everything on this page that does not require a photograph has
been built and run — the shot list, the registration, the severity fits, and
the cross-arm calibration in §8, whose numbers are real and were measured on
real DENTEX radiographs.

Written 2026-08-19.

## 1. Why this is the pilot the project keeps deferring

Three separate open items in this repo are the same missing input:

| Open item | Where it is recorded | What it needs |
|---|---|---|
| Which synthetic arm matches real phone artifacts | `TASKS.md` Stream 1 — the OpenCV-vs-albumentations findings are eyeball judgements on one image | photographs |
| The GAN degradation arm | `docs/phase2_data_notes.md` — deliberately deferred, because it cannot be trained *or* evaluated without real photos | photographs |
| The sim-to-real gap named as the framework's principal limitation | `docs/simulator_grounding.md` §5–6 | photographs |

And one item is *not* on that list, deliberately: this pilot is not a
deployment trial. The paper's framing defers clinical deployment to a sequel,
and photographing printed films does not undo that. What the pilot buys is the
right to say the simulator's artifacts resemble real ones — a validity claim
about the instrument, not an efficacy claim about the system.

## 2. What this pilot can and cannot settle

**Can:**

- Whether a photograph taken under a named condition falls inside the range of
  images the simulator produces for that artifact, and at what severity
  (`src/pilot/realism.py:fit_severity`).
- Which of the two synthetic arms is closer to reality, per artifact, as a
  number that can come out either way (`compare_arms`).
- Whether the severity ranges used for training (`severity_range=(0.3, 0.9)`)
  cover where real photographs actually sit, or whether they train on a
  regime nobody photographs.
- Whether there is an artifact the pilot photographs show and *no* arm
  produces. The statistics table carries two candidates on purpose — `moire`
  (print halftone beating against the sensor grid) and `illum_gradient`
  (one-sided lighting) — neither of which any arm in this repo models.

**Cannot:**

- Establish that a detector trained on synthetic degradations works on real
  photographs. That is a downstream experiment needing the trained detector
  (`docs/kaggle_instructions.md` steps 3–6), and tens of photographs would not
  power it anyway.
- Validate `SIGNAL_LOSS_WEIGHTS`. That needs damage curves from a competent
  reader on labelled data, and E6 already showed the public split is too small
  (`docs/simulator_grounding.md` §5). The pilot can say the *artifacts* are
  realistic; it cannot say how much diagnostic signal each destroys.
- Speak to intra-oral photography, phone-to-monitor capture, or any device
  class not in the shot list.
- Prove realism from a small residual. A large residual is strong evidence an
  arm is wrong; a small one is weak evidence it is right (`src/pilot/realism.py`
  module docstring).

## 3. Materials

- **Films.** 6 source radiographs, printed. Take the cheap governance path in
  `docs/irb_determination_request.md` §B variant (1): print DENTEX images we
  already hold. That keeps the pilot on public, already-de-identified data,
  and it has a scientific advantage that is easy to miss — the digital source
  is *exactly* the reference each photograph gets registered against, which
  variant (2)'s clinic films could never provide.
- **Which 6.** Span the dataset rather than picking the prettiest: two
  radiographs with visible caries annotations, two with only impacted or
  periapical findings, two chosen at random. Record the filenames; they are
  the manifest's `source_image` column and the report's reference images.
- **Printing.** Whatever a clinic would plausibly have. If a choice exists,
  print two of the six on plain paper and four on film or photo paper, and
  record which — paper stock changes the halftone screen, which is exactly
  what the `moire` statistic measures.
- **Phone.** One consumer smartphone, one photographer. Record the model in
  the manifest's `device` column. A second device is a nice-to-have, not a
  requirement; two devices with three films each is a worse pilot than one
  device with six, because condition and device become confounded.
- **Backlight.** A lightbox if available, a bright window if not, a phone
  screen or tablet as a last resort. Record which in `notes`.

## 4. Sample size

11 shots per film (2 controls + 2 each for glare, angle, blur, low_light + 1
free-form) × 6 films = **66 photographs**. The number is set by what the
analysis needs, not by a power calculation:

- Every artifact gets a mild and a severe condition, so the fit sees a *range*
  rather than a point, and a flat fit curve can be distinguished from a good
  one (`SeverityFit.identifiable`).
- Two controls per film give the noise floor. Any real-vs-synthetic difference
  smaller than the control-to-control difference on the same statistic is not
  a finding, and without repeated controls there is no way to say so.
- 6 films × 2 conditions per artifact = 12 photographs per artifact, enough
  for a median and a visible spread, not enough for a confidence interval.
  Report medians and per-image values; do not report a p-value off this set.

If the photographer's time is shorter than that, cut *films*, not conditions.
Four films still covers every artifact; dropping the severe conditions leaves
the fits unidentifiable.

## 5. Procedure

1. **Clear the IRB gate.** `docs/irb_determination_request.md` §B, in writing,
   naming the photography step. Record the determination in that file before
   step 2.
2. **Print the films** and label the *back* of each with its source filename.
   Never write on the front: it lands in the photograph, and in variant (2)
   any front-of-film marking (name, DOB, accession number, clinic stamp) must
   be masked before photography, not cropped afterwards.
3. **Generate the shot list:**
   ```bash
   python scripts/pilot_report.py plan --films data/pilot/references \
       --out data/pilot/manifest.csv --seed 0
   ```
   Record the seed. The condition order is randomised within each film — print
   drift, moving daylight, and operator fatigue (which `src/sim/state.py`
   explicitly models) all confound with condition if every film is shot in the
   same order.
4. **Shoot in manifest order**, filling `filename`, `device`, `distance_cm`
   and `notes` as you go. An empty `filename` means "not taken" and the report
   counts it that way; do not backfill guesses.
5. **The transport step, for `jpeg`.** Compression is not something the hands
   can do (`src/pilot/plan.py`, `CONDITIONS`). After shooting, send the six
   *control* photographs through whatever messaging app a clinic would
   actually use, save what comes out the other end, and add them to the
   manifest as extra rows with `condition` = `control_transport` and
   `stresses` = `jpeg` (the report keys the fit off `stresses`, so a row
   without it is registered and measured but never fitted). That is the only
   honest way to measure
   what transport does, and it is the artifact the simulator calls unfixable.
6. **Ingest.** Copy the originals unmodified into `data/pilot/photos/`. Do not
   crop, rotate, auto-enhance, or convert — every one of those changes the
   statistics the whole analysis is built on. Registration does the cropping.
7. **Report:**
   ```bash
   python scripts/pilot_report.py report --manifest data/pilot/manifest.csv \
       --photos data/pilot/photos --references data/pilot/references
   ```
   which writes `results/pilot_realism.json`: per-photograph registration
   quality and statistics, per-artifact severity fits for both arms, and the
   arm comparison.

## 6. Acceptance criteria

Decide these before looking at the output.

| Check | Threshold | If it fails |
|---|---|---|
| Registration success | ≥ 80% of photographs register (`RegistrationResult.ok`) | Re-shoot the failures with more of the film's border in frame; severe glare erases the features it covers, and that is a real, reportable finding, not only a nuisance |
| Registration accuracy | median reprojection error < 3 px | Do not report any conclusion that depends on high-frequency statistics (`sharpness`, `hf_ratio`, `blockiness`) |
| Noise floor | control-vs-control distance, per statistic | Any arm difference below this is reported as a tie, whatever the medians say |
| Fit identifiability | `SeverityFit.identifiable` on ≥ 2/3 of fits for an artifact | Report that artifact as "not resolvable from this pilot" rather than reporting its median severity |
| Arm winner | median residual ratio > 1.1 (`ArmComparison.winner`) | Report a tie. `TASKS.md`'s current glare claim is exactly the kind of judgement this exists to overturn or confirm |
| Intent held | photographs taken under a condition actually score highest on the statistic it stresses | The protocol, not the simulator, is what failed; fix the instruction and re-shoot |

## 7. What the results feed back into

In this order, because each step depends on the one before:

1. `TASKS.md` Stream 1 — replace the eyeballed arm comparison with the
   measured one, and say which arm the project standardises on.
2. `src/data/degradation.py` — retune each degradation's parameter ranges so
   the severity a real photograph fits at means what it says. If the fitted
   severities cluster outside `(0.3, 0.9)`, that default is training the
   detector on a regime nobody photographs, and it changes.
3. **A new degradation, if the unfitted gaps demand one.** `per_stat_gap`
   reports all ten statistics, not only the fitted ones. Large, consistent
   `moire` or `illum_gradient` gaps mean the taxonomy in
   `docs/simulator_grounding.md` §1 is missing a primitive; adding it keeps
   the `[0,1]` severity contract and the label vector, per `TASKS.md`.
4. **The GAN arm** (`docs/phase2_data_notes.md`). This pilot is what unblocks
   it, and registration decides *which kind* it can be:
   - If most photographs register, the pilot is **paired** — each photograph
     sits in its source radiograph's frame, which is a pix2pix-style setup and
     needs far less data than an unpaired one.
   - If registration mostly fails, it is **unpaired** — CycleGAN-style, on the
     clean radiographs versus the photographs as two unaligned domains.
   - Either way, 66 photographs trains nothing on its own. Its realistic use
     is as the *validation* set for a GAN trained on synthetic pairs, and as
     the honest denominator for the same `compare_arms` comparison the other
     two arms get. A GAN arm evaluated on the data it was trained on is not a
     third arm; it is a memorisation check.
5. `docs/simulator_grounding.md` §5–6 — the sim-to-real row stops being
   "attempted, inconclusive" for the artifact-realism half of the question.
   The `SIGNAL_LOSS_WEIGHTS` half stays open; the pilot does not touch it.
6. `docs/paper_draft.md` §6 — the limitations text about the synthetic-vs-real
   gap gets a number instead of a caveat.

## 8. Results

Empty until photographs exist, except the one analysis that needs none.

### 8.1 Cross-arm severity calibration (done, 2026-08-19)

`TASKS.md` records that the two arms' severity scales "are not currently
calibrated to mean the same perceptual thing" and warns against pooling
severity-conditioned results across arms. That warning was based on one
side-by-side image at severity 0.7. Comparing the two simulators to each other
needs no photographs, so it is measured here: for each OpenCV severity, the
albumentations severity whose artifact statistics land closest.

Five real DENTEX validation radiographs, 3 draws per severity, median across
images. `results/pilot_arm_calibration.json` holds the per-image values.
Residuals are in units of the target arm's own sweep spread, so ~0 means "the
same image", ~1 means "as different as a whole severity sweep".

| OpenCV severity | blur | jpeg | low_light | glare |
|---|---|---|---|---|
| 0.2 | 0.2 (res 0.11) | 0.2 (0.00) | 0.2 (0.13) | 0.0 (0.83) |
| 0.4 | 0.4 (0.01) | 0.4 (0.00) | 0.4 (0.38) | 1.0 (2.07) |
| 0.6 | 0.6 (0.00) | 0.6 (0.00) | 0.4 (0.59) | 1.0 (2.85) |
| 0.8 | 0.8 (0.00) | 0.8 (0.00) | 0.4 (0.94) | 1.0 (7.41) |
| 1.0 | 1.0 (0.00) | 1.0 (0.00) | 0.4 (1.05) | 1.0 (15.59) |

Three findings, one of which corrects the existing note:

- **blur and jpeg are on the same scale.** Identity map, residuals at or below
  0.01 above severity 0.2. Severity-conditioned results for these two artifacts
  *can* be pooled across arms — the existing blanket warning is stricter than
  necessary.
- **low_light does not need recalibrating so much as bounding.** The two arms
  agree up to about 0.4 and then albumentations **saturates**: every OpenCV
  severity from 0.6 to 1.0 matches albumentations ≈0.4, with the residual
  climbing to 1.05. So it is not the offset `TASKS.md` implies — above 0.4
  there is no albumentations severity that reproduces the OpenCV arm at all,
  and a translation table cannot fix that.
- **The two glare models are different artifacts, not different scales.** No
  albumentations severity matches, and the residual *grows* with OpenCV
  severity to 15.6 — an order of magnitude past a whole sweep's spread. This
  is the quantitative version of `TASKS.md`'s "diffuse brightness wash versus
  circular halo". Glare results must never be pooled across arms, and which of
  the two is right is exactly what the photographs decide.

`angle` is absent from the table on purpose: it is geometric, and geometry is
recovered from the registration homography rather than from pixel statistics
(`fit_angle_severity`), so there is no statistic on which to compare the arms
without a real photograph.

### 8.2 Registration quality

*Pending photographs.*

### 8.3 Which arm matches reality, per artifact

*Pending photographs.* This is the section `TASKS.md` Stream 1's findings get
replaced by.

### 8.4 Artifacts no arm produces

*Pending photographs.* Watch `moire` and `illum_gradient`.

## 9. Known risks

- **The photographer knows what the conditions are for.** Someone told to
  produce glare produces theatrical glare. The `freeform` condition is the
  control for this, and if free-form photographs fit at systematically lower
  severities than the staged ones, the staged set is a caricature and the
  paper should say so.
- **Printed-then-photographed is not clinic-film-photographed.** Variant (1)
  adds a print step real capture does not have — it is the source of any
  halftone `moire` — and drops whatever a real film's emulsion does. This is
  the cost of the cheap governance path and it belongs in the limitations
  section, not in a footnote.
- **One phone is one camera pipeline.** Modern phones denoise, sharpen, and
  tone-map aggressively and differently. A single device makes the pilot
  internally consistent and externally narrow; say which phone.
- **Registration failure is not random.** It concentrates on exactly the
  severe captures the analysis most wants, so a report with 80% registration
  is not an 80% random sample. Count the failures by condition, which
  `results/pilot_realism.json` does.
