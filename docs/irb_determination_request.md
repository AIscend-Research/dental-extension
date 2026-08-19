# IRB / ethics determination request — draft packet

**Status: unsent.** This is the material needed to turn
`docs/phase1_background.md`'s IRB *note* into a written institutional
*determination*. Nobody in this repo can produce that determination — it is
an action by the institution's IRB or equivalent research-ethics office. What
this file does is remove every reason to delay asking: the facts are already
assembled, the three questions are already separated, and the text below can
be pasted into a determination-request form largely as-is.

Written 2026-08-19. Sections A/B/C are three *separate* questions; they have
different answers and different urgency, and asking them as one blurred
question is the main way this comes back as "please clarify" instead of as a
determination.

Why this can't stay a note: ML4H 2026's call for papers requires IRB
status/exemption to be documented for camera-ready (see
`docs/phase1_background.md`, CFP confirmed 2026-08-14). "We read the license
and concluded it was fine" is not a documented determination. The deadline is
Sep 10, 2026 for submission, so the determination is needed on the
camera-ready timeline, not the submission one — but B below is a hard blocker
*now*, because it gates collecting data at all.

## Priority

| | Question | Blocks | Urgency |
|---|---|---|---|
| **A** | Secondary use of DENTEX as-is | Nothing today; needed in writing for camera-ready | Ask now, low risk of a surprising answer |
| **B** | Re-photographing printed films (the "real phone pilot") | The pilot set, and the GAN degradation arm deferred behind it (`docs/phase2_data_notes.md`) | **Hard blocker — must be resolved before any photo is taken, not after** |
| **C** | Clinician feedback / qualitative review (Phase 4) | The clinical gut-check named as a limitation in `docs/paper_draft.md` §6 | Ask now if that review is going to happen at all |

Ask all three in one submission. They share a study description, and splitting
them into three separate requests mostly buys three separate queues.

## Study description (shared preamble)

> We are developing a machine-learning method for detecting dental caries in
> panoramic dental radiographs that have been degraded by smartphone-capture
> artifacts (glare, blur, moiré, perspective, compression) — the setting where
> a clinician in a low-resource environment photographs a printed film or a
> lightbox rather than exporting a digital image. The method adds a confidence
> mechanism that decides, per image, whether to predict, ask for a retake, or
> refer the case to a clinician. The intended output is a workshop paper
> (ML4H 2026, submission deadline Sep 10, 2026) and a public code release. No
> clinical deployment, no treatment decisions, and no patient contact are
> involved at any stage.

## A. Secondary use of the DENTEX dataset

**Facts to state.**

- Data source: DENTEX (Hamamci et al., MICCAI 2023 challenge dataset),
  distributed publicly on Hugging Face
  (`https://huggingface.co/datasets/ibrahimhamamci/DENTEX`) under CC BY-NC-SA
  4.0, gated only by license acceptance.
- Content: ~11 GB of panoramic dental radiographs with quadrant / enumeration
  / diagnosis annotations. Already de-identified by the original release.
- The dataset carries **no patient identifiers at all** — confirmed in this
  project, not assumed: there is no patient ID field, which is why our data
  split is image-level rather than patient-level (`src/data/dentex.py`,
  `_patient_key`; recorded as a limitation in `docs/paper_draft.md` §6).
- We receive no key, code, or linkage to identifiers, and have no agreement,
  contact, or arrangement with the originating clinics.
- Use is method development and benchmarking only. Derived images (synthetic
  degradations of DENTEX images) stay internal or are released only within
  what CC BY-NC-SA 4.0 permits.

**What we are asking.**

> Does secondary analysis of this public, de-identified, license-gated
> dataset constitute human-subjects research requiring IRB review at this
> institution, or is it a not-human-subjects-research / exempt
> determination? We are requesting that determination in writing so it can be
> cited in a publication's ethics statement.

**What a sufficient answer looks like.** A dated written statement — an
official determination letter, or an email from the IRB office — naming this
project and saying either "not human subjects research" or citing the exemption
category relied on. A verbal "that's fine" is not sufficient for camera-ready.

## B. Real phone pilot — new data collection (hard blocker)

**Facts to state.**

- Planned activity: a small pilot set produced by photographing printed
  radiographs / lightbox-displayed films with a consumer smartphone, to check
  that our synthetic degradation pipeline resembles real phone-capture
  artifacts (`TASKS.md` Phase 2, Stream 1; `docs/kaggle_instructions.md` §7).
- Expected scale: tens of images, not thousands. Photographs of *films*, not
  of people; no patient is present, photographed, or contacted.
- Two variants exist and they are ethically different, so **state which one is
  actually planned**:
  1. **Films already in hand and already de-identified** — e.g. printing
     DENTEX images and re-photographing the printout. This involves no new
     patient data whatsoever; it is arguably a lab procedure on public data.
  2. **Films sourced from a clinic or archive**, including any film not
     already public and de-identified. This is new collection of
     patient-derived data, and is the case that plausibly requires review or a
     data-use agreement even though the underlying films are de-identified.
- If (2): state who holds the films, under what authority we would get access,
  whether any identifiers (name, date of birth, accession number, clinic
  markings) appear physically on the film or in its margins, and how those
  would be masked *before* photography.

**What we are asking.**

> Does photographing already-de-identified printed radiographs to create a
> small methodological validation set require IRB review, an exemption
> determination, or a data-use agreement at this institution? If the answer
> depends on the source of the films, we would like that distinction stated
> explicitly, so we can choose the variant that is covered.

**What a sufficient answer looks like.** Written approval, exemption, or
not-human-subjects determination that *names the photography step*. Do not
begin collecting photographs on the strength of A's answer — A covers
analysing DENTEX, not producing new images.

**Note on the cheap path.** If the determination for variant (2) is slow or
uncertain, variant (1) — printing and re-photographing DENTEX images we
already hold — recovers most of the scientific value (it validates the
synthetic pipeline against real optics, glare, and moiré) at a much lower
governance cost. That is a fallback worth naming in the request, not a
concession to be discovered after a month in a queue.

## C. Clinician feedback and qualitative review

**Facts to state.**

- Planned activity: asking practising clinicians to review a small number of
  model outputs — specifically the ~30 escalated cases the decision policy
  defers — and to say whether they agree those cases are undecidable from the
  photograph (`docs/paper_draft.md` §6 names the absence of this review as a
  limitation).
- The clinicians are subject-matter reviewers giving professional opinions
  about images; no patient data about the *clinicians* is collected, and no
  patient is involved.
- What varies, and what the answer turns on: whether responses are
  **anonymous and aggregate** (no names, no demographics, reported only as
  counts / summary), or whether clinicians are **named, acknowledged, or
  individually quoted** in the paper.

**What we are asking.**

> Does soliciting professional opinions from clinicians about model outputs
> constitute human-subjects research at this institution, or is it expert
> consultation outside the scope of review? If the answer differs depending on
> whether responses are anonymous versus attributable, we would like both
> cases addressed, since we can design the review either way.

**What a sufficient answer looks like.** Either a determination that this is
consultation rather than research, or instructions on the consent language
required. If the answer is "review required and it will take weeks," the
anonymous-aggregate design is the version to pursue.

## Submission checklist

- [ ] Identify the correct office (IRB, research-ethics committee, or a data-
      governance office if the institution routes secondary-data questions
      elsewhere). Ask a departmental research administrator if unclear —
      submitting to the wrong queue is the most common source of delay.
- [ ] Decide and state which variant of B is actually planned (see above).
- [ ] Fill in institution-specific fields: PI of record, department, funding
      source (state "unfunded" if so), study title, expected dates.
- [ ] Attach or link the DENTEX license and dataset page for A.
- [ ] Submit all three questions together, using the shared preamble.
- [ ] Record the outcome (see below) the day it arrives.

## Where the answer gets recorded

When the determination comes back, in this order:

1. Add a dated **Determination** section at the bottom of this file, quoting
   the operative sentence verbatim and noting the reference number, date, and
   which of A/B/C each part covers. Verbatim, not paraphrased — the paper's
   ethics statement should be traceable to actual IRB language.
2. Update `docs/phase1_background.md` § "Ethics / IRB" to point here and drop
   "Confirmation from your institution on the IRB question above" from its
   "Still open" list.
3. Update `TASKS.md` Phase 1 (the "get the IRB note confirmed" line) and
   Phase 2 (the pilot-set line, if B clears it).
4. Write the ethics statement in `docs/paper_draft.md` from the quoted
   language — ML4H requires it documented for camera-ready.

Until step 1 exists, every other document in this repo should keep treating
the IRB question as open, which is what they currently do.

## Caveats on this draft

- The framing above is written in the vocabulary of US Common Rule–style
  review (human-subjects determination, exemption categories). If the
  institution operates under a different regime, the *facts* in each section
  still transfer; the *terminology* may not.
- Nothing here is legal or regulatory advice, and none of it substitutes for
  the institution's own judgement. It is a prepared request, and its only
  claim is that the facts it states about this project are accurate as of
  2026-08-19.
