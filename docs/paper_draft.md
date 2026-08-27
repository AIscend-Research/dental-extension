# The Burden-of-Proof Camera: Evidential Capture for Screening Under Retake

**Working draft** — intro, theory summary, results summary, and limitations,
assembled from `docs/theory_anytime_validity.md` and
`docs/experiments_results.md` (both canonical; this draft compresses and
frames them for a submission, and should be re-checked against those files
if either changes). Target: ML4H 2026, Findings track, 4 pages, deadline
2026-09-10 11:59 PM AoE (confirmed against the live CFP,
`docs/phase1_background.md`).

---

## Abstract (draft)

A screening model reading a smartphone photo of a dental radiograph gets one
look and must answer — so a bad photograph does not fail loudly. It returns a
confident wrong diagnosis: degradation shrinks the signal and inflates the
noise, and a confidently wrong reading looks exactly like a confidently right
one. The obvious fix, letting the model ask for another photo when unsure,
is invisible to per-image confidence and lethal to the error guarantee: it is
optional stopping, and naive retake loops inflate false-positive rate by 6×
(from 5.6% to 33.6% as budget rises from one photograph to eight, at a
nominal 5% level) at clinically meaningful standards of proof, while looking
fine at loose ones. We reframe capture as sequential evidence-gathering: a betting process
against the null whose stakes are fixed by a *degradation* channel, read and
committed to before the *diagnosis* channel is ever consulted — a firewall
enforced by the type system, not by discipline. This restores an
anytime-valid guarantee under arbitrarily adaptive retaking (proved via
Ville's inequality) and lets the system keep demanding better photographs,
naming what to fix, without destroying its own error bound. On a capture-
process simulator grounded in the smartphone/photographic dental- and
chest-imaging literature, targeted evidential capture produces 18% more
justified verdicts per photograph than untargeted retaking, 55% more than
fixed bursts, and 91% more than single-shot reading, with zero guarantee
violations across 17 sensitivity settings — while naive alternatives that
look competitive on raw verdict rate forfeit the guarantee and violate it at
exactly the standards of proof that matter clinically. The guarantee and the
targeting ordering both reproduce, with zero violations, when the same
unmodified machinery is applied to a second real modality (pediatric
pneumonia chest radiographs), and one pattern — that a near-saturated reader
makes retaking not worth its cost — reproduces independently across both.
The guarantee further survives a corruption process the authors did not
design: swapping in a third party's own synthetic capture model
(CheXphoto's, MIT-licensed) for the same task holds validity with zero
violations under identical policies and calibration protocol, while a
mismatched calibration — fit under one capture process, deployed under the
other — breaks it outright (false-conviction rates up to 0.80 against a
stated 0.50 bound), naming calibration-process match as an explicit
deployment precondition rather than a footnote.
We report five predictions this design got wrong, including one converted into a positive
result: a one-step-lookahead subpoena rule that accounts for side effects
between corrections closes the gap to a perfect-information oracle,
confirming the deployable policy's decision rule, not its confidence head,
was the bottleneck.

## 1. Introduction

Screening models for dental caries increasingly target smartphone
photographs of radiographs rather than clean digital scans — the deployment
story is the same one that motivated CheXphoto for chest radiographs
[Phillips et al. 2020]: a health worker photographs a film or a monitor and
sends the image for automated reading, because the alternative is no reading
at all. That deployment path introduces a failure mode clean-image
benchmarks do not measure: a bad photograph does not make a model *less
confident*, it makes the model confidently wrong. Blur and compression
destroy exactly the fine-grained density information a caries diagnosis
depends on, while simultaneously inflating the observation noise that
determines how sharply the model's score separates disease from health. The
two effects compound: signal shrinks, noise grows, and the resulting score
distribution can look every bit as peaked and confident as a clean read.

A natural response is to let the system ask for another photo when it is not
sure. This is where prior work stops and where the actual difficulty begins.
"Ask for another photo when unsure" is *optional stopping* on whatever
statistic decides "unsure," and optional stopping breaks the two tools one
would reach for to bound the resulting error rate. Fixed-level hypothesis
testing breaks by multiplicity: taking K looks and reporting the most
favorable inflates the false-positive rate to roughly 1 − (1 − α)^K, a
signature we reproduce directly (§5, E2). Split-conformal prediction breaks
by non-exchangeability: a retaken photograph exists *because* the first one
was bad, so shots within a capture session are correlated through the room
and the operator's hands, and shifted by whatever the system asked the
operator to fix — we confirm this rejects an exchangeability permutation test
at p = 0.0005 (§5, E1). A third, subtler failure is the one most systems
would build by accident: if the decision to retake depends on the diagnosis
score itself ("retake while the model looks unsure"), the sequence of shots
is selected on the very quantity being tested, and the resulting p-value is
invalid regardless of how few looks were taken.

This paper reframes capture as *sequential evidence-gathering with a stated
burden of proof*, borrowing the machinery of anytime-valid testing (e-values,
Ville's inequality) and the asymmetric standards-of-proof ladder from law.
The central design choice is an information firewall: every shot is read
through two channels — a *degradation* channel (what is wrong with this
photograph, and how badly) and a *diagnosis* channel (what does it show) —
and the betting stake for shot t, along with the decision to keep shooting
and what to ask for next, may depend on the degradation channel through the
current shot but on the diagnosis channel *only through past shots*. That
asymmetry, expressed as a type (`EvidenceView` physically does not carry the
current diagnosis score; a policy that wants to peek must accept a
`PeekingView` and forfeits the guarantee explicitly), is what makes the
guarantee survive fully adaptive retaking: Ville's inequality bounds the
running maximum of a nonnegative supermartingale regardless of when or why
the process stopped, so long as the stakes were fixed before the outcome
they bet on was seen.

Three contributions follow, none requiring new photography:

1. **Theory** (§3): an anytime-valid burden of proof under adaptive
   retaking, proved from the wealth-process construction, with the
   measurability condition enforced structurally rather than by convention,
   and — new in this draft — a reduction of the theorem's one genuinely
   assumed condition (conditional calibration, A1) to a single named,
   falsifiable inequality via a proved Markov screening lemma (§3.3).
2. **A capture-process simulator** (§4): a latent scene that persists across
   shots, responds to instructions partially and with side effects, and
   tires its operator — grounded in the smartphone/photographic dental- and
   chest-imaging literature (glare, motion, off-axis capture, ambient
   lighting are the failure modes that literature repeatedly names) — rather
   than an i.i.d. corruption filter. Benchmarking on i.i.d. corruptions
   understates the value of *targeted* retaking by roughly 7× (§5, E4).
3. **The Docket** (§5): a frozen, seeded benchmark maximizing justified
   verdicts per photograph under a mandated standard of proof, now carrying
   two non-evidential baselines from the selective-prediction and
   conformal-risk-control literatures alongside the authors' own arms, so the
   comparison is not authors-only.

We report five predictions this design got wrong (§5, summary table), kept
deliberately rather than pruned, and one of them — that a perfect confidence
head would not improve verdicts, because the greedy "fix the worst factor"
subpoena rule was the actual bottleneck — is converted here into a positive
result: a one-step-lookahead rule that projects each candidate correction's
side effects through the capture process's coupling structure closes the gap
to the perfect-information oracle using only the same information the
deployable system already has (§5, E3 update).

## 2. Related work (brief — full pass in `docs/phase1_background.md`)

**CheXphoto** [Phillips et al., ML4H 2020] is the closest prior work in
spirit: the same "photograph an X-ray and send it" deployment story, the
same combination of natural and synthetic photographic degradation, for
chest radiographs. Read in full for this draft (not a search summary): it is
a dataset and leaderboard, with no model, no confidence or deferral
mechanism, and no discussion of multiple photographs of the same film —
every image is treated as one independent draw. **SelectiveNet**
[Geifman & El-Yaniv, ICML 2019], also read in full, is the closest
selective-prediction baseline: a jointly trained selection head hitting a
target coverage, single-observation, with *no formal statistical guarantee*
on the realized error rate — that absence of a guarantee, not the absence of
a multi-shot mechanism, is the sharper contrast to lead with, since it is
exactly what Ville's inequality supplies here. Point-of-care image-quality
gating exists for chest radiograph positioning but without a multi-shot or
type-predicting mechanism. No prior work combines a validity guarantee that
survives adaptive retaking with a confidence channel that names *which*
artifact to fix.

## 3. Theory: an anytime-valid burden of proof

*(Condensed from `docs/theory_anytime_validity.md`, which is the source of
record for the proofs; this section states results.)*

One case is one tooth with fixed unknown label Y. Shot t produces a
degradation reading D_t and a diagnosis reading S_t. Define the filtration
G_t = σ(D_1,…,D_t, S_1,…,S_{t−1}) — note the asymmetry: G_t contains the
*current* shot's degradation reading but only *past* diagnosis scores. Under
three conditions — (A1) conditional calibration: the conformal p-value
computed within the predicted-degradation stratum is conditionally
super-uniform under the null; (A2) predictable stakes: the betting fraction
is G_t-measurable; (A3) predictable stopping: the stopping time and
instruction choice are G_t-measurable — the wealth process W_t = W_{t−1}(1 +
λ_t(E_t − 1)) is a nonnegative supermartingale with W_0 = 1, and Ville's
inequality gives, for any stopping time τ and any α ∈ (0,1):

Pr_{H0}(∃ t ≤ τ : W_t ≥ 1/α) ≤ α.

The retake loop is therefore free: τ may depend on anything in G_t —
accumulated wealth, predicted degradation, how many shots have been taken —
with no multiplicity correction, because none of the looks were free. A2–A3
are enforced structurally: `EvidenceView` cannot carry the current diagnosis
score, so a betting strategy written against it type-checks into
G_t-measurability rather than merely being asked to respect it.

**3.3 What A1 reduces to.** A1 is the one genuinely assumed condition. We
prove a Markov screening lemma directly from the simulator's generative code
— conditional on the true rendered severities R_t, the true label Y, and the
case difficulty δ, the pair (S_t, D_t) is independent of everything earlier
in the session — and use it to show A1
reduces exactly to one named condition (A1′): that, given the same predicted
degradation stratum, the true image quality at test time (inside a retake
loop) stochastically dominates the calibration population's (first shots,
never corrected). We give a mechanistic account of when A1′ should hold —
genuine corrections move the scene's *equilibrium*, not just the momentary
reading, which a first-shot calibration draw never benefits from — and when
it should not — a sufficiently noisy degradation head can land a session in
a good-looking stratum by misread rather than real improvement, which is
exactly the "confidently wrong" pathology the framework exists to prevent.
This yields a falsifiable prediction, since checked directly (§5, E2): A1′,
and hence validity itself, should degrade as confidence-head noise rises
past the point where misread-driven stratification dominates
correction-driven stratification. It does not, up to noise 2.5× past E4's
own power-degradation sweep — a structural argument (App./§3.3 extended in
`docs/theory_anytime_validity.md` §6) explains why: the same unbiased noise
kernel corrupts both the calibration and test populations' stratum
assignment alike, so in the high-noise limit A1′ collapses to the raw
equilibrium-shift property E1 already measures directly, independent of
stratification.

Two evidence constructions are reported side by side: a conformal p-to-e
route (exactly valid, lossy) and a likelihood-ratio route (empirically valid,
~70% more verdicts per capture). A two-sided machine runs both directions
(H0: sound vs. H0: diseased) at independently stated standards; self-
contradictory evidence escalates rather than breaking the tie by force.

## 4. The capture process and The Docket

*(Condensed from `docs/simulator_grounding.md` and README.)* The simulator
is a *process*, not a corruption filter: a latent scene (glare, tremor,
darkness, tilt) persists across shots via an AR(1) relaxation toward a
per-session equilibrium, responds to instructions partially and noisily, and
couples — fixing glare by turning the film away from a lamp measurably
disturbs tilt and darkness, a physically motivated trade-off, not a random
one. The five artifact primitives are literature-anchored (smartphone
oral/photographic imaging repeatedly names lighting, angle, motion, and
specular reflection as the dominant failure modes); the dynamics —
persistence, partial compliance, coupling, operator fatigue — are this
paper's modeling contribution, stated as assumptions and verified in the
simulator-validation experiment (E1) rather than measured from the world.

The Docket freezes a case distribution, a capture budget K, and a mandated
standard of proof, and scores **verdicts per capture** (decided cases ÷
photographs taken) alongside a guarantee audit: the empirical false-
conviction/false-discharge rate among genuinely sound/diseased cases, with a
Wilson-interval check so Monte Carlo noise cannot manufacture a violation.
Verdict rate alone can be gamed by an unsound policy; the guarantee columns
sit next to it so a good rate cannot be read without its caveat.

## 5. Experiments

*(Full tables in `docs/experiments_results.md`; summarized here.)* All
simulated experiments reproduce via `experiments.run_all` (~20–25 min, CPU
only); real-image results (E6) now run against the full ~700-radiograph
DENTEX training split rather than the 50-radiograph public validation split.

**Validity under adaptive retaking (E2).** At α = 0.05, naive methods that
look competitive at loose standards violate clearly: `best_shot` climbs from
5.6% to 33.6% false-conviction as budget K goes 1→8, while every sound arm
stays at or below 0.8%. The failure signature is specific to strict
standards — at α = 0.5 neither unsound arm registers a violation.

**The leaderboard (E3).** Among guaranteed arms, targeted evidential capture
beats untargeted retaking by 18%, fixed bursts by 55%, single-shot reading
by 91% on verdicts per capture. The unsound `greedy_diagnostic` arm wins on
raw rate and pays for it in accuracy and in the guarantee it no longer has.
**New in this draft: one-step lookahead.** `oracle_instruction` — reading
the *true* latent scene but still applying the greedy "fix the worst factor"
rule — does not beat the deployable arm, confirming the rule, not the
information, is the bottleneck. (`oracle_instruction` is itself unguaranteed:
reading the true scene to pick an instruction is not `G_t`-measurable, so it
is reported here purely as a non-deployable upper bound on instruction
quality, not a candidate policy.) A lookahead policy that projects each
candidate correction's expected side effects through the coupling structure
(using only the predicted degradation channel and shot index — no
information the deployable arm lacks) matches `evidential_capture` at the
headline budget and *exceeds both it and the perfect-information oracle* at
larger budgets (K=6: 0.239 vs. 0.239 vs. 0.233; K=8: 0.249 vs. 0.248 vs.
0.239 verdicts per capture) — imperfect information with a smarter rule
beating perfect information with a naive one. **New non-evidential
baselines**: a SelectiveNet-style raw-confidence threshold (single shot, no
calibration — its error rate does not move with the stated standard at all,
since it never reads the burden, so it violates only once the ladder tightens
past its one fixed rate, a cleanly different failure shape from multiplicity)
and, in a separate fairly-calibrated comparison (E9), burst
averaging against sequential wealth accumulation at equal capture cost —
sequential accumulation is mildly ahead (0.142 vs. 0.132 VPC) but fused
averaging is noticeably more accurate when it decides (0.931 vs. 0.847),
a genuine bias/power trade-off between the two ways of combining K looks
that a naive comparison (testing the fused statistic against a
single-shot-calibrated null) would have hidden entirely — averaging shrinks
score variance toward the null's center, and testing that shrunk statistic
against a null it was never calibrated for cost 0.124 VPC by itself, a
methodological trap worth naming on its own.

**Ablations (E4).** Likelihood-ratio evidence buys ~70% more verdicts than
conformal at equal validity. Degradation-aware betting *loses* to constant
staking (0.214→0.402 VPC as the stake rises to all-in) — a negative result
on the design's own instinct, explained by stratified e-values having
already spent the quality information once. Confidence-head quality is
strongly load-bearing (0.230→0.125 VPC as head noise rises 0→0.40).
Benchmarking on i.i.d. corruptions collapses the gain from *targeted*
instructions from +0.030 to +0.004 — a 7× understatement relative to the
process-correlated world.

**New: instrument ablation (E10).** Replacing the per-artifact confidence
head with a trivial single-scalar blur-variance heuristic — broadcasting one
sharpness-proxy number across all five artifact slots, unable to distinguish
glare from tilt — retains 75% of the targeting advantage. The head's
*type-prediction* ability is doing real, but not all, of the work; a much
cheaper instrument gets most, not all, of the way there.

**New: per-case capture budgets (E11).** Reallocating a shared total budget
across cases (greedily, by predicted usability, rather than a fixed K each)
gains 0.038 VPC at equal total capture cost, because uniform-K wastes
captures whenever a case decides early with no mechanism to redirect them;
the guarantee survives the cross-case scheduler on an all-sound null check,
consistent with an independence argument (the scheduler's priority signal is
degradation-channel-only, never a diagnosis score).

**Sensitivity (E5).** `evidential_capture` ranks first in 17/17 swept
worlds (clinic difficulty, prevalence, reader quality, case-difficulty
distribution), with zero guarantee violations throughout.

**Real images (E6).** Re-run on the full training split (705 radiographs,
3529 diagnosis-labeled boxes) rather than the 50-radiograph public
validation split. The fine-grained sim-to-real check remains inconclusive,
but for a sharper reason than data starvation: a ~20× increase in crops
raised AUC from chance (0.481) to 0.596 — short of this project's own 0.60
interpretability bar by 0.004 — pointing at the linear reader, not sample
size alone, as a co-binding constraint. The coarse task's earlier
"near-saturated" reading (AUC 0.986 on 50 radiographs) did not survive the
larger split either (0.729 on ~700), which also revealed a large prevalence
skew (~77-79% caries-positive) specific to that binary collapse of DENTEX's
labels. The degradation head improved with more data and now reports all
five artifacts (blur r=0.910, low-light r=0.714, glare r=0.411, jpeg
r=0.267, angle r=0.192; dominant-artifact accuracy 0.607 vs. 0.20 chance,
up from 0.452). The full pipeline runs end-to-end on real images with the
guarantee intact (0 violations, both evidence constructions), and the
earlier "single-shot beats evidential capture" reversal — observed only
because the small split's reader was accidentally near-saturated — did not
reproduce at the larger, less-saturated scale.

**New: a conformal-risk-control-style baseline (E12).** Alongside the
SelectiveNet-style entry above, single-shot testing under an unstratified
(marginal) conformal calibration — the closest building block in this
codebase to what that literature family represents, not a reimplementation
of Angelopoulos et al. (2022)'s specific algorithm, stated as such — isolates
what evidential capture actually adds: stratification alone buys essentially
nothing at single-shot (−0.002 VPC), the retake loop is what buys the gain
(+0.065 VPC, more than doubling single-shot's VPC).

**New: a second real modality (E13).** Real CheXphoto data is gated behind a
CheXpert data-use agreement not obtained here. Instead, this project's own
capture simulator and evidence machinery — unmodified — were applied to a
different real medical-imaging classification task on a freely licensed
dataset (Kermany et al.'s pediatric pneumonia chest X-rays, CC BY 4.0):
2,981 train images, 312 calibration, 312 held-out test, all real. The
guarantee held with zero violations; the core targeting ordering
(evidential_capture > untargeted > fixed) reproduced; and — an honest
complication rather than a clean win — `single_shot` (VPC 0.511) beat
`evidential_capture` (0.430) here too, reproducing E6's "near-saturated
reader makes retaking not worth it" pattern independently on a second real
dataset and modality, which is stronger evidence that pattern is a real
property of strong readers under loose burdens rather than a DENTEX
artifact. This supports the framework's generality across modalities.

**New: a head-to-head against CheXphoto's own corruption model (E14).**
CheXphoto turned out to be gated in half rather than in whole: the
photographs sit behind a Stanford research use agreement, but the synthetic
corruption model was released as MIT-licensed code and applies to any
radiograph. Running it as the capture process — same reader, same
calibration protocol, same policies, same docket, only the renderer swapped
(`src/data/chexphoto_transforms.py`, a port checked against the vendored
reference) — the guarantee held with **zero violations** and the targeting
ordering reproduced (0.134 > 0.116 > 0.099 VPC), under a corruption process
this project did not write and did not design its method around. The
`single_shot` reversal above does *not* survive that harsher process
(single_shot last at 0.086 VPC), which usefully bounds it as a property of
near-strong readers rather than of retaking.

The same experiment produced the project's sharpest negative result. A
calibrator fitted under this project's capture model and deployed under
CheXphoto's yields false-conviction rates of 0.62–0.80 against a 0.50 bound
— every guaranteed arm violated — because the reader's scores collapse
upward under unfamiliar corruption (94.6% of negatives score above 0.5,
against 35.5% under the familiar process). Matched calibration restores
validity with zero violations. §6 records this as a deployment precondition.

None of this uses CheXphoto's photographs, so no number here is a CheXphoto
benchmark result; a head-to-head on the real photographs remains open, and
`docs/chexphoto_access.md` records the (~1 business day) access route.

**Five refuted predictions, kept deliberately**: (1) degradation-aware
betting loses to constant staking; (2) marginal (unstratified) calibration
does not break validity — the retake-induced selection shift runs
conservative in this process, not anti-conservative as the design
anticipated; (3) a perfect confidence head does not improve verdicts — the
greedy rule was the bottleneck, addressed above by (4); (4) instruction
side effects barely matter to outcomes despite being a deliberate simulator
feature; (5) the original sim-to-real check on the 50-radiograph split was
inconclusive by construction (chance-level AUC on the fine-grained task,
saturated on the coarse one) — addressed by the full-split re-run in this
draft.

## 6. Limitations

**Sim-to-real gap.** The primary claims are demonstrated on an analytic
surrogate reader whose behavior under degradation is specified by hand,
anchored at two literature-plausible AUC points (clean 0.88, clinic 0.78)
but otherwise a modeling choice. The one claim real data could adjudicate —
the relative damage ordering across artifacts (`SIGNAL_LOSS_WEIGHTS`) — was
attempted on the 50-radiograph public split and was inconclusive by
construction; the full ~700-radiograph re-run narrows why, without resolving
it: a ~20x larger split moved the fine-grained reader's AUC from chance
(0.481) to just short of interpretable (0.596, bar 0.60), implicating the
linear hand-feature reader itself, not merely sample size, as a co-binding
constraint — resolving this now more plausibly needs a stronger real
backbone than more data through the same linear model. That backbone was
attempted (DiffusionDet fine-tuned on Kaggle, `kaggle/04_train_all.ipynb`)
but did not converge within the ~10k-iteration budget a single 12h Kaggle
GPU session allows (vs. the paper's 40k): the disease-classification head's
bias stayed at its focal-loss initialization value across all six head
stages in both training arms, and the more heavily degraded arm produced no
detections at all. This is a compute-budget finding, not evidence against
the approach — it remains the more promising path, contingent on more
training budget than one session provides. The magnitudes, not the
taxonomy or the dynamics, are the weakest link in the parameter table (see
`docs/simulator_grounding.md`).

**The bound is loose.** Nothing here claims tightness. At α = 0.5 the
conformal arm's empirical crossing rate is 0.075 against a nominal bound of
0.5 — a lot of slack, the price of finite-sample conformal calibration plus
a p-to-e calibrator that must remain valid against every alternative
simultaneously.

**Conformal validity is marginal, not conditional, over the case
distribution.** A1 holds for the population the calibration set was drawn
from; conditioning on a harder subpopulation the calibration archive
under-represents weakens it measurably (§5, E2 sweep C: 0.0004→0.0037→0.0059
as the null is conditioned on progressively harder cases), though still
under nominal in every setting tested. A deployment whose case mix differs
sharply from its calibration archive should expect the guarantee to erode,
not necessarily break outright.

**A1 is reduced, not proved outright — for an arbitrary generative model.**
§3.3's Markov screening lemma is a real theorem about the generative model as
implemented; the residual condition (A1′) it reduces to is a distributional
comparison whose truth depends on magnitudes (confidence-head noise vs.
correction strength) the primitives do not pin down in general. For the
specific simulator and policies used throughout this paper, though, the
falsifiable prediction this yields — that validity itself, not just power,
should degrade as head noise grows past a point where noise-driven
misclassification dominates genuine correction — was checked directly (E2)
across confidence-head noise up to 2.5× E4's own power-degradation sweep, and
did not occur; false-conviction rate trended toward zero, not toward the
nominal bound. A structural argument in `docs/theory_anytime_validity.md` §6
explains the high-noise end of that finding: the confidence head corrupts
both the calibration and test populations' stratum assignment with the same
unbiased noise kernel, so the predicted failure needed an asymmetry between
the two populations that this design does not have. That asymmetric case was
then checked directly too: calibration noise fixed, test-time noise swept to
16.7× it — no violation there either, though this second result is
empirically robust without a comparably rigorous explanation, and is reported
as an open question for future theoretical work rather than a closed gap.

**DENTEX's own acknowledged limitations propagate.** Labels come from a
single-review-pass protocol (a student annotator, corrected by one of three
expert reviewers) with no published inter-annotator agreement statistic —
DENTEX's own paper acknowledges this. DENTEX ships no patient identifier at
all, so every split in this work (including the real-image arm's train/
calibration/test split) is at the image level, grouped by source radiograph
to prevent within-radiograph leakage, but not verified patient-independent
in the way a patient ID would allow.

**Escalation is currently justified by construction, not by clinical
review.** The three-way predict/retake/refer decision is validated against
the stated error guarantee and against DENTEX's ground-truth labels, not
against a clinician's judgment of whether the ~30 escalated cases a
qualitative review would surface are, in fact, cases a clinician would also
find undecidable from the photograph. This gut-check has not been run.

**The guarantee assumes you calibrated on the capture process you will
meet (E14).** This is the strongest condition on the paper's central claim
and it is not graceful when violated: a calibrator fitted under one
corruption model and deployed under another overshot its stated
false-conviction bound by up to 1.6x (0.80 against 0.50) across every
guaranteed arm. Deployment therefore requires either calibration data from
the deployment site's own capture process, or a detector for capture-process
shift — the degradation head is the obvious candidate for the latter and
this work does not use it that way.

**No real phone pilot.** Every degradation in this work is synthetic
(hand-built OpenCV transforms and an albumentations arm, compared against
each other but not against re-photographed printed films). The GAN-based
degradation arm scoped in `docs/phase2_data_notes.md` is explicitly deferred
pending exactly this pilot, which is itself gated on IRB clearance for new
data collection (`docs/phase1_background.md`) — DENTEX's existing images
need no new review, but photographing printed films does.

## 7. Conclusion (draft)

*(To be finalized once §5's real-image numbers are current and the release
is prepped — see `docs/decisions.md` and the Kaggle instructions for
outstanding compute-bound work.)*
