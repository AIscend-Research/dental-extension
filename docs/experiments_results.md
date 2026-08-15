# Experimental results

Every number below is reproduced by `.venv/bin/python -m experiments.run_all`
(~25-30 min, CPU only — grew from ~13 min once E6 was pointed at the full
~700-radiograph training split instead of the 50-radiograph validation split,
and E9-E13 were added). Raw output lives in `results/*.json` and
`results/*.csv`; figures in `figures/`. `experiments.run_all` also needs data
on disk for E6/E7 (DENTEX, see `docs/kaggle_instructions.md` step 0) and E13
(a small chest X-ray dataset, see `src/data/chest_xray_crops.py`'s
docstring); without it those experiments are skipped with a clear message,
everything else still runs.

**Figures.** The quantitative panels are `e1`&ndash;`e6`, plus `e13` (second
modality). E9-E12 (new in this update) write tables and JSON only
(`save_table`/`save_results`), no `figures/e9*.png` etc. -- table-sized
results, not enough independent sweep dimensions to earn a multi-panel
figure the way E3/E4 do. Three
qualitative figures on real radiographs come from E7 (`q1` capture session,
`q2` artifact atlas, `q3` glare geometry), and two hand-authored method
diagrams live as SVG (`diagram_firewall`, `diagram_verdict_loop`; rasterise
with `scripts/render_diagrams.py`). A browsable dossier of the original
eleven figures is published as an artifact &mdash; see the README (E9-E11
postdate that dossier and are not in it).

Operating regime for all simulated experiments: reader anchored at clean-image
AUC 0.88 and clinic AUC 0.776; clinic difficulty 0.5; prevalence 0.35;
calibration on 8000 first shots; 4 quantile strata.

**Headline burden** for policy comparisons is convict at *preponderance*
(α = 0.5), discharge at *clear and convincing* (α = 0.2) — a conviction only
means "refer for treatment", while discharging sends someone home untreated.
The library default is stricter; E3 sweeps the whole ladder.

---

## E1 — the simulator is a process, not a filter

| Claim | Measurement |
|---|---|
| The scene persists between shots | lag-1 correlation **0.963** within session vs **0.011** shuffled across sessions |
| Targeted instructions work; untargeted ones do not | targeted Δ **−0.073 to −0.125** on the named factor; untargeted **≈ 0.000** |
| Instructions are followed imperfectly | compliance **0.66–0.68** |
| Fixing one fault disturbs others | coupling recovered with the specified structure; largest leak glare→darkness **+0.022**, glare→tilt **+0.019** |
| Captures are non-exchangeable | permutation test **p = 0.0005**; usability rises **+0.110** from first shot to last |
| Retaking tires the operator | tremor **0.316 → 0.407** over 8 shots *while glare is being fixed* |

The exchangeability rejection is the one that matters downstream: it is the
assumption standard split-conformal rests on, and it is false here by
construction.

## E2 — anytime-validity under adaptive retaking

False-conviction rate on genuinely sound teeth, K = 4, 20 000 null sessions:

| nominal α | stratified | likelihood-ratio | marginal | best_shot | greedy |
|---|---|---|---|---|---|
| 0.50 | 0.075 | 0.167 | 0.068 | **0.936** ✗ | 0.233 |
| 0.20 | 0.015 | 0.027 | 0.010 | **0.578** ✗ | 0.167 |
| 0.05 | 0.0005 | 0.0013 | 0.0001 | **0.192** ✗ | **0.116** ✗ |
| 0.01 | 0.0000 | 0.0000 | 0.0000 | **0.037** ✗ | **0.021** ✗ |

✗ = violation (Wilson lower bound above α).

The multiplicity signature is unmistakable in the budget sweep at α = 0.05:
`best_shot` climbs **0.056 → 0.336** as K goes 1 → 8, while the e-processes stay
flat at ≤ 0.008. More photographs cost an unsound method its error rate and cost
a sound one nothing.

**Where the naive methods break is where it matters.** At α = 0.5 neither
unsound arm registers a violation; both violate from α = 0.1 downward. Naive
retake loops look fine exactly until you demand a clinically serious standard
of proof.

**Scope (α = 0.05).** Conformal validity is marginal over the calibrated case
mix. Conditioning the null on harder subpopulations moves the crossing rate
0.0004 → 0.0037 → 0.0059: still under nominal, but degrading in the direction
that would eventually break it.

**Refuted design hypothesis.** The marginal (unstratified) calibrator was
expected to be anti-conservative, since the retake loop shifts which shots get
tested. It is not — the shift runs conservative here, because retakes yield
cleaner shots and a sound tooth photographed cleanly scores low against a pool
of dirtier shots. Stratification's case is weaker assumptions and a little
power, not rescued coverage.

## E3 — The Docket leaderboard

n = 4000 cases, K = 4, likelihood-ratio evidence. Updated 2026-08-14 with two
new arms (see below); the seven original arms are numerically unchanged from
the first run (same seeds), confirming the additions didn't disturb anything:

| policy | VPC | decided | accuracy | escalated | shots | status |
|---|---|---|---|---|---|---|
| greedy_diagnostic | 0.460 | 97.8% | 0.819 | 2.2% | 2.12 | **unguaranteed** |
| confidence_threshold_selective | 0.300 | 30.0% | 0.877 | 70.0% | 1.00 | **unguaranteed** |
| **evidential_capture** | **0.225** | 68.2% | 0.842 | 31.8% | 3.03 | ok |
| one_step_lookahead | 0.223 | 67.6% | 0.841 | 32.4% | 3.03 | ok |
| oracle_instruction | 0.219 | 66.4% | 0.843 | 33.6% | 3.04 | ok |
| untargeted_evidential | 0.190 | 59.2% | 0.839 | 40.8% | 3.12 | ok |
| fixed_retake | 0.145 | 58.0% | 0.853 | 42.0% | 4.00 | ok |
| single_shot | 0.118 | 11.8% | 0.750 | 88.2% | 1.00 | ok |
| naive_best_shot | 0.052 | 20.6% | 0.927 | 79.3% | 4.00 | **unguaranteed** |

Among sound arms the ordering is clean: **targeted evidential capture beats
untargeted by 18%, fixed bursts by 55%, and single-shot by 91%**. The unsound
arm wins on raw VPC and pays for it in accuracy (0.819 vs 0.842) and in the
guarantee it no longer has. `confidence_threshold_selective` (SelectiveNet-
style: threshold the raw score, no calibration, no retake — see "New
baselines" below) sits between the two: higher VPC than any guaranteed arm at
K=4, but unguaranteed, and less accurate on what it does decide than the
methods that bother to calibrate.

**A perfect confidence head buys nothing — the rule was the bottleneck, and
fixing the rule closes the gap.** `oracle_instruction` reads the true latent
scene and still does not beat `evidential_capture`; both apply the same
greedy "fix the worst factor" rule, just with different information, so
neither improving nor perfecting the confidence head was ever going to help.
**`one_step_lookahead`** (new) tests the actual prediction this implies: same
information as the deployable arm (predicted degradation and shot index
only, never the true scene), but a decision rule that projects each
candidate correction's side effects through the coupling structure
(`src/sim/lookahead.py`) before choosing what to ask for. At the headline
budget it ties `evidential_capture` and *beats* `oracle_instruction` (0.223
vs 0.219) — imperfect information with a smarter rule already edges out
perfect information with the naive one. The budget sweep makes this sharper:

| K | evidential_capture | oracle_instruction | one_step_lookahead |
|---|---|---|---|
| 1 | 0.117 | 0.117 | 0.117 |
| 2 | 0.172 | 0.170 | 0.172 |
| 3 | 0.198 | 0.199 | **0.201** |
| 4 | 0.219 | 0.216 | **0.220** |
| 6 | 0.239 | 0.233 | **0.239** |
| 8 | 0.248 | 0.239 | **0.249** |

From K=3 onward, `one_step_lookahead` matches or exceeds *both* other arms,
including the perfect-information oracle. A null result about the confidence
head turns out to be a real result about the decision rule.

**New baselines: the benchmark now carries entries that aren't the authors'
own methods.** `confidence_threshold_selective` (`src/bench/policies.py`) is
a SelectiveNet-style [Geifman & El-Yaniv 2019] fixed-threshold selector: one
shot, no calibration, predict/discharge/refer by how far the raw score sits
from 0.5. It is closer to the *softmax-response* baseline SelectiveNet's own
paper beats than to SelectiveNet itself (no jointly trained selection head),
which is an honest limitation of this addition, not a claim to soften — see
`docs/phase1_background.md`'s full read of that paper. It carries no
guarantee (`peeks=True`: thresholding the raw diagnosis score directly is,
definitionally, conditioning on the quantity being tested) and none was
found to be egregiously violated at this headline (loose) burden — consistent
with E2's finding that naive methods look fine at loose standards and fail
at strict ones, not yet checked here at a strict standard for this specific
arm.

**Budget.** Evidential capture rises monotonically with K (0.117 → 0.248 from
K=1 to K=8); `fixed_retake` *peaks at K=3 and then declines*, because a policy
that cannot stop early spends captures it does not need.

**The ladder — where a phone stops being admissible:**

| standard | VPC | decided | accuracy | escalated |
|---|---|---|---|---|
| preponderance | 0.346 | 85.0% | 0.847 | 14.9% |
| headline | 0.219 | 66.5% | 0.831 | 33.5% |
| clear and convincing | 0.136 | 48.1% | 0.951 | 51.9% |
| clear+conv / beyond doubt | 0.069 | 26.4% | 0.939 | 73.6% |
| beyond reasonable doubt | 0.033 | 12.8% | 0.981 | 87.1% |
| near certainty | 0.003 | 1.2% | 1.000 | 98.8% |

At a criminal standard of proof, a smartphone reader at clinic AUC 0.78 can
decide **12.8%** of cases within four photographs and must escalate the rest.
That is a property of the reader and the capture process, not of the policy,
and it is the number a deployment decision should start from.

## E4 — ablations

| ablation | finding |
|---|---|
| **Evidence construction** | likelihood-ratio **0.214** vs conformal **0.125** — the LR route buys ~70% more verdicts per capture. Marginal (0.127) ≈ stratified: stratification costs nothing and gains little *here*. |
| **Betting strategy** | **Negative result on our own design.** Constant staking beats degradation-aware: 0.214 (adaptive) → 0.280 (λ=0.5) → 0.354 (λ=0.8) → 0.402 (all-in). Stratified e-values have *already* conditioned on image quality, so shrinking the stake discounts the same evidence twice. λ≈0.8 is the recommended default: all-in also drops accuracy (0.804 vs 0.826), because one near-zero e-value destroys the wealth permanently. |
| **Confidence-head quality** | Strongly load-bearing: VPC **0.230** (oracle head) → **0.125** (noise 0.40). Everything keys off the degradation channel. |
| **Instruction side effects** | Nearly irrelevant to outcomes: 0.218 (off) → 0.211 (double strength). The mechanism is real in the simulator but not driving the results. |
| **i.i.d. vs process** | The gain from *targeted* instructions collapses from **+0.030** to **+0.004** when the scene is redrawn every shot. An ImageNet-C-style i.i.d. benchmark understates targeted retaking by ~7×. |
| **Stratum count** | Flat from 1 to 16 strata (0.207–0.216). |

The i.i.d. row required fixing a broken ablation: setting `persistence = 0`
does *not* produce an i.i.d. world, because the per-session equilibrium
survives and the session stays correlated. Only redrawing the whole scene per
shot does, and the ρ sweep is reported alongside precisely to show it is flat
(0.210 → 0.215) and that the i.i.d. arm is doing the work.

## E5 — sensitivity across 17 worlds

Sweeping clinic difficulty (0.2–0.8), prevalence (0.10–0.50), reader quality
(clinic AUC 0.70–0.84) and case-difficulty distribution:

- **evidential_capture ranked first in 17/17 worlds.**
- **0 guarantee violations in 17/17.**
- Margin over untargeted: min **+0.026**, median **+0.029**, max **+0.039**.
- Full ordering preserved in 15/17; both exceptions are `single_shot` and
  `fixed_retake` — the two *weakest* arms — swapping in easy worlds, where
  spending four captures is simply waste.

## E6 — the real-image arm (DENTEX)

**Updated 2026-08-14: re-run on the full ~700-radiograph training split**
(705 radiographs, 3529 diagnosis-labelled boxes) instead of the 50-radiograph
public validation split the first pass was limited to. This is a ~20x
increase in the fine-grained task's crop count (133 → 2767) and ~19x in the
coarse task's (182 → 3529). The 50-radiograph numbers are kept below for
comparison; they are no longer the current result.

**The sim-to-real check is still inconclusive — but the reason changed, and
that is itself a finding.** More data moved the fine-grained AUC from
chance to just short of interpretable, not past it:

- *Caries vs Deep Caries* (fine-grained, 2767 teeth from 650 radiographs, 5-fold
  grouped CV): **AUC 0.596**, up from 0.481 on the 50-radiograph split but
  still under this project's own `MIN_INTERPRETABLE_AUC = 0.60` bar — by
  0.004. E6 still flags it inconclusive in code, correctly: the bar exists
  precisely so a result this close to it doesn't get reported as resolved by
  a rounding error.
- **This reframes the original diagnosis.** A ~20x data increase produced a
  real but small AUC gain (0.481 → 0.596), not the jump to a clearly
  interpretable reader a pure data-starvation story would predict. That
  points at the *linear model over hand-designed features*
  (`src/models/real_channel.py`, chosen deliberately for low variance on a
  small split — see that module's docstring) as a co-binding constraint
  alongside sample size, not sample size alone. A stronger real backbone
  (the Kaggle-trained detector, once it exists) is now the more promising
  path to resolving this than more data through this same linear reader
  would be.
- *Caries+Deep vs Impacted+Periapical* (coarse, 3529 teeth from 705
  radiographs): **AUC 0.729**, down sharply from 0.986 on the small split.
  The full split's class balance is also far more skewed on this task
  (77–79% caries-positive across train/calibration/test, vs the ~35%
  prevalence the simulated Docket assumes) — a real, DENTEX-specific
  scope note for any claim that reuses this task's numbers. The previous
  "near-saturated" characterisation was itself partly an artifact of the
  small split, not a property of the task.
- Measured damage ordering (still not read as a finding, per the AUC gate):
  low_light > glare > angle > blur > jpeg, vs. the surrogate's assumed
  blur > glare > jpeg > low_light > angle (Spearman −0.30). Reported for the
  record, not as evidence either way — a chance-adjacent reader's damage
  curve is not evidence about which artifacts matter, regardless of which
  direction it points.

So `SIGNAL_LOSS_WEIGHTS` remains a stated assumption, now on stronger
grounds for *why* it can't yet be resolved this way: the fine-grained task
needs a real detector-quality reader, not just more crops through a linear
one.

**What did work on real images, and improved with more data:**

- The **degradation head trained by weak supervision transfers, and reading
  five artifacts (not three) is now reported**: predicted vs true severity
  correlates r = **0.910** (blur, was 0.887), **0.714** (low light, was
  0.681), **0.411** (glare, was 0.337), **0.267** (jpeg, not previously
  reported), **0.192** (angle, not previously reported). Dominant-artifact
  accuracy **0.607** against a 0.20 chance baseline, up from 0.452 — the
  clearest evidence in this update that more data helps this channel
  specifically, even where it didn't resolve the diagnosis channel's AUC.
- The **full pipeline runs end to end on real radiographs** and the guarantee
  holds (0 violations across all arms, both evidence constructions).

**The "instructive reversal" from the 50-radiograph split did not
reproduce.** Previously, on the near-saturated small-split reader,
`single_shot` (VPC 0.392) beat `evidential_capture` (0.373) — one photo
already sufficed, so retaking was pure cost. On the full split, with the
coarse task's AUC down to 0.729 (no longer near-saturated), that ordering
flips back: `evidential_capture` (VPC 0.454) edges past `single_shot`
(0.448) under likelihood-ratio evidence. The qualitative lesson survives
even though the specific numbers didn't: retaking earns its keep when the
reader is imperfect, and stops earning it as the reader approaches
saturation — this dataset moved between those two regimes depending on
split size, which is itself a caution about reading too much into any one
fixed-size real-data result.

## E9 — burst fusion vs sequential accumulation, calibrated fairly

`src/models/fusion.py` (learned attention-weighted burst merge over detector
features) is built and tested but unused end to end — the project's actual
answer to "how do you combine K looks" is the wealth process: bet on each
shot, multiply. This experiment makes that comparison explicit rather than
implicit, at the level the framework actually operates (fuse the two channel
readings, not raw pixels — a trained `BurstFusion` needs real detector
features from Kaggle and is a separate, deferred comparison).

n = 4000, K = 4, both arms untargeted and sound:

| policy | VPC | decided | accuracy |
|---|---|---|---|
| fixed_retake (sequential) | 0.142 | 56.7% | 0.847 |
| burst_fusion (own null) | 0.132 | 52.7% | **0.931** |
| burst_fusion (mismatched null) | 0.008 | 3.1% | 0.992 |

**A methodological trap worth naming on its own.** Averaging K scores
shrinks their variance toward the null's center (regression toward the
mean), so testing that shrunk statistic against a calibration pool fit on
*single*-shot scores — the naive way to run this comparison — cripples the
fused arm for reasons that have nothing to do with whether fusion is a good
idea: VPC 0.008 vs. 0.132 once a dedicated null is fit for the actual
K-shot-averaged statistic (`collect_burst_calibration`,
`experiments/e9_burst_vs_sequential.py`). That is a 0.124 VPC swing from a
calibration bug that would be silent, not a crash, in a less careful
implementation.

**Fairly calibrated, sequential accumulation is mildly ahead on verdicts per
capture, but fusion is meaningfully more accurate when it decides**
(0.931 vs 0.847). Sequential wealth accumulation extracts slightly more
*decisions* from the same K shots; averaging first extracts a more reliable
signal per decision it does make. Both are sound and spend identical
capture cost — this is a real bias/power trade-off between two ways of
combining evidence, not a case where one dominates the other.

## E10 — instrument ablation: a trivial blur-variance heuristic

Every result elsewhere reads the degradation channel through a noisy but
*per-artifact* estimator — a stand-in for a learned confidence head that can
tell blur from glare from tilt. This ablation swaps it for `BlurVarianceChannel`
(`experiments/e10_instrument_ablation.py`): one scalar, a proxy for what a
classical Laplacian-variance sharpness detector would give, driven only by
the two artifacts that destroy high-frequency content (blur, jpeg), reported
identically across all five artifact slots — the instrument a team would
reach for before building a learned head at all.

n = 4000, K = 4, targeting gain = evidential_capture VPC − untargeted VPC:

| instrument | targeting gain |
|---|---|
| full per-artifact confidence head | +0.0371 |
| blur-variance heuristic (one scalar, broadcast) | +0.0277 |
| **retained fraction** | **74.6%** |

**Ordering partially survives.** Targeting still helps with a scalar-only
instrument — it is not the case that per-artifact type prediction is the
*only* thing making targeted retaking work — but noticeably less than with
real type prediction. The confidence head's ability to distinguish *which*
artifact dominates is doing real, but not all, of the targeting advantage;
roughly a quarter of it depends on knowing which artifact is present, not
merely that the shot is degraded.

## E11 — per-case capture budgets under a shared cost

Every other experiment gives each case the same budget K. This asks whether
that is wasteful: a case whose first photo is already excellent spends the
same allowance as one that needs every retake.
`src/bench/allocation.py::run_docket_global_budget` spends the identical
*total* number of photographs (N × K, equal cost to the uniform baseline)
but greedily hands the next shot to whichever pending case's most recent
photo looks most usable.

n = 4000, K = 4 (16,000 total photographs either way):

| allocation | VPC | decided | accuracy | mean shots |
|---|---|---|---|---|
| shared budget (adaptive) | **0.254** | **100.0%** | 0.843 | 3.93 |
| uniform K=4 | 0.216 | 66.3% | 0.844 | 3.07 |

**Reallocating helps: +0.038 VPC (+18% relative) at equal total cost.**
The mechanism is not subtle once stated: uniform K *wastes* captures
whenever a case decides early — with nobody else able to use them — while
adaptive allocation redirects those unused captures to cases that are still
pending, up to a per-case cap (4× the uniform K). At this headline burden
(loose enough that most cases eventually cross one threshold given enough
shots), that redirection is enough to decide every single case in the
docket, at the cost of some hard cases consuming up to 16 shots each.

## E12 — a conformal-risk-control-style baseline

The second literature baseline the roadmap asked for, alongside E3's
SelectiveNet-style entry. Scoping note first: conformal risk control proper
[Angelopoulos, Bates, Fisch, Lei, Schuster 2022] is a specific algorithm for
calibrating a threshold against an arbitrary bounded loss; reimplementing it
faithfully was out of scope here. What this experiment adds is the closest
existing building block to what that family of methods represents in this
codebase: a single-look, unstratified, distribution-free-calibrated decision
with no retake (`single_shot` under `MarginalCalibrator`).

| policy | VPC | decided | accuracy |
|---|---|---|---|
| evidential_capture (stratified, retake) | 0.125 | 43.2% | 0.897 |
| conformal_risk_control_style (single-shot, marginal) | 0.061 | 6.2% | 0.833 |
| single_shot (stratified) | 0.060 | 6.0% | 0.816 |

Stratification alone buys essentially nothing at single-shot (−0.002 VPC,
consistent with E4's finding that stratified ≈ marginal); the retake loop is
what buys the gain (+0.065 VPC, more than double the single-shot VPC). This
isolates what evidential capture actually adds over a conformal-style
single-look baseline: not the stratification, the sequential evidence
accumulation.

**Validity survives the shared scheduler.** On a 20,000-case all-sound null
docket under the same shared-budget scheme, false-conviction rate stayed at
0.156 against a nominal α of 0.50 (symmetric preponderance burden) — no
violation. `src/bench/allocation.py`'s docstring argues this from
independence (the scheduler's priority signal is predicted usability,
never a diagnosis score, and cases are independent processes), and this is
the empirical check of that argument, in the same spirit E2 checks the
single-case guarantee.

## E13 — a second real modality: chest radiographs, not teeth

"If the guarantee and the ordering reproduce in a second modality, the claim
stops being about teeth." Real CheXphoto data (natural photos of CheXpert
x-rays, plus CheXphoto's own synthetic photographic transforms) is gated
behind a CheXpert data-use agreement, not obtainable in this environment.
**Scoping, stated plainly**: this experiment does not reproduce CheXphoto.
It applies this project's own capture simulator and evidence machinery,
completely unmodified, to a different real medical-imaging classification
task on a freely licensed dataset — Kermany et al.'s pediatric pneumonia
chest X-ray dataset (CC BY 4.0, no DUA, `src/data/chest_xray_crops.py`,
`experiments/e13_second_modality.py`). It tests whether the framework
generalizes past dental radiographs to a second real modality; it does not
test CheXphoto's specific corruption model or capture rig, neither of which
were obtained.

2,981 train images (1,341 normal / 1,640 pneumonia), 312 calibration, 312
held-out test, all real, no synthetic labels:

- **Clean AUC 0.899** — well above the 0.60 interpretability bar this time,
  unlike the dental fine-grained task. The damage curve is a real finding
  here, not flagged inconclusive.
- **Damage ordering: glare > angle > blur > low_light > jpeg** (worst
  first), measured, not assumed. This differs from the dental
  `SIGNAL_LOSS_WEIGHTS` ordering (blur > glare > jpeg > low_light > angle)
  — expected and not a contradiction of anything: these are two different
  readers on two different modalities, and the dental weights were never
  claimed to transfer. What this *does* establish is that the damage-curve
  machinery itself produces a real, differentiated, non-degenerate ordering
  when given an interpretable reader — the dental task's inconclusiveness
  (E6) is a property of that task's difficulty and split size, not a defect
  in the method.
- **The guarantee holds: zero violations among guaranteed arms.**
- **The targeting ordering partially reproduces, with an honest
  complication.** Among the untargeted-vs-targeted-vs-fixed comparison,
  `evidential_capture` (VPC 0.430) beats `untargeted_evidential` (0.404)
  beats `fixed_retake` (0.207) — the core ordering holds. But
  `single_shot` (VPC 0.511) beats all three, including
  `evidential_capture`. This is not a new failure mode — it is the same
  "instructive reversal" E6 first found on its small split and lost on the
  full-split re-run (above): when the reader is strong enough (AUC 0.899
  here, similarly high on E6's small split), one photograph already meets
  a loose burden most of the time, and retaking is close to pure cost. That
  this exact pattern shows up independently on a second real dataset,
  rather than only on DENTEX's small split, is evidence it is a real
  property of near-strong readers under a loose burden, not a one-off
  artifact of one dataset.

**What this experiment supports and what it doesn't.** Supports: the
validity guarantee is not dental-specific — it held with zero violations on
a completely different real classification task. Supports: the "retaking
helps least when the reader is already strong" pattern is real and
reproduces independently. Does not support: any claim about CheXphoto
specifically, or about photographic (as opposed to this project's five-
artifact) corruption models on chest radiographs — a genuine CheXphoto
head-to-head remains open, gated on CheXpert data access.

---

## Summary of negative and refuted results

Kept deliberately, because they were predictions this design got wrong —
updated 2026-08-14: one of them (#3) now has a positive follow-up, and #5 was
re-run at ~20x the data rather than left as originally recorded.

1. **Degradation-aware betting loses to constant staking** (E4). The
   information is already spent in the stratified e-value.
2. **Marginal calibration does not break validity** (E2). The selection shift
   runs conservative in this capture process.
3. **A perfect confidence head does not improve verdicts** (E3) — **and the
   follow-up turns this into a contribution**: a one-step-lookahead subpoena
   rule, using only the same information the deployable arm already has,
   fixes the actual bottleneck (the greedy "fix the worst factor" rule
   ignoring correction side effects) and matches or beats the
   perfect-information oracle from K=3 upward (E3 update, above).
4. **Instruction side effects barely matter** to outcomes (E4), despite being
   a deliberate feature of the simulator.
5. **The sim-to-real check is still unresolved on the fine-grained task, but
   the diagnosis changed** (E6, re-run on ~700 radiographs instead of 50):
   more data raised AUC from chance (0.481) to just short of interpretable
   (0.596, bar is 0.60) — a real but insufficient gain that points at the
   linear reader as a co-binding constraint alongside sample size, not
   sample size alone. The coarse task's "near-saturated" characterisation
   (AUC 0.986 on 50 radiographs) also did not survive the larger split
   (0.729 on ~700) — partly an artifact of the earlier split's size, not a
   stable property of the task.

Two further findings from this update, not corrections of earlier ones but
in the same spirit — a comparison that could have looked clean came with a
trap, and an ablation gave a split verdict rather than a clean one:

6. **Comparing burst fusion to sequential accumulation naively is a
   calibration bug, not a fair test** (E9): testing an averaged K-shot
   statistic against a null fit to single shots cost 0.124 VPC by itself,
   before any real conclusion about fusion vs. sequential accumulation could
   be drawn.
7. **A dumb instrument keeps ~75% of the targeting benefit, not all of it
   and not none of it** (E10): the confidence head's ability to name *which*
   artifact dominates is real and load-bearing, but a scalar-only heuristic
   is not useless the way a clean "the head is essential" story would want.
8. **The "near-saturated reader makes retaking not worth it" pattern is
   real, not a DENTEX artifact** (E13): it reproduced independently on a
   second real dataset and a second modality (chest radiographs, unrelated
   reader, unrelated prevalence), which is stronger evidence for it being a
   genuine property of strong readers under loose burdens than either single
   observation was alone.
