# Anytime-valid burdens of proof under adaptive retaking

This note states the guarantee the framework claims, proves it, and — more
important for a paper — draws the line around what it does *not* cover. The
empirical counterpart is `experiments/e2_validity.py`; the executable
counterpart is `tests/test_evidence.py`.

## 1. The problem

A capture policy may keep asking for another photograph until it is satisfied.
That is *optional stopping*, and it breaks the two tools one would reach for
first.

**Fixed-level testing breaks by multiplicity.** Taking `K` looks at the same
null and reporting the most favourable one inflates the error rate to roughly
`1 − (1 − α)^K`. E2 measures this directly: the `best_shot` arm's
false-conviction rate climbs from 0.056 at `K = 1` to 0.336 at `K = 8` against
a nominal α of 0.05.

**Split conformal breaks by non-exchangeability.** Conformal prediction needs
the test point to be exchangeable with the calibration set. Here it is not, and
not by accident: shot 2 exists *because* shot 1 had glare, so the shots within
a session are correlated through the room and the hands, and shifted by the
instruction shot 1 triggered. E1 rejects exchangeability with a permutation
test at p = 0.0005, with mean usability rising 0.110 from first shot to last.

There is a third, subtler failure that is the one worth naming. If the decision
to stop depends on the diagnosis score itself — "retake while the model is
unsure" — then the sequence of shots is *selected on the very quantity being
tested*. E2's `greedy` arm shows this costs 0.116 false convictions at α = 0.05,
independently of multiplicity.

## 2. Setup

One case is one tooth with a fixed, unknown label `Y ∈ {0, 1}`. Retaking
changes the evidence, never the truth.

Shot `t` produces two readings:

- `D_t` — the **degradation channel**: predicted per-artifact severities and a
  scalar usability, from the confidence head.
- `S_t` — the **diagnosis channel**: a caries score.

Define the filtration

$$\mathcal{G}_t = \sigma\big(D_1, \dots, D_t,\; S_1, \dots, S_{t-1}\big).$$

Note the asymmetry, which is the whole design: `G_t` contains the degradation
reading of the *current* shot but only the diagnosis scores of *previous* ones.
The current shot's photograph may be inspected for quality before anyone bets
on what it says about disease.

Let `Z_t = ζ(D_t) ∈ {1, …, m}` be the calibration stratum, a deterministic
function of `D_t` and therefore `G_t`-measurable.

## 3. Assumptions

**(A1) Conditional calibration.** Under the null `H₀: Y = 0`, the conformal
p-value `P_t` computed within stratum `Z_t` is conditionally super-uniform:

$$\Pr\big(P_t \le u \mid \mathcal{G}_t\big) \le u \quad \text{for all } u \in [0,1].$$

This is the substantive assumption, and it is where stratification earns its
place. Split conformal gives marginal super-uniformity when the test point is
exchangeable with the calibration pool; conditioning on `Z_t` replaces
"exchangeable with the pool" by "exchangeable with the pool *of the same
stratum*", which the retake loop does not disturb, because the stratum is
precisely what the loop selects on. It is an assumption, not a theorem: the
loop also conditions on past scores and on accumulated wealth, and any residual
dependence between `S_t` and that history *within* a stratum is not ruled out
by the construction. §6 reports what happens when it is stressed.

**(A2) Predictable stakes.** The betting fraction `λ_t ∈ [0,1]` is
`G_t`-measurable.

**(A3) Predictable stopping.** The stopping time `τ` and the choice of
instruction are `G_t`-measurable.

A2 and A3 are not assumptions about the world; they are constraints on the
policy, and the code enforces them structurally rather than by convention. A
bettor is handed an `EvidenceView` (`src/evidence/view.py`) which does not
carry the current diagnosis score, so a policy satisfying A2–A3 is one that
type-checks. Policies that want to peek must accept a `PeekingView`, and every
result they produce is reported as unguaranteed.
`tests/test_evidence.py::test_stake_is_independent_of_the_diagnosis_score`
checks the firewall holds by construction.

## 4. The wealth process

**Grounding, read in full rather than cited secondhand (2026-08-14).** Two
papers this section leans on:

- **Vovk & Wang (2021), *E-values: calibration, combination, and
  applications*** [arXiv:1912.06116]. Their Proposition 2.1 characterizes
  admissible p-to-e calibrators exactly: a decreasing `f: [0,1] → [0,∞]` is
  a calibrator iff `∫f ≤ 1`, and admissible iff upper semicontinuous,
  `f(0) = ∞`, and `∫f = 1`. The family `f_κ(p) = κp^{κ−1}` (their eq. 1) is
  exactly this construction's calibrator, and satisfies all three
  conditions for every `κ ∈ (0,1)` — so "admissible p-to-e calibrator" below
  is not an assertion, it is this proposition applied to a specific `f_κ`.
  Their Remark 2.3 is also the precise statement behind "the p-to-e route is
  lossy" (§5 below): the reciprocal of a p-value is only *approximately* an
  e-value, `f(p) = 1/p` is achievable only in a degenerate limiting case, and
  every genuine calibrator pays for validity against every alternative
  simultaneously.
- **Ramdas, Ruf, Larsson & Koolen (2022), *Admissible anytime-valid
  sequential inference must rely on nonnegative martingales***
  [arXiv:2009.03167]. Their central result is stronger than "nonnegative
  supermartingales are *a* valid way to get anytime-validity" — they prove
  nonnegative (super)martingales are *necessary*: every admissible p-process,
  e-process, sequential test, or confidence sequence can be recovered or
  dominated by one built from a nonnegative martingale. This upgrades the
  wealth-process construction below from "a reasonable choice among several"
  to "the general form any admissible construction must reduce to" — the
  choice of framework in this paper is not a design decision competing with
  alternatives so much as an instance of the only structure that works.

Let `κ` be an admissible p-to-e calibrator — this implementation uses
Vovk–Wang, `e = κ p^{κ−1}` with `κ = 1/2`, for which `E[e] = 1` when `p` is
uniform. Set `E_t = κ(P_t)` and

$$W_0 = 1, \qquad W_t = W_{t-1}\big(1 + \lambda_t (E_t - 1)\big).$$

**Lemma (nonnegativity).** `E_t ≥ 0` and `λ_t ∈ [0,1]` give
`1 + λ_t(E_t − 1) ≥ 1 − λ_t ≥ 0`, so `W_t ≥ 0` for all `t`.

**Lemma (supermartingale).** By A2, `λ_t` is `G_t`-measurable, so

$$\mathbb{E}[W_t \mid \mathcal{G}_t] = W_{t-1}\big(1 + \lambda_t(\mathbb{E}[E_t \mid \mathcal{G}_t] - 1)\big) \le W_{t-1},$$

since A1 makes `P_t` conditionally super-uniform and hence
`E[E_t | G_t] ≤ 1`. Note `W_{t-1}` is `G_t`-measurable, which is why the
current shot's degradation reading being in `G_t` costs nothing.

**Theorem (anytime-valid burden of proof).** Under A1–A3, for any stopping
time `τ` and any `α ∈ (0,1)`,

$$\Pr_{H_0}\Big(\exists\, t \le \tau : W_t \ge 1/\alpha\Big) \le \alpha.$$

*Proof.* `(W_t)` is a nonnegative supermartingale with `W_0 = 1` by the two
lemmas. Ville's inequality gives `Pr(sup_t W_t ≥ 1/α) ≤ E[W_0] α = α`, and the
event in question is contained in that supremum event. ∎

Three consequences worth spelling out:

1. **The retake loop is free.** `τ` may depend on anything in `G_t` — wealth so
   far, predicted degradation, how tired the operator looks. No correction for
   the number of looks is needed, because none of them were free looks.
2. **A verdict, once earned, stays earned.** The bound is on the *running
   maximum*, so a verdict rendered at shot 2 remains legitimate even if shot 3
   would have dragged the wealth back down. The implementation checks
   `running_max`, not current wealth (`EWealth.has_crossed`).
3. **The ladder is calibrated, not ordinal.** Each rung's threshold is `1/α`,
   so "beyond reasonable doubt" names an error rate rather than a vibe.

The two-sided machine runs a second, independent process against
`H₀: Y = 1`, at its own α. When both cross, the evidence is self-contradictory
and the machine escalates rather than breaking the tie — a tiebreak is not
proof.

## 5. The likelihood-ratio construction

The p-to-e route is exactly valid and *lossy*: a calibrator must be valid
against every alternative simultaneously, so it cannot use the fact that we
have a labelled calibration set of what caries scores look like. The
alternative is the canonical e-value, the density ratio

$$E_t = \frac{f_1(S_t \mid Z_t)}{f_0(S_t \mid Z_t)}, \qquad \mathbb{E}_{H_0}[E_t \mid \mathcal{G}_t] = \int f_1 = 1,$$

exact for the true densities and most powerful by Neyman–Pearson. In practice
the densities are estimated (smoothed histograms on quantile bins within a
stratum), so **validity here is empirical, not proved**, and the honest thing
is to measure it rather than assert it. E2 does: across the α grid the LR arm
sits below nominal everywhere (0.167 at α = 0.5, 0.0013 at α = 0.05), and
`tests/test_evidence.py` checks `E[e] ≈ 1` on held-out nulls. It buys roughly a
70% increase in verdicts per capture over the conformal route (E4, 0.214 vs
0.125). Use conformal when the guarantee must be provable and LR when power
matters; the benchmark reports both, always side by side.

## 6. Scope, and where it gives way

**Conformal validity is marginal over the case distribution.** A1 holds for the
population the calibration set was drawn from. Conditioning on a subpopulation
the calibration set under-represents weakens it. E2's sweep C conditions the
null on progressively harder cases: the crossing rate rises from 0.0004
(matched) to 0.0037 (hard) to 0.0059 (hardest) at α = 0.05. Still under
nominal — but moving in the direction that would eventually break it, and a
deployment whose case mix differs sharply from its calibration archive should
expect the guarantee to erode.

**The bound is loose.** Nothing here claims tightness, and the measurements
show a lot of slack: at α = 0.5 the conformal arm's crossing rate is 0.075.
The slack is the price of finite-sample conformal plus a calibrator that must
work against every alternative, and it is why the LR construction exists.

**Ignoring the degradation channel turned out not to break validity.** The
design anticipated that a marginal (unstratified) calibrator would be
anti-conservative, because the retake loop shifts which shots get tested.
E2 refutes this: the shift runs *conservative* here, since retakes produce
cleaner shots and a sound tooth photographed cleanly scores low against a pool
containing dirtier shots. Marginal calibration is therefore safe in this
capture process and merely a little less powerful. That direction is a property
of this process, not a theorem — a process whose retakes made images worse
would flip it — so stratification remains the defensible default, on the
grounds of weaker assumptions rather than rescued coverage.

**What is assumed, not proved, about A1 -- and how far a primitive-level
argument actually gets.** The construction gives conditional validity given
the stratum; it does not rule out residual dependence between `S_t` and the
session history within a stratum. This subsection does what §6's last
version flagged as not done: it derives A1 as far as the generative model's
own primitives take it, and shows precisely where the argument runs out and
an empirical claim has to take over -- rather than leaving the whole thing
as one undifferentiated assumption.

*Notation.* Let `R_t` be the true rendered severities at shot `t` (`SceneState.
severities()`), `Y` the case's true label (constant across the session), `δ`
the lesion difficulty, and `q_t = quality(R_t)` (`SurrogateChannel.quality`,
deterministic given `R_t`).

**Lemma (Markov screening).** Conditional on `(R_t, Y, δ)`, the pair
`(S_t, D_t)` is independent of `G_{t-1}`.

*Proof.* Read the generative code directly. `SurrogateChannel.read` sets
`mu = separation * (2Y-1) * (1-δ) * q_t^γ`, `sigma = sigma_base +
sigma_degraded*(1-q_t)`, and draws `S_t = sigmoid(Normal(mu, sigma))` with a
fresh `rng.normal` call; `_read_degradation` sets `D_t[name] = R_t[name] +
head_bias + Normal(0, head_noise)`, again with fresh, per-shot, per-name
noise. Neither formula references anything about the session besides `R_t`,
`Y`, `δ`, and noise drawn independently of every earlier draw. `G_{t-1}` is
measurable with respect to `(D_1..D_{t-1}, S_1..S_{t-1})`, all of which enter
`R_t`'s own generation (via which instruction the policy issued and how the
scene responded) but do not appear in `(S_t, D_t)`'s formulas once `R_t` is
fixed. ∎

This is a real theorem about the code as written, not an assumption -- and it
is also the precise sense in which `R_t` is a sufficient state for one shot's
readings: everything upstream of shot `t` matters only through what it did to
the scene, never directly.

**What the lemma buys, and what it does not.** Conditioning further on `R_t`
(not just `G_t`), the lemma gives `S_t | (G_t, R_t, Y{=}0) \sim f_0(\cdot \mid
q_t)`. But `A1` conditions on `G_t`, which observes only the *noisy* `D_t`,
not `R_t`. Marginalising `R_t` out,

$$\Pr(S_t \le s \mid G_t, Y{=}0) = \mathbb{E}\big[F_0(s \mid q_t) \;\big|\; G_t, Y{=}0\big],$$

an average over the conditional law of `q_t` given what `G_t` actually saw,
`Z_t = ζ(D_t) = z`. The calibration pool for stratum `z` was built the same
way from a *different* population: `collect_calibration` draws a fresh
session's FIRST shot each time, i.e. `R^{cal} \sim \pi_1` (`sample_initial_
scene`, never touched by an instruction), conditioned on landing in `z` via
its own noisy `D^{cal}`. Classical split-conformal validity within stratum
`z` needs the conformal p-value's calibration-pool comparison to be honest,
which reduces `A1` to exactly one condition:

> **(A1'), stratum-conditional stochastic dominance.** For every `z`, the law
> of `q_t` given `(Z_t{=}z, Y{=}0, G_t)` is stochastically dominated by (or
> equal to) the calibration law of `q^{cal}` given `(Z^{cal}{=}z, Y{=}0)`,
> *in the direction that lowers `S_t`* -- i.e. test-time true quality, given
> the same predicted stratum, is at least as good as calibration's.

This is strictly sharper than "A1 holds" -- it names the one inequality doing
all the work, and the Markov lemma is what licenses reducing the whole
filtration `G_t` down to a statement about `q_t \mid Z_t` in the first place.
It is also, honestly, still not a theorem: (A1') is a claim about how two
distributions compare, and nothing in the primitives forces it in general.

**A mechanistic account of when (A1') should hold, and when it should not.**
Two things can put a session in a good-looking stratum `z` at shot `t \ge 2`,
and they point in opposite directions:

1. *A genuine correction.* `CaptureSession._apply_instruction` moves the
   scene's *equilibrium*, not just the momentary value
   (`self.scene.equilibrium[target] = min(equilibrium[target], after + ...)`),
   so a session that reaches a clean-looking `Z_t` because an instruction was
   followed has a true `R_t` (and hence `q_t`) that is durably better than a
   first-shot draw from `π_1` could be -- `π_1` never receives a correction.
   This pushes (A1') to hold, and is the mechanism E2 already observed
   empirically ("retakes produce cleaner shots... a sound tooth photographed
   cleanly scores low against a pool containing dirtier shots").
2. *A lucky misread.* `D_t` is a noisy estimate of `R_t` (`head_noise`); a
   session can land in a good stratum `z` purely because the noise
   under-reported a still-bad `R_t`, with no real improvement to `q_t`. This
   pushes the OTHER way -- exactly the "confidently wrong" failure mode the
   whole framework exists to guard against -- and if it dominates, `q_t`
   given `Z_t{=}z` at test time is no better, or worse, than calibration's,
   and (A1') can fail.

Which mechanism dominates is governed by `head_noise` relative to how much a
genuine correction moves `q`, i.e. it is an empirical question about
magnitudes, not one the primitives settle on their own -- but it was a
*named* empirical question with a specific, falsifiable shape: **(A1') should
degrade, and eventually fail, as `head_noise` rises past the point where
mechanism 2 dominates mechanism 1**, holding everything else fixed. E4's
confidence-head-quality ablation already sweeps `head_noise` and shows *power*
(VPC) falling as it rises (0.230 -> 0.125).

**Checked directly (2026-08-23, E2 sweep D).** The prediction does not hold:
across `head_noise` in `{0.00, 0.06, 0.12, 0.25, 0.40, 0.60, 0.80, 1.00}` --
extended well past E4's original 0.40 endpoint, to 1.00, a noise standard
deviation as large as the severity scale itself -- the STRATIFIED arm's
false-conviction rate against `alpha = 0.05` never exceeds 0.0009 (Wilson
upper bound at every point stays below 0.0018), and if anything trends
*toward* zero as noise rises (0.0005 at `head_noise=0`, 0.0000 by
`head_noise=0.60`), not toward the nominal bound. No violation anywhere in
the tested range. `results/e2_validity.json:head_noise_sweep`,
`figures/e2_validity.png` panel (d).

**Why mechanism 2 does not win here, as far as the limit case makes
precise.** `D_t = R_t + head_bias + Normal(0, head_noise)` is the *same*
noise kernel, and `quality(·)` the same deterministic map, for both the
calibration population (`R^cal ~ pi_1`) and the test population (`R_t` under
the policy's correction history) -- nothing in the generative code lets the
confidence head be more or less accurate for one population than the other.
As `head_noise -> infinity`, conditioning on `Z = zeta(D) = z` becomes
asymptotically uninformative about `R` for *both* populations alike, so
`q | Z=z` converges to the unconditional law of `q` under each population's
own `R`-distribution, independently of `z`. In that limit (A1') collapses to
asking whether the policy's *unconditional* post-correction quality
stochastically dominates a fresh first draw's -- exactly the raw
equilibrium-shift property E1 already measures directly (mean true usability
+0.1098, first shot to last), with the stratification machinery contributing
nothing to that comparison one way or the other. Symmetric noise coarsening
a shared conditioning variable degrades how much *extra* power stratification
buys on top of that raw improvement (E4's falling VPC), but it does not, by
this argument, reverse an ordering that already holds unconditionally. This
is a rigorous account of the high-noise end of the sweep, not a proof that
covers every intermediate `head_noise` or every policy; it identifies the
structural reason (a common, unbiased noise kernel across both populations)
the predicted failure needed to be violated in order to appear, and that
premise is a real property of this generative model, not an assumption
smuggled in.

**The asymmetric case, checked directly too (2026-08-23, E2 sweep E).** The
gap just named -- a confidence head whose noise is *not* symmetric across
calibration and test -- was checked, not left as a caveat. Calibration noise
held fixed at 0.12; test-time (retake-loop/deployment) noise swept up to
2.00, a 16.7x asymmetry, with separation/loss_scale/sigma/gamma/head_bias
held identical across both so only the confidence head's noise differs.
**No violation anywhere in that range either** -- the crossing rate again
trends toward zero, not toward `alpha = 0.05`. `results/e2_validity.json:
head_noise_asymmetry_sweep`, `figures/e2_validity.png` panel (e).

This is a more surprising result than sweep D's, and it should be stated
honestly as *less* explained. The limiting-case argument for sweep D relies
specifically on the *same* noise kernel corrupting both populations, so that
as noise grows, conditioning on the stratum becomes uninformative for both
alike and the comparison collapses to a shared unconditional baseline; that
premise does not hold here by construction (calibration noise is held fixed,
low, and genuinely informative, while test noise grows toward
uninformative). A plausible partial account: the raw equilibrium-shift
property (E1) that ultimately does the work in the symmetric case is a
property of the *policy's correction dynamics*, not of the confidence head at
all, so a noisier test-time head degrades how well the stratification
targets that improvement without necessarily erasing the improvement itself
-- but this is an intuition, not a derivation, and unlike sweep D's finding
it does not come with a limiting-case proof attached. Treat the asymmetric
result as a robust empirical finding across the range tested, and a
concrete, sharper open question for future theoretical work: is there an
asymmetry magnitude, or a differently-shaped one (e.g. a systematic bias
rather than added variance), where it does eventually break?

**Summary of the reduction.** A1 is not proved outright for an arbitrary
generative model or policy -- proving it in general would mean proving a
distributional comparison that depends on magnitudes (`head_noise` vs.
correction strength) the primitives leave free, and no clean theorem removes
that in general. What the argument above does establish, for the simulator
and policies actually used here, is stronger than "reduced to a named
assumption": A1 reduces *exactly* to (A1') via the Markov screening lemma,
the specific failure mode that reduction predicted was checked directly
across a wide `head_noise` range (both symmetric and, further, asymmetric
between calibration and test) and did not occur in either, and there is a
structural explanation -- not just a numerical coincidence -- for the
symmetric case specifically. The remaining gap is now a matter of theoretical
completeness rather than an unchecked empirical worry: nothing found here
constitutes a proof that (A1') holds in general, but the two most natural
ways it was predicted to fail were both tested directly and neither
occurred.

## 7. What is claimed

1. Under A1–A3, the burden of proof is anytime-valid under fully adaptive
   retaking (§4). Proved.
2. A2–A3 are enforceable by construction, not by discipline (§3). Implemented
   and tested.
3. Two evidence constructions with a stated trade-off: provable-and-lossy
   versus powerful-and-estimated (§5). Both measured.
4. The assumption A1 is the load-bearing one, it is empirically supported in
   the regimes tested, and it degrades gracefully rather than catastrophically
   as the case mix drifts (§6). Not proved outright, but reduced via a proved
   Markov screening lemma to one named condition (A1', stratum-conditional
   stochastic dominance of true quality, test vs. calibration), with a
   mechanistic account of when that condition should hold (genuine
   equilibrium-shifting corrections) versus fail (noise-driven stratum
   misclassification as `head_noise` grows) — a falsifiable prediction that
   was then checked directly two ways (§6): symmetric `head_noise` up to
   1.00 (E2 sweep D, 2.5x past E4's original sweep — did not occur, with a
   structural, limiting-case explanation for why), and, further, asymmetric
   noise between calibration and deployment up to a 16.7x gap (E2 sweep E —
   also did not occur, though this second result is empirically robust
   without a comparably rigorous mechanistic account, and is named as an
   open question rather than claimed as understood).
