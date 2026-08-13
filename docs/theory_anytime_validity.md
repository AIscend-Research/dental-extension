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

**What is assumed, not proved, about A1.** The construction gives conditional
validity given the stratum; it does not rule out residual dependence between
`S_t` and the session history within a stratum. Proving A1 from primitives
about the capture process — rather than assuming it and measuring the
consequences — is the obvious next theoretical step and is not done here.

## 7. What is claimed

1. Under A1–A3, the burden of proof is anytime-valid under fully adaptive
   retaking (§4). Proved.
2. A2–A3 are enforceable by construction, not by discipline (§3). Implemented
   and tested.
3. Two evidence constructions with a stated trade-off: provable-and-lossy
   versus powerful-and-estimated (§5). Both measured.
4. The assumption A1 is the load-bearing one, it is empirically supported in
   the regimes tested, and it degrades gracefully rather than catastrophically
   as the case mix drifts (§6). Measured, not proved.
