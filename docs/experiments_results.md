# Experimental results

Every number below is reproduced by `.venv/bin/python -m experiments.run_all`
(~9 min, CPU only). Raw output lives in `results/*.json` and `results/*.csv`;
figures in `figures/e[1-6]_*.png`.

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

n = 4000 cases, K = 4, likelihood-ratio evidence:

| policy | VPC | decided | accuracy | escalated | shots | status |
|---|---|---|---|---|---|---|
| greedy_diagnostic | 0.460 | 97.8% | 0.819 | 2.2% | 2.12 | **unguaranteed** |
| **evidential_capture** | **0.225** | 68.2% | 0.842 | 31.8% | 3.03 | ok |
| oracle_instruction | 0.219 | 66.4% | 0.843 | 33.6% | 3.04 | ok |
| untargeted_evidential | 0.190 | 59.2% | 0.839 | 40.8% | 3.12 | ok |
| fixed_retake | 0.145 | 58.0% | 0.853 | 42.0% | 4.00 | ok |
| single_shot | 0.118 | 11.8% | 0.750 | 88.2% | 1.00 | ok |
| naive_best_shot | 0.052 | 20.6% | 0.927 | 79.3% | 4.00 | **unguaranteed** |

Among sound arms the ordering is clean: **targeted evidential capture beats
untargeted by 18%, fixed bursts by 55%, and single-shot by 91%**. The unsound
arm wins on raw VPC and pays for it in accuracy (0.819 vs 0.842) and in the
guarantee it no longer has.

**A perfect confidence head buys nothing.** `oracle_instruction` reads the true
latent scene and still does not beat `evidential_capture`. The bottleneck is
not identifying the worst artifact — it is that "fix the worst factor" is
itself a suboptimal rule when corrections have side effects.

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

50 panoramic radiographs, 182 dentist-annotated teeth, split by source image.

**The sim-to-real check was attempted and is inconclusive.** The public split
brackets the question without answering it:

- *Caries vs Deep Caries* (fine-grained, the task whose signal is genuinely
  fine density detail): **AUC 0.481** — chance — using all 133 teeth via
  5-fold grouped CV over radiographs. There are 32 positives in the entire
  public split. A damage curve from a chance-level reader measures noise, so
  E6 flags it inconclusive in code rather than reporting the ranking.
- *Caries+Deep vs Impacted+Periapical* (coarse): **AUC 0.986** — near-saturated,
  because impacted teeth are separable on morphology alone. A saturated reader
  has no signal left to lose.

So `SIGNAL_LOSS_WEIGHTS` remains a stated assumption. The machinery to validate
it is written and tested; it needs the full DENTEX training split (~700
diagnosis-labelled radiographs) or purpose-collected data.

**What did work on real images:**

- The **degradation head trained by weak supervision transfers**: predicted vs
  true severity correlates r = **0.887** (blur), **0.681** (low light),
  **0.337** (glare); dominant-artifact accuracy **0.452** against a 0.20
  chance baseline. Blur and lighting are read reliably from real radiographs;
  glare and geometry much less so.
- The **full pipeline runs end to end on real radiographs** and the guarantee
  holds (0 violations across all arms).

**One instructive reversal.** On the near-saturated real task, `single_shot`
(VPC 0.392) beats `evidential_capture` (0.373). When the reader is nearly
perfect, one photograph already meets the burden and retaking is pure cost.
Evidential capture earns its keep when the reader is imperfect — which is the
regime the framework is for, but it is worth stating that the method is not
free.

---

## Summary of negative and refuted results

Kept deliberately, because they were predictions this design got wrong:

1. **Degradation-aware betting loses to constant staking** (E4). The
   information is already spent in the stratified e-value.
2. **Marginal calibration does not break validity** (E2). The selection shift
   runs conservative in this capture process.
3. **A perfect confidence head does not improve verdicts** (E3). The greedy
   "fix the worst factor" rule is the binding constraint.
4. **Instruction side effects barely matter** to outcomes (E4), despite being
   a deliberate feature of the simulator.
5. **The sim-to-real check is unresolved** (E6). The public data is too small
   for the fine task and too easy on the coarse one.
