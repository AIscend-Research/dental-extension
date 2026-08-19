# Where the simulator's parameters come from

A simulator is only as useful as the honesty of its parameter table. This
document says, for every number that matters, whether it is anchored to
something external, inferred from the structure of the problem, or simply
chosen — and what the sensitivity experiment does about the ones that were
chosen.

The short version: the **taxonomy** is literature-backed, the **dynamics** are
a modelling contribution stated as assumptions, and the **magnitudes** are
mostly chosen and therefore swept. No number here should be cited as a
measurement of the real world.

## 1. The degradation taxonomy — literature-backed

The five artifact primitives (`src/data/degradation.py`) and the four
instructable latent factors (`src/sim/state.py`) are not invented. The
smartphone and photographic dental-imaging literature repeatedly identifies the
same failure modes for phone-captured oral and radiographic images: **ambient
lighting**, **camera angle / off-axis capture**, **motion and focus**, and
**specular reflection** (from saliva intra-orally, or from the film surface and
lightbox when photographing a radiograph). `docs/phase1_background.md` records
the related-work pass, of which CheXphoto (ML4H 2020) is the closest prior
work: it constructs a smartphone-photograph corruption benchmark for chest
radiographs and identifies substantially this set of artifacts.

Two deliberate departures from that literature:

- **`jpeg` is modelled as a channel property, not a scene property.** Transport
  compression is applied by the messaging app the photo travels through, and no
  retake fixes it. It is constant within a session and has no instruction. This
  is what puts an irreducible floor under the escalation rate: some sessions
  must go to a clinician however large the capture budget.
- **Glare has a *position*, not just an intensity.** A bright reflection in the
  corner of a film is harmless; the same reflection over the lesion hides it.
  `SceneState.glare_azimuth` and `effective_glare()` encode this, which is why
  `REDUCE_GLARE` can succeed either by dimming the highlight or by moving it —
  two physically different fixes behind one instruction.

## 2. The dynamics — the modelling contribution, stated as assumptions

These are the claims that make this a capture *process* rather than a
corruption filter. None is measured; each is a structural assumption, and E1
verifies the simulator actually implements it.

| Assumption | Implementation | E1 measurement |
|---|---|---|
| The scene persists between shots | AR(1) toward a per-session equilibrium, ρ = 0.85 | lag-1 correlation 0.963 within session vs 0.011 shuffled across |
| Instructions are followed partially and sometimes not at all | Bernoulli compliance × Beta-distributed correction | compliance 0.66–0.68; targeted effect −0.07 to −0.12 vs ≈0 untargeted |
| Fixing one fault disturbs others | `COUPLING` matrix scaled by operator haste | recovered empirically with the specified structure and sign |
| Retaking tires the operator | fatigue += 0.08/shot, raising the tremor equilibrium | tremor rises 0.316 → 0.407 over 8 shots while glare is being fixed |
| The resulting sequence is non-exchangeable | all of the above | permutation test p = 0.0005 |

The coupling matrix's entries are chosen, but their *structure* is physical
and each row is a specific claim: turning a film away from a lamp puts you
off-axis and further from the light (glare → tilt, darkness); moving toward a
window to gain light puts the reflection back (darkness → glare); squaring up
means re-gripping (tilt → tremor). The glare↔darkness pair is a genuine
trade-off, which predicts that a greedy "fix the worst factor" policy can
oscillate. E4 finds the effect on outcomes is small (VPC 0.218 at coupling 0
vs 0.211 at double strength), so this mechanism is real in the simulator but
not load-bearing for the results.

## 3. The reader — anchored at two points

The analytic reader (`SurrogateChannel`) is fitted, not chosen, at two stated
anchors:

- **Clean-image AUC = 0.88**, set by `calibrate_separation`. In the range
  reported for automated caries detection on good-quality radiographs, and
  deliberately not heroic.
- **Clinic AUC = 0.78** at median conditions, set by `calibrate_loss_scale`.
  The assumption is that phone capture costs a reader real accuracy without
  destroying it.

The second anchor exists because it has to. The five artifacts compose
multiplicatively, so per-artifact weights that are individually plausible stack
into an implausible whole: unconstrained, the default weights drove a
clean-AUC-0.88 reader to **0.574** under median conditions — near-useless.
Every comparison run in that regime measures noise rather than policy, so the
anchor is a correctness requirement, not a convenience. E5 sweeps the anchor
from 0.70 to 0.84 and the policy ordering survives all four settings.

`SIGNAL_LOSS_WEIGHTS` — the relative damage per artifact — is the weakest link
in the table. The *ordering* is argued from the physics of the task (caries on
a radiograph is a fine-grained density change, so blur and compression, which
destroy high-frequency detail, should hurt most; off-axis geometry distorts but
does not erase). The **magnitudes are chosen**, and §5 explains why the attempt
to validate the ordering on real data was inconclusive.

## 4. What the case distribution encodes

- **Prevalence 0.35** — roughly the caries share among annotated DENTEX teeth.
  Swept over 0.10–0.50 in E5; ordering holds throughout.
- **Lesion difficulty ~ Beta(2, 4)** — most lesions resolvable, with a real
  tail that no photograph settles. This exists so escalation rate is not
  merely a proxy for image quality: without genuinely undecidable cases, a
  policy could look perfect by retaking forever. Swept over four Beta settings
  in E5.
- **Clinic difficulty 0.5** — the midpoint between a district hospital and a
  torch-lit outpost. Swept over 0.2–0.8.

## 5. The sim-to-real check: attempted, inconclusive

The one claim in §3 that real data could adjudicate is the *ordering* of
per-artifact damage. E6 attempted it on real DENTEX radiographs and **could not
resolve it**, for a reason worth recording rather than burying:

- On the fine-grained task (Caries vs Deep Caries, the one whose signal is
  genuinely fine density detail), the reader scores **AUC 0.481** — chance —
  even using all 133 teeth via grouped cross-validation over source
  radiographs. There are 32 positives in the entire public split. A damage
  curve computed from a chance-level reader measures noise, so E6 refuses to
  draw an ordering conclusion from it and flags the result inconclusive in code
  (`MIN_INTERPRETABLE_AUC`).
- On the coarse task (Caries+Deep vs Impacted+Periapical) the reader reaches
  **AUC ≈ 0.99**, because impacted teeth are unerupted and separable on
  morphology alone. A saturated reader has no signal left to lose, so its
  damage curve is equally uninformative about the weights.

So the public 50-radiograph DENTEX split brackets the question without
answering it: too small to learn the fine task, too easy on the coarse one. The
machinery to run the check is written and tested; what it needs is the full
DENTEX training split (~700 diagnosis-labelled radiographs, the 11 GB archive)
or a purpose-collected set. Until then `SIGNAL_LOSS_WEIGHTS` should be read as
a stated assumption that the framework's conclusions were shown insensitive to
(E5), not as a calibrated measurement.

## 6. The honest summary

| Claim | Status |
|---|---|
| Artifact taxonomy | literature-backed |
| jpeg as unfixable channel property | structural argument |
| Glare position matters, not just intensity | structural argument |
| Persistence, partial compliance, coupling, fatigue | assumptions; implemented and verified in E1 |
| Clean AUC 0.88, clinic AUC 0.78 | anchors, stated; swept in E5 |
| Per-artifact damage weights | **chosen; validation attempted and inconclusive** |
| Coupling magnitudes | chosen; shown non-load-bearing in E4 |
| Prevalence, difficulty distributions | chosen; swept in E5 |

One half of that gap is now addressable without new labelled data. The
weights question in 5 needs a competent reader on labelled radiographs and
stays open, but whether the *artifacts themselves* look like phone artifacts
is settled by photographs, not by more DENTEX: `docs/phone_pilot_protocol.md`
is the protocol and `src/pilot/` the analysis, both built and waiting on the
IRB determination rather than on compute. Its 8.1 already reports one result
that needed no photographs -- the two synthetic arms' severity scales measured
against each other on real radiographs, which agree for blur and jpeg,
saturate for low_light, and never meet for glare.

The sim-to-real gap is the framework's principal limitation, and it is the same
one the robotics simulators this design borrows from carry: parameters grounded
where evidence exists, stated where it does not, and a conclusion whose
robustness is demonstrated by sweeping rather than asserted by fiat.
