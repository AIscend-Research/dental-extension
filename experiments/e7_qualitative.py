"""E7 -- qualitative figures on real radiographs.

The other experiments produce numbers. This one produces the pictures that make
the mechanism legible: what a capture session actually looks like, what the
instructions actually change, and what the evidence is doing while it happens.

Nothing here is staged. The images are real DENTEX radiographs, the degradation
is rendered by the same simulator the benchmark uses, and the e-values, stakes
and verdicts are produced by the same `VerdictMachine` -- the session loop is
re-implemented inline only so each shot's internals can be drawn, not to change
what they are.

Produces:
    figures/q1_capture_session.png  -- one tooth, one retake loop, live evidence
    figures/q2_degradation_atlas.png -- the artifact vocabulary on a real panoramic
    figures/q3_glare_geometry.png   -- why glare position matters, not just brightness

Run: .venv/bin/python -m experiments.e7_qualitative
"""

from __future__ import annotations

import cv2
import numpy as np

from experiments.common import CLINIC_DIFFICULTY, HEADLINE_BURDEN, banner, figure_path
from src.data.degradation import DEGRADATION_NAMES
from src.data.dentex_crops import (
    DEFAULT_ANNOTATIONS,
    DEFAULT_IMAGES,
    load_tooth_crops,
    split_by_source_image,
)
from src.evidence.calibration import LikelihoodRatioCalibrator
from src.evidence.verdict import DEGRADATION_TO_FACTOR, VerdictMachine, VerdictOutcome
from src.models.diagnostic import Case
from src.models.real_channel import train_real_channel
from src.sim.instructions import instruction_for_factor
from src.sim.render import render_capture, render_severities
from src.sim.session import CaptureSession
from src.sim.state import FACTORS, SceneState

# House style. Kept in one place so the qualitative figures read as a set with
# the quantitative ones rather than as a separate document.
INK = "#1a1a1a"
MUTED = "#6b6b6b"
ACCENT = "#1f6feb"
WARN = "#d1451b"
GOOD = "#1a7f37"
PAPER = "#ffffff"
BUDGET = 4


def _rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Q1 -- one tooth through one retake loop
# ---------------------------------------------------------------------------


def run_traced_session(crop, channel, calibrator, seed: int, budget: int = BUDGET):
    """Run one real session, keeping every intermediate for drawing.

    This is `EvidentialCapture.run` with the internals kept rather than
    discarded. It must stay behaviourally identical, so it uses the same
    VerdictMachine and the same instruction mapping.
    """
    rng = np.random.default_rng(seed)
    session = CaptureSession(rng=rng, difficulty=CLINIC_DIFFICULTY)
    machine = VerdictMachine(calibrator=calibrator, burden=HEADLINE_BURDEN)
    case = Case(label=crop.label, difficulty=0.0, payload=crop.image)

    shots, instruction = [], None
    for t in range(budget):
        capture = session.capture(instruction if t else None)
        rendered = render_severities(crop.image, capture.severities, rng=rng)
        reading = channel.read(case, capture.severities, rng)
        verdict = machine.observe(
            score=reading.score,
            degradation=reading.degradation,
            predicted_usability=reading.usability,
            captures_remaining=budget - t - 1,
        )
        shots.append({
            "image": rendered.image,
            "true_severities": dict(capture.severities),
            "reading": reading,
            "verdict": verdict,
            "wealth_convict": machine.convict.wealth,
            "wealth_discharge": machine.discharge.wealth,
            "instruction_in": instruction,
        })
        if verdict.is_terminal:
            break
        instruction = verdict.instruction
    return shots


def find_illustrative_session(crop, channel, calibrator, max_seed=400):
    """Find a seed where the loop takes several shots and then decides correctly.

    Searching for a legible example is a presentational choice, and stating it
    is the honest version: this figure shows what a *successful* session looks
    like, not a typical one. The aggregate behaviour -- including the 32% that
    escalate -- is E3's job, and the caption says so.
    """
    best = None
    for seed in range(max_seed):
        shots = run_traced_session(crop, channel, calibrator, seed)
        outcome = shots[-1]["verdict"].outcome
        correct = (
            (outcome is VerdictOutcome.CARIES and crop.label == 1)
            or (outcome is VerdictOutcome.SOUND and crop.label == 0)
        )
        if len(shots) < 3 or not correct:
            continue
        # Rank by how much the capture actually improved, and reward sessions
        # that issue more than one distinct instruction -- a figure showing the
        # same instruction three times illustrates nothing about targeting.
        gain = shots[-1]["reading"].usability - shots[0]["reading"].usability
        distinct = len({
            s["instruction_in"].name for s in shots if s["instruction_in"] is not None
        })
        score = gain + 0.12 * distinct
        if best is None or score > best[1]:
            best = (shots, score, seed)
    return best


def figure_capture_session(crop, channel, calibrator) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    found = find_illustrative_session(crop, channel, calibrator)
    if found is None:
        raise RuntimeError("no illustrative session found")
    shots, _, seed = found
    n = len(shots)

    fig = plt.figure(figsize=(3.7 * n + 1.8, 10.0), facecolor=PAPER)
    gs = GridSpec(
        3, n, figure=fig, height_ratios=[2.5, 1.15, 1.5],
        hspace=0.46, wspace=0.16, left=0.07, right=0.88, top=0.775, bottom=0.06,
    )

    for i, shot in enumerate(shots):
        # -- the photograph ------------------------------------------------
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(_rgb(shot["image"]))
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(INK); spine.set_linewidth(1.2)
        r = shot["reading"]
        ax.set_title(
            f"shot {i + 1}\nusability {r.usability:.2f}   caries score {r.score:.2f}",
            fontsize=10, color=INK, pad=8,
        )
        if shot["instruction_in"] is not None:
            ax.text(
                0.5, -0.055, f'after: "{shot["instruction_in"].name}"',
                transform=ax.transAxes, ha="center", va="top",
                fontsize=8.5, color=ACCENT, style="italic",
            )

        # -- what the degradation channel saw -------------------------------
        axd = fig.add_subplot(gs[1, i])
        y = np.arange(len(DEGRADATION_NAMES))
        pred = [r.degradation[k] for k in DEGRADATION_NAMES]
        true = [shot["true_severities"].get(k, 0.0) for k in DEGRADATION_NAMES]
        axd.barh(y + 0.18, true, 0.34, color="#c9c9c9", label="true")
        axd.barh(y - 0.18, pred, 0.34, color=ACCENT, label="predicted")
        axd.set_yticks(y, DEGRADATION_NAMES if i == 0 else [""] * len(y), fontsize=8)
        axd.set_xlim(0, 1)
        axd.invert_yaxis()
        axd.tick_params(labelsize=7.5)
        for spine in ("top", "right"):
            axd.spines[spine].set_visible(False)
        if i == 0:
            axd.legend(fontsize=7, loc="lower right", frameon=False)
            axd.set_xlabel("severity", fontsize=8)

        # -- the subpoena, drawn between panels ------------------------------
        v = shot["verdict"]
        if v.instruction is not None and i < n - 1:
            fig.text(
                0.07 + 0.81 * (i + 1) / n - 0.006, 0.565, "→",
                fontsize=26, color=ACCENT, ha="center", va="center",
            )

    # -- the evidence, across the whole session -----------------------------
    axw = fig.add_subplot(gs[2, :])
    x = np.arange(1, n + 1)
    axw.plot(x, [s["wealth_convict"] for s in shots], "o-", color=WARN, lw=2,
             label="wealth against 'the tooth is sound'  (convict)")
    axw.plot(x, [s["wealth_discharge"] for s in shots], "s-", color=MUTED, lw=2,
             label="wealth against 'the tooth is carious'  (discharge)")
    for value, colour, label in [
        (HEADLINE_BURDEN.convict.threshold, WARN, "burden to convict"),
        (HEADLINE_BURDEN.discharge.threshold, MUTED, "burden to discharge"),
    ]:
        axw.axhline(value, color=colour, ls=":", lw=1.5)
        axw.text(
            1.008, value, f"{label} = {value:.0f}x",
            transform=axw.get_yaxis_transform(), fontsize=8.5, color=colour,
            va="center", ha="left",
        )
    axw.axhline(1.0, color="#cccccc", lw=1)
    axw.set_yscale("log")
    axw.set_xticks(x, [f"shot {i}" for i in x], fontsize=9)
    axw.set_xlim(0.85, n + 0.9)
    axw.set_ylabel("evidence (wealth)", fontsize=9)
    axw.legend(fontsize=8, loc="upper left", frameon=False)
    for spine in ("top", "right"):
        axw.spines[spine].set_visible(False)

    outcome = shots[-1]["verdict"].outcome
    truth = "caries" if crop.label == 1 else "sound"
    colour = GOOD if outcome in (VerdictOutcome.CARIES, VerdictOutcome.SOUND) else MUTED
    axw.text(
        0.995, 0.06,
        f"VERDICT: {outcome.value.upper()}   (ground truth: {truth})",
        transform=axw.transAxes, ha="right", fontsize=11, color=colour, weight="bold",
    )

    fig.suptitle(
        "One tooth, one retake loop:\nthe system photographs until it can meet its burden of proof",
        fontsize=15.5, color=INK, y=0.99, va="top",
    )
    fig.text(
        0.5, 0.905,
        "Real DENTEX radiograph. Degradation rendered by the capture simulator; e-values, stakes and verdict\n"
        "from the same VerdictMachine the benchmark uses. Seed "
        f"{seed} shows a session that succeeds — 32% escalate to a clinician instead (E3).",
        ha="center", va="top", fontsize=9.5, color=MUTED, linespacing=1.6,
    )

    path = figure_path("q1_capture_session.png")
    fig.savefig(path, dpi=150, facecolor=PAPER)
    plt.close(fig)
    return str(path)


# ---------------------------------------------------------------------------
# Q2 -- the artifact vocabulary, on a real panoramic
# ---------------------------------------------------------------------------


def figure_degradation_atlas() -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import json

    data = json.loads(DEFAULT_ANNOTATIONS.read_text())
    meta = data["images"][0]
    pano = cv2.imread(str(DEFAULT_IMAGES / meta["file_name"]), cv2.IMREAD_COLOR)
    pano = cv2.resize(pano, (760, int(760 * pano.shape[0] / pano.shape[1])))

    severities = [0.0, 0.35, 0.7, 1.0]
    names = list(DEGRADATION_NAMES)
    # A detail column at full severity. Without it `jpeg` looks like a no-op:
    # block artifacts live at a spatial scale the whole-panoramic view throws
    # away, and jpeg is the artifact the paper leans on most (it is the one no
    # retake can fix), so it cannot be the one the figure fails to show.
    h, w = pano.shape[:2]
    detail = (slice(int(0.42 * h), int(0.80 * h)), slice(int(0.52 * w), int(0.78 * w)))

    fig, axes = plt.subplots(
        len(names), len(severities) + 1,
        figsize=(2.75 * (len(severities) + 1), 1.62 * len(names)), facecolor=PAPER,
    )
    for r, name in enumerate(names):
        full_severity_image = None
        for c, sev in enumerate(severities):
            ax = axes[r, c]
            rng = np.random.default_rng(11)
            out = render_severities(pano, {name: sev}, rng=rng)
            if sev == 1.0:
                full_severity_image = out.image
            ax.imshow(_rgb(out.image))
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#dddddd")
            if r == 0:
                ax.set_title(
                    "clean" if sev == 0 else f"severity {sev:.2f}",
                    fontsize=10, color=INK, pad=6,
                )
            if c == 0:
                ax.set_ylabel(name, fontsize=11, color=INK, rotation=0,
                              ha="right", va="center", labelpad=14)

        ax = axes[r, -1]
        patch = full_severity_image[detail[0], detail[1]]
        ax.imshow(_rgb(cv2.resize(patch, None, fx=2.4, fy=2.4, interpolation=cv2.INTER_NEAREST)))
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(ACCENT); spine.set_linewidth(1.4)
        if r == 0:
            ax.set_title("detail, severity 1.00", fontsize=10, color=ACCENT, pad=6)

    fig.suptitle(
        "The artifact vocabulary, rendered on a real panoramic radiograph",
        fontsize=14.5, color=INK, y=0.995, va="top",
    )
    fig.text(
        0.5, 0.955,
        "The first four are scene properties a retake can change. `jpeg` is the messaging app the photo travels\n"
        "through — no instruction fixes it, which is why some cases must escalate however large the budget.",
        ha="center", va="top", fontsize=9.5, color=MUTED, linespacing=1.6,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.915])
    path = figure_path("q2_degradation_atlas.png")
    fig.savefig(path, dpi=150, facecolor=PAPER)
    plt.close(fig)
    return str(path)


# ---------------------------------------------------------------------------
# Q3 -- glare has a position, not just a brightness
# ---------------------------------------------------------------------------


def figure_glare_geometry(crop) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    azimuths = [0.0, 0.08, 0.16, 0.28, 0.5]
    intensity = 0.85
    fig, axes = plt.subplots(1, len(azimuths) + 1, figsize=(3.0 * (len(azimuths) + 1), 4.9),
                             facecolor=PAPER)

    effective = []
    for i, az in enumerate(azimuths):
        scene = SceneState(
            factors={"glare": intensity, "tremor": 0.0, "darkness": 0.0, "tilt": 0.0},
            glare_azimuth=az,
        )
        eff = scene.effective_glare()
        effective.append(eff)
        rng = np.random.default_rng(4)
        # positioned_glare so the picture shows what the model means: the
        # highlight keeps its brightness and moves, rather than dimming in place
        out = render_capture(crop.image, scene, rng=rng, positioned_glare=True)
        ax = axes[i]
        ax.imshow(_rgb(out.image))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(
            f"hotspot offset {az:.2f}\neffective glare {eff:.2f}",
            fontsize=10, color=WARN if eff > 0.4 else GOOD, pad=6,
        )
        for spine in ax.spines.values():
            spine.set_edgecolor("#dddddd")

    ax = axes[-1]
    fine = np.linspace(0, 0.5, 200)
    curve = [
        SceneState(
            factors={"glare": intensity, "tremor": 0.0, "darkness": 0.0, "tilt": 0.0},
            glare_azimuth=a,
        ).effective_glare()
        for a in fine
    ]
    ax.plot(fine, curve, color=ACCENT, lw=2)
    ax.scatter(azimuths, effective, color=WARN, zorder=3, s=36)
    ax.axhline(intensity, color=MUTED, ls=":", lw=1.2)
    ax.text(0.26, intensity + 0.02, "glare intensity (unchanged)", fontsize=8, color=MUTED)
    ax.set_xlabel("hotspot offset from the tooth", fontsize=9)
    ax.set_ylabel("effective glare", fontsize=9)
    ax.set_title("same brightness,\ndifferent harm", fontsize=10, color=INK, pad=6)
    ax.set_ylim(-0.03, 1.0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.suptitle(
        'Why "REDUCE_GLARE" has two physically different solutions',
        fontsize=14.5, color=INK, y=0.985, va="top",
    )
    fig.text(
        0.5, 0.915,
        "The reflection is equally bright in all five photographs — only its position changes. Turning the film moves the\n"
        "hotspot off the tooth, fixing the image without dimming anything, which a scalar severity cannot express.",
        ha="center", va="top", fontsize=9.5, color=MUTED, linespacing=1.6,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.845])
    path = figure_path("q3_glare_geometry.png")
    fig.savefig(path, dpi=150, facecolor=PAPER)
    plt.close(fig)
    return str(path)


def main() -> None:
    banner("E7 -- qualitative figures on real radiographs")
    rng = np.random.default_rng(0)

    crops = load_tooth_crops(task="caries_vs_other", crop_size=256, context=0.75)
    train, cal_crops, test = split_by_source_image(crops, seed=3)
    print(f"training the real reader on {len(train)} crops ...", flush=True)
    channel = train_real_channel(train, rng, n_renders=12, clinic_difficulty=CLINIC_DIFFICULTY)

    from experiments.e6_real_images import collect_real_calibration

    cal_data = collect_real_calibration(channel, cal_crops, rng)
    calibrator = LikelihoodRatioCalibrator(n_strata=4).fit(
        cal_data.scores, cal_data.labels, cal_data.usabilities
    )

    carious = [c for c in test if c.label == 1]
    print(f"q1: tracing a session on a real carious tooth ...", flush=True)
    print("  ->", figure_capture_session(carious[0], channel, calibrator))
    print("q2: degradation atlas ...", flush=True)
    print("  ->", figure_degradation_atlas())
    print("q3: glare geometry ...", flush=True)
    print("  ->", figure_glare_geometry(carious[0]))


if __name__ == "__main__":
    main()
