"""Capture-session simulation: the "world" contribution.

`src/data/degradation.py` is a degradation *filter* -- it applies i.i.d.
corruptions to independent images, ImageNet-C style. This package is a
degradation *process*: a latent scene state that persists across shots within
one capture session and responds causally (and imperfectly) to instructions.

That difference is the whole point. In a retake loop, shot 2 exists *because*
shot 1 had glare, so the captures in a session are neither independent nor
exchangeable. A filter cannot produce that structure; a process can.

Entry points:
    SceneState / Operator  -- src.sim.state
    CaptureSession         -- src.sim.session
    Instruction            -- src.sim.instructions
    render_severities      -- src.sim.render (reuses src/data/degradation.py)
"""

from src.sim.instructions import INSTRUCTIONS, Instruction, instruction_for_factor
from src.sim.state import FACTORS, Operator, SceneState
from src.sim.session import Capture, CaptureSession, SessionConfig

__all__ = [
    "FACTORS",
    "INSTRUCTIONS",
    "Capture",
    "CaptureSession",
    "Instruction",
    "Operator",
    "SceneState",
    "SessionConfig",
    "instruction_for_factor",
]
