"""The instruction (subpoena) space.

When the system cannot meet its burden of proof, it does not simply say
"image rejected" -- it issues an instruction naming what to fix. Each
instruction targets exactly one latent scene factor. The simulator's job is to
make the response to that instruction realistic: partial, noisy, sometimes
refused, and with side effects on the *other* factors (see
`src.sim.state.COUPLING`).

`RETAKE_ANY` is the untargeted control ("just take another one"). It exists so
experiments can isolate how much of the benefit comes from retaking at all
versus from retaking with a *specific* instruction. In a process with
persistent latent state, an untargeted retake is nearly worthless -- the scene
is still lit the same way and the hand still shakes the same amount -- which
is precisely what an i.i.d. corruption model cannot represent.

There is deliberately NO instruction for `jpeg`. Transport compression is a
property of the channel the photo travels through (the messaging app), not of
the capture, so no retake can fix it. It is the irreducible floor of the
process, and it is why some sessions must escalate to a clinician no matter
how large the capture budget is.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Instruction:
    """One actionable retake instruction.

    Attributes:
        name: machine identifier.
        target: the `src.sim.state` factor this instruction tries to reduce,
            or None for the untargeted control.
        message: what a health worker would actually be shown.
    """

    name: str
    target: str | None
    message: str


REDUCE_GLARE = Instruction(
    "REDUCE_GLARE",
    "glare",
    "Too much reflection -- turn the film away from the lamp or window and retake.",
)
STEADY = Instruction(
    "STEADY",
    "tremor",
    "Photo is blurred -- brace your elbows or rest the phone on something and retake.",
)
MORE_LIGHT = Instruction(
    "MORE_LIGHT",
    "darkness",
    "Too dark -- move somewhere brighter or turn on the lightbox and retake.",
)
ALIGN = Instruction(
    "ALIGN",
    "tilt",
    "Shot is off-angle -- hold the phone square to the film and retake.",
)
RETAKE_ANY = Instruction(
    "RETAKE_ANY",
    None,
    "Image not usable -- please retake.",
)

INSTRUCTIONS: list[Instruction] = [REDUCE_GLARE, STEADY, MORE_LIGHT, ALIGN, RETAKE_ANY]

_BY_TARGET = {i.target: i for i in INSTRUCTIONS}
_BY_NAME = {i.name: i for i in INSTRUCTIONS}


def instruction_for_factor(factor: str | None) -> Instruction:
    """The instruction that targets `factor`, or RETAKE_ANY if none does.

    `jpeg` has no instruction by design (see module docstring), so it maps to
    the untargeted control rather than raising -- a policy that identifies
    compression as the dominant problem has nothing useful to ask for.
    """
    return _BY_TARGET.get(factor, RETAKE_ANY)


def instruction_by_name(name: str) -> Instruction:
    return _BY_NAME[name]
