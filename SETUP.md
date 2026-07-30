# Setup

Two tracks. Everyone does Track A. One person does Track B first and writes down
what actually worked, because Track B is finicky.

## Track A — the startable half (everyone, ~2 min, no GPU)

```bash
bash setup_env.sh
```

If you would rather do it by hand:

```bash
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements-core.txt
bash scripts/clone_baseline.sh
python -m tests.test_degradation
python -m tests.test_metrics
python demo_degradation.py            # writes degradation_demo.png
```

That is enough to work on the degradation pipeline, the DENTEX loader, the
metrics, and the confidence/decision logic. You do not need Track B for any of
those.

To pull the dataset:

```bash
pip install huggingface_hub
huggingface-cli login                 # DENTEX is gated; accept the terms on the HF page
python scripts/download_dentex.py
```

Then look at what landed and point `configs/default.yaml` at the real paths:

```bash
find data/dentex -name '*.json'
find data/dentex -type d
```

## Track B — the detector stack (one person first)

This is HierarchicalDet: DiffusionDet on top of a **modified Detectron2**, with a
Swin-L backbone. It needs a GPU and it is version-sensitive. Budget real time
for it.

### The main gotcha, up front

`external/HierarchicalDet/detectron2` is a copy of Detectron2 **with no
`setup.py` and no compiled ops** (`csrc/`, `.cu` files). Detectron2 normally
needs compiled C++/CUDA ops (`detectron2._C`) for things like ROIAlign and NMS.
So you have two realistic options:

1. **Install the official Detectron2** (which provides the compiled `_C`), then
   run HierarchicalDet's Python code against it. The catch is import shadowing:
   the repo puts its `detectron2/` folder at the import root, so you have to make
   sure their modified Python modules win while the compiled `_C` comes from the
   installed package. Test early that both `import detectron2` and
   `from detectron2 import _C` resolve without conflict.

2. **Build a matching upstream Detectron2 from source** and port their
   modifications on top. Cleaner in principle, more work.

**Confirmed working (2026-07-29, macOS arm64, CPU/MPS, no GPU): option 1,
and it's less painful than this section's tone implies.** Diffed the vendored
`external/HierarchicalDet/detectron2/` against a fresh `pip install` of
official detectron2 -- both report `__version__ == "0.6"`, but the vendored
copy is an older 2023-era snapshot (e.g. missing `@disable_torch_compiler`
decorators the current one has). It is **not** a deliberately modified fork
for DiffusionDet -- the actual DiffusionDet-specific code lives entirely in
`hierarchialdet/`, not in `detectron2/`. That means you don't need to reconcile
two divergent detectron2s -- just get the real one installed and let
`hierarchialdet` import against it.

The exact recipe that worked, in a **separate venv from the core one**
(`python3.12`, not whatever the core venv uses -- 3.14 is too new for
detectron2's build):

```bash
python3.12 -m venv .venv-detector && source .venv-detector/bin/activate
python -m pip install torch torchvision          # macOS: CPU+MPS wheels, no special index needed
python -m pip install ninja setuptools wheel
# --no-build-isolation is required: detectron2's setup.py does `import torch`
# at build time, and pip's isolated build env can't see the venv's torch
# without this flag. Omitting it is the #1 way this install fails.
python -m pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
python -m pip install timm scipy Pillow
python -c "import detectron2; from detectron2 import _C; print('ops ok')"  # confirmed: works
```

Then, to import `hierarchialdet` **without** the vendored `detectron2/` or
`pycocotools/` folders shadowing the real installed ones: import the real
packages fully *before* adding `external/HierarchicalDet` to `sys.path`.
Python's import system checks `sys.modules` first, so once `detectron2` and
`pycocotools` are already cached there, appending the HierarchicalDet path
afterward can't un-cache and re-resolve them to the vendored copies:

```python
import sys
import pycocotools.mask, pycocotools.coco, pycocotools.cocoeval  # cache the real one first
import detectron2
from detectron2 import _C  # confirms compiled ops loaded

sys.path.insert(0, "external/HierarchicalDet")
from hierarchialdet.config import add_diffusiondet_config  # now safe
```

Confirmed end to end on this machine: `build_model()` on
`diffdet.custom.swinbase.nonpretrain.yaml` (Swin-L + DiffusionDet, 281.8M
params, `MODEL.WEIGHTS=""` since the backbone `.pkl` isn't downloaded) builds
and runs a forward pass on a dummy 800x800 image on CPU in ~2.7s, returning
zero instances (expected/correct for random-init weights, not a bug -- nothing
survives score thresholding without trained weights). This confirms the
architecture wires up correctly; it says nothing about training time or
quality, which still need the real GPU/Kaggle run.

### Ordering that matters

Install in this order or the compiled ops will refuse to load:

1. **Torch first, matching your CUDA.** On Kaggle, do NOT reinstall torch — use
   the version the image already ships and match everything else to it. On your
   own box, pick the torch build for your CUDA from pytorch.org.
2. **Detectron2 matching that torch/CUDA.** Use the prebuilt wheel for your exact
   torch+CUDA combo if one exists; building from source otherwise. A mismatch
   here is the number one failure.
3. The rest: `pip install -r requirements-detector.txt` (timm, fvcore, scipy,
   etc.).

### pycocotools note

HierarchicalDet also vendors its own `pycocotools` (no compiled `_mask`
extension, same shadowing issue as `detectron2/` above, same fix: `import
pycocotools.mask` etc. before adding `external/HierarchicalDet` to
`sys.path`). If you `pip install pycocotools` as well you can get a clash --
resolved by the import-order trick above, not by uninstalling either copy.

### Backbone weights -- a real bug in the upstream config, confirmed and fixed

**The config's own weights filename is misleading and will silently give you
the wrong checkpoint if followed literally.**
`configs/diffdet.custom.swinbase.nonpretrain.yaml` sets `MODEL.WEIGHTS:
"models/swin_base_patch4_window7_224_22k.pkl"` (implying Swin-**Base**) but
also sets `MODEL.SWIN.SIZE: L-22k` (Swin-**Large**) -- these are inconsistent.
DiffusionDet's own official release only ships Swin-Base weights under that
exact filename (confirmed: 128-dim embeddings). Loading those into this
config's actual architecture (confirmed built: 192-dim embeddings, 281.9M
total params -- genuinely Large-scale, matching every benchmark number in
`docs/phase3_model_benchmarks.md`) silently fails to load the majority of the
backbone -- `DetectionCheckpointer` warns per-tensor ("will not be loaded")
but does not raise an error, so this is easy to miss and would produce a
near-useless model trained from mostly-random backbone weights.

**Confirmed correct recipe (2026-07-29, verified zero shape mismatches):**

```bash
# 1. Get the RAW Swin-Large-22k classification checkpoint (Microsoft's
#    official release, NOT DiffusionDet's -- DiffusionDet never shipped one)
curl -sL "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth" \
  -o models_weights/swin_large_patch4_window7_224_22k_raw.pth

# 2. Convert to detectron2's expected pkl format (just rewraps the state
#    dict -- same pattern as e.g. FoundationVision/GenerateU's
#    convert-pretrained-swin-model-to-d2.py)
python -c "
import torch, pickle
ckpt = torch.load('models_weights/swin_large_patch4_window7_224_22k_raw.pth', map_location='cpu', weights_only=False)
converted = {'model': ckpt['model'], '__author__': 'third_party', 'matching_heuristics': True}
with open('models_weights/swin_large_patch4_window7_224_22k.pkl', 'wb') as f:
    pickle.dump(converted, f)
"
```

Then point `cfg.MODEL.WEIGHTS` at
`models_weights/swin_large_patch4_window7_224_22k.pkl` (NOT the filename the
config literally states). Confirmed on this machine: `DetectionCheckpointer`
loads it with zero "will not be loaded" warnings on any `backbone.bottom_up.*`
key -- the only "not found in checkpoint" keys afterward are things that were
never going to be in an ImageNet classification checkpoint anyway (the FPN
lateral/output convs, the DiffusionDet head, the diffusion schedule buffers) --
that part is normal and expected, not a sign of a bad load.

Do not use DiffusionDet's own `swin_base_patch4_window7_224_22k.pkl` release
asset for this config -- it is real, downloads fine, and will load into the
model with *zero errors* (`DetectionCheckpointer` doesn't hard-fail on shape
mismatches, just warns per-key), which makes the bug easy to miss.

### Kaggle specifics

The proposal requires this to run on Kaggle compute, so sort it out on Kaggle
early rather than discovering a blocker in Phase 3. The recipe above was
verified locally on CPU/MPS (macOS, no GPU) to de-risk the install path and
confirm the architecture -- it is NOT the Kaggle recipe: on Kaggle, skip the
`pip install torch torchvision` step entirely and use the image's preinstalled
torch (see below), then run the same `--no-build-isolation` detectron2 install
against that.

- Kaggle ships a fixed torch+CUDA per image. Match Detectron2 to it; do not
  upgrade torch.
- Kaggle notebooks have limited internet — enable it in notebook settings, or
  attach the repo and the DENTEX data as Kaggle Datasets and run offline (which
  is closer to the real deployment story anyway).
- GPU sessions are time-limited. HierarchicalDet's config runs 40k iterations at
  batch size 2, which will not finish in one session. Plan for checkpointing and
  resuming, and benchmark iteration time before committing to a full run.

### Sanity check for Track B

Do **not** `cd external/HierarchicalDet` and `import detectron2` from there --
that's the exact shadowing trap above, and it will silently give you the
vendored copy with no compiled ops instead of erroring clearly. Use the
import-order recipe above (real packages first, then `sys.path.insert` the
HierarchicalDet dir) from the repo root instead. Confirm the baseline imports
and a forward pass runs on a single image before touching training -- see the
worked example above (`build_model()` + a dummy-image forward pass); a real
run should additionally load actual weights and a real DENTEX image via
`src/data/dentex.py` instead of a dummy tensor.

If `from detectron2 import _C` fails, stop and fix the install — nothing
downstream will work until it resolves.
