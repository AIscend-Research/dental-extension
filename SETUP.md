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

`external/HierarchicalDet/detectron2` is a customised copy of Detectron2 **with
no `setup.py` and no compiled ops** (`csrc/`, `.cu` files). Detectron2 normally
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

Most teams get further faster with option 1. Whoever does this: write the exact
sequence that worked into this file so nobody else re-derives it.

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

HierarchicalDet also vendors its own `pycocotools`. If you `pip install
pycocotools` as well you can get a clash. If COCO eval imports break, prefer the
vendored one or uninstall the pip version.

### Backbone weights

The Swin-L weights are not in the repo. Grab
`swin_base_patch4_window7_224_22k.pkl` and put it in `models_weights/`, then make
sure `configs/default.yaml:model.backbone_weights` points at it. (The upstream
DiffusionDet / Swin-Transformer repos document where these come from; the config
in `external/HierarchicalDet/configs/` shows the exact filename it expects.)

### Kaggle specifics

The proposal requires this to run on Kaggle compute, so sort it out on Kaggle
early rather than discovering a blocker in Phase 3.

- Kaggle ships a fixed torch+CUDA per image. Match Detectron2 to it; do not
  upgrade torch.
- Kaggle notebooks have limited internet — enable it in notebook settings, or
  attach the repo and the DENTEX data as Kaggle Datasets and run offline (which
  is closer to the real deployment story anyway).
- GPU sessions are time-limited. HierarchicalDet's config runs 40k iterations at
  batch size 2, which will not finish in one session. Plan for checkpointing and
  resuming, and benchmark iteration time before committing to a full run.

### Sanity check for Track B

Once installed, confirm the baseline imports and a forward pass runs on a single
image before touching training:

```bash
cd external/HierarchicalDet
python -c "import detectron2; from detectron2 import _C; print('ops ok')"
# then a minimal demo.py / train_net.py dry run per that repo's usage
```

If `import detectron2._C` fails, stop and fix the install — nothing downstream
will work until it resolves.
