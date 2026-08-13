"""
Standalone training run for the confidence head (Phase 3, item 22 on the roadmap).

Scope, stated honestly up front: `src/models/confidence_head.py`'s ConfidenceHead
is designed to sit on the real detector's image-encoder features (FPN p5, 256
channels). That backbone only has meaningful features once it's trained on
real caries data -- which needs Phase 3's GPU/Kaggle run, not done yet. This
script does NOT wait for that. It trains the *same* ConfidenceHead against a
small, lightweight CNN trunk built just for this script, on real DENTEX images
degraded synthetically via src/data/degradation.py (already built and tested).
This validates the actual thing that was uncertain -- can this architecture
learn to predict degradation type/severity from an image at all -- without
needing the caries detector to exist first.

When the real detector is eventually trained, retrain ConfidenceHead against
its real FPN p5 features instead of this trunk; this script's trunk is a
stand-in for that, not a replacement.

Run from the detector venv (needs torch + opencv):
    source .venv-detector/bin/activate
    python scripts/train_confidence_head_standalone.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.degradation import DEGRADATION_NAMES, apply_degradations
from src.data.dentex import load_coco, patient_level_split
from src.models.confidence_head import ConfidenceHead
from src.utils.config import load_config, resolve_path
from src.utils.seed import set_seed

# Paths and degradation params come from configs/default.yaml rather than being
# duplicated here -- that file is the single place they're recorded, and a copy
# in this script is a copy that silently goes stale.
_CFG = load_config()
IMAGE_ROOT = resolve_path(_CFG.data.image_root)
COCO_JSON = resolve_path(_CFG.data.coco_annotations)
SEVERITY_RANGE = tuple(_CFG.degradation.severity_range)
MAX_SIMULTANEOUS = _CFG.degradation.max_simultaneous
IMG_SIZE = 256
VARIANTS_PER_IMAGE = 4  # 3 degraded severities + 1 clean
N_TRAIN_IMAGES = 200
N_VAL_IMAGES = 50


class TinyTrunk(nn.Module):
    """A small CNN feature extractor, NOT the real detector backbone.

    Stands in for the (currently untrained/meaningless) Swin-L backbone so
    ConfidenceHead's actual task -- predicting degradation severity from an
    image -- can be validated now. Output matches ConfidenceHead's expected
    256-channel input.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


def load_image(file_name: str) -> np.ndarray:
    img = cv2.imread(str(IMAGE_ROOT / file_name), cv2.IMREAD_COLOR)
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE))


def build_examples(file_names: list[str], seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Returns list of (degraded_image_bgr_uint8, label_vector) pairs."""
    rng = random.Random(seed)
    examples = []
    for fn in file_names:
        base = load_image(fn)
        # one clean example (all-zero severities) per image
        examples.append((base, np.zeros(len(DEGRADATION_NAMES), dtype=np.float32)))
        for _ in range(VARIANTS_PER_IMAGE - 1):
            result = apply_degradations(
                base,
                severity_range=SEVERITY_RANGE,
                max_simultaneous=MAX_SIMULTANEOUS,
                seed=rng.randint(0, 1_000_000),
            )
            examples.append((result.image, result.label_vector()))
    return examples


def to_tensor_batch(examples: list[tuple[np.ndarray, np.ndarray]]) -> tuple[torch.Tensor, torch.Tensor]:
    imgs = np.stack([e[0] for e in examples]).astype(np.float32) / 255.0  # (N,H,W,3)
    imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2)  # (N,3,H,W)
    labels = torch.from_numpy(np.stack([e[1] for e in examples]))  # (N, n_degradations)
    return imgs, labels


def main():
    set_seed(0)
    coco = load_coco(COCO_JSON)
    split = patient_level_split(coco, val_frac=0.15, test_frac=0.15, seed=0)
    id_to_name = {im["id"]: im["file_name"] for im in coco["images"]}

    train_ids = split["train"][:N_TRAIN_IMAGES]
    val_ids = split["val"][:N_VAL_IMAGES]
    train_files = [id_to_name[i] for i in train_ids]
    val_files = [id_to_name[i] for i in val_ids]
    print(f"train images: {len(train_files)}, val images: {len(val_files)}")

    print("building training examples (real DENTEX images + synthetic degradation)...")
    train_examples = build_examples(train_files, seed=1)
    val_examples = build_examples(val_files, seed=2)
    print(f"train examples: {len(train_examples)}, val examples: {len(val_examples)}")

    train_x, train_y = to_tensor_batch(train_examples)
    val_x, val_y = to_tensor_batch(val_examples)
    # usability target: cheap derived definition per confidence_head.py's own
    # docstring -- 1 - max severity. Not learned from detector correctness,
    # since there's no trained detector to be correct/incorrect yet.
    train_usability_target = 1.0 - train_y.max(dim=1).values
    val_usability_target = 1.0 - val_y.max(dim=1).values

    trunk = TinyTrunk()
    head = ConfidenceHead(in_features=256)
    params = list(trunk.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)
    loss_fn = nn.SmoothL1Loss()

    n_epochs = 15
    batch_size = 16
    n_train = train_x.shape[0]

    # Validation loss on this task is genuinely unstable epoch to epoch (small
    # dataset, no augmentation beyond the degradation itself -- see
    # docs/phase3_confidence_head_training.md). Reporting whatever the LAST
    # epoch happens to produce therefore makes the headline numbers a coin
    # flip: an unlucky final epoch reports a model 10x worse on val loss than
    # one seen mid-run, which reads as a broken pipeline rather than a noisy
    # one. Keep the best-by-val-loss weights and report those, and say so.
    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None

    print("training...")
    for epoch in range(n_epochs):
        trunk.train()
        head.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            imgs, labels = train_x[idx], train_y[idx]
            usability_target = train_usability_target[idx]

            optimizer.zero_grad()
            feats = trunk(imgs)
            severity_pred, usability_pred = head(feats)
            loss = loss_fn(severity_pred, labels) + loss_fn(usability_pred, usability_target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * imgs.shape[0]
        epoch_loss /= n_train

        trunk.eval()
        head.eval()
        with torch.no_grad():
            val_feats = trunk(val_x)
            val_severity_pred, val_usability_pred = head(val_feats)
            val_loss = (loss_fn(val_severity_pred, val_y)
                        + loss_fn(val_usability_pred, val_usability_target)).item()
            val_mae_severity = (val_severity_pred - val_y).abs().mean().item()
            val_mae_usability = (val_usability_pred - val_usability_target).abs().mean().item()

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch + 1
            best_state = (
                {k: v.detach().clone() for k, v in trunk.state_dict().items()},
                {k: v.detach().clone() for k, v in head.state_dict().items()},
            )
            marker = "  <- best so far"

        print(f"epoch {epoch+1:2d}/{n_epochs}  train_loss={epoch_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_severity_MAE={val_mae_severity:.4f}  "
              f"val_usability_MAE={val_mae_usability:.4f}{marker}")

    # Restore the best-by-val-loss weights before reporting. This is early
    # stopping, so the reported numbers are model-selected on val -- state
    # that alongside them rather than presenting them as held-out-clean.
    if best_state is not None:
        trunk.load_state_dict(best_state[0])
        head.load_state_dict(best_state[1])
    print(f"\nreporting epoch {best_epoch}/{n_epochs} (best val_loss={best_val_loss:.4f}); "
          f"final epoch was {val_loss:.4f}")

    # Quantitative summary over the FULL val set, not just a handful of
    # examples -- this is the honest number to report, not cherry-picked rows.
    trunk.eval()
    head.eval()
    with torch.no_grad():
        val_severity_pred, val_usability_pred = head(trunk(val_x))

    nonclean_mask = val_y.max(dim=1).values > 0
    true_dom = val_y[nonclean_mask].argmax(dim=1)
    pred_dom = val_severity_pred[nonclean_mask].argmax(dim=1)
    dom_acc = (true_dom == pred_dom).float().mean().item()
    chance = 1.0 / len(DEGRADATION_NAMES)

    corr = np.corrcoef(val_usability_pred.numpy(), val_usability_target.numpy())[0, 1]
    clean_mean = val_usability_pred[~nonclean_mask].mean().item()
    degraded_mean = val_usability_pred[nonclean_mask].mean().item()
    degraded_true_mean = val_usability_target[nonclean_mask].mean().item()

    print()
    print("=== full-val-set summary (the honest numbers, not cherry-picked) ===")
    print(f"dominant-degradation accuracy: {dom_acc:.3f}  "
          f"(n={int(nonclean_mask.sum())} non-clean examples, chance level={chance:.3f})")
    print(f"usability score Pearson correlation (pred vs true): {corr:.3f}")
    print(f"mean predicted usability -- CLEAN images: {clean_mean:.3f} (true=1.0)")
    print(f"mean predicted usability -- DEGRADED images: {degraded_mean:.3f} "
          f"(true mean={degraded_true_mean:.3f})")

    print()
    print("sample held-out predictions (dominant degradation, true vs predicted):")
    with torch.no_grad():
        for i in range(min(8, val_x.shape[0])):
            true_label = val_y[i]
            true_dominant = DEGRADATION_NAMES[int(true_label.argmax())] if true_label.max() > 0 else "clean"
            pred_dominant = head.dominant_degradation(val_severity_pred[i])
            print(f"  true={true_dominant:10s} pred={pred_dominant:10s}  "
                  f"pred_usability={val_usability_pred[i].item():.2f} true_usability={val_usability_target[i].item():.2f}")


if __name__ == "__main__":
    main()
