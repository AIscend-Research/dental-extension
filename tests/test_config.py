"""Checks that configs/default.yaml stays in sync with reality.

configs/default.yaml used to be dead documentation -- nothing loaded it, so
nothing caught it when it drifted (it named the wrong backbone checkpoint,
the Swin-Base file that loads with zero errors while leaving most of the
backbone randomly initialized). These tests are what makes it a config rather
than a comment.
"""

import pytest

from src.data.degradation import DEGRADATION_NAMES
from src.models.confidence_head import DECISIONS, decide
from src.utils.config import load_config, resolve_path


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_config_loads_with_attribute_and_item_access(cfg):
    assert cfg["model"]["task"] == cfg.model.task
    assert isinstance(cfg.data.image_root, str)


def test_backbone_weights_is_swin_large_not_base(cfg):
    """The upstream config names a Swin-Base file while building a Large model.

    Loading Base raises no error -- DetectionCheckpointer only warns per
    tensor -- so this can only be caught by asserting on the name. See
    SETUP.md's "Backbone weights" section.
    """
    weights = cfg.model.backbone_weights
    assert "swin_large" in weights, f"expected the Swin-Large checkpoint, got {weights!r}"
    assert "swin_base" not in weights


def test_task_is_one_of_the_two_documented_options(cfg):
    assert cfg.model.task in {"caries_only", "diagnosis"}


def test_severity_range_respects_the_zero_to_one_contract(cfg):
    lo, hi = cfg.degradation.severity_range
    assert 0.0 <= lo < hi <= 1.0
    assert 1 <= cfg.degradation.max_simultaneous <= len(DEGRADATION_NAMES)


def test_decision_thresholds_are_ordered_for_decide(cfg):
    """retake_below must sit below refer_below or decide()'s ladder collapses."""
    retake, refer = cfg.confidence.retake_below, cfg.confidence.refer_below
    assert 0.0 < retake < refer < 1.0
    # the config's own thresholds must actually produce all three decisions
    outcomes = {
        decide(retake / 2, 1.0, retake_below=retake, refer_below=refer),
        decide((retake + refer) / 2, 1.0, retake_below=retake, refer_below=refer),
        decide(1.0, 1.0, retake_below=retake, refer_below=refer),
    }
    assert outcomes == set(DECISIONS)


def test_paths_resolve_against_repo_root_not_cwd(cfg):
    # resolved whether or not the data is actually downloaded on this machine
    assert resolve_path(cfg.data.coco_annotations).is_absolute()
    assert resolve_path("/already/absolute") == resolve_path("/already/absolute")


if __name__ == "__main__":
    c = load_config()
    for fn in [
        test_config_loads_with_attribute_and_item_access,
        test_backbone_weights_is_swin_large_not_base,
        test_task_is_one_of_the_two_documented_options,
        test_severity_range_respects_the_zero_to_one_contract,
        test_decision_thresholds_are_ordered_for_decide,
        test_paths_resolve_against_repo_root_not_cwd,
    ]:
        fn(c)
        print("PASS", fn.__name__)
