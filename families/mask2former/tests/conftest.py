"""Fixtures of the Mask2Former tests: a tiny random model (fast on a CPU)."""

import pytest
import torch
from fashion_seg_mask2former.network import build_model
from transformers import Mask2FormerConfig, SwinConfig

LABELS = [f"c{k}" for k in range(46)]


@pytest.fixture
def tiny_model():
    """Build a small random Mask2Former with the 46 fashion classes."""
    torch.manual_seed(0)
    # pylint: disable-next=unexpected-keyword-arg  # transformers configs take fields as keyword arguments, built at run time
    config = Mask2FormerConfig(
        # pylint: disable-next=unexpected-keyword-arg  # transformers configs take fields as keyword arguments, built at run time
        backbone_config=SwinConfig(
            embed_dim=16,
            depths=[1, 1, 1, 1],
            num_heads=[1, 1, 1, 1],
            out_features=["stage1", "stage2", "stage3", "stage4"],
        ),
        hidden_dim=32,
        feature_size=32,
        mask_feature_size=32,
        decoder_layers=3,
        encoder_layers=1,
        dim_feedforward=64,
        encoder_feedforward_dim=64,
        num_attention_heads=2,
        num_queries=8,
        train_num_points=64,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id={label: i for i, label in enumerate(LABELS)},
    )
    return build_model(LABELS, config=config)
