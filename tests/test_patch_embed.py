"""Tests for the Patch Embedding module."""

import torch
import pytest

from vit_scratch.models.vit import PatchEmbed

# get seed from TrainConf
from vit_scratch.configs.train import TrainConf


torch.manual_seed(TrainConf.seed)


@pytest.mark.parametrize(
    "B, C, H, W, P, D, flatten, expected_shape",
    [
        (2, 3, 32, 32, 4, 192, True, (2, (32 // 4) * (32 // 4), 192)),
        (2, 3, 32, 32, 4, 192, False, (2, 192, 32 // 4, 32 // 4)),
    ],
)
def test_patch_embed_shapes(B, C, H, W, P, D, flatten, expected_shape):
    x = torch.randn(B, C, H, W)
    patch_embed = PatchEmbed(
        img_size=(H, W), patch_size=P, in_chans=C, embed_dim=D, flatten=flatten
    )
    out = patch_embed(x)
    assert out.shape == expected_shape


def test_raises_on_invalid_image_size():
    with pytest.raises(AssertionError):
        _ = PatchEmbed(img_size=(30, 32), patch_size=4, in_chans=3, embed_dim=192)
