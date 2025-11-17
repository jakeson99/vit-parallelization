import torch
import pytest

from vit_scratch.models.vit import ViT


def test_vit_forward_shape():
    """Test the output shape of the ViT model."""
    B, C, H, W = 4, 3, 32, 32

    model = ViT(
        img_size=(H, W),
        patch_size=4,
        in_chans=C,
        num_classes=10,
        embed_dim=192,
        depth=3,
        num_heads=3,
    )
    x = torch.randn(B, C, H, W)
    out = model(x)

    assert out.shape == (B, 10), f"Expected output shape {(B, 10)}, but got {out.shape}"


def test_vit_backward():
    """Test that gradients can be computed through the ViT model."""
    B, C, H, W = 2, 3, 32, 32

    model = ViT(
        img_size=(H, W),
        patch_size=4,
        in_chans=C,
        num_classes=10,
        embed_dim=64,
        depth=2,
        num_heads=4,
    )
    x = torch.randn(B, C, H, W, requires_grad=True)
    out = model(x).sum()
    out.backward()

    assert x.grad is not None, "Gradients were not computed for the input"
    assert not torch.isnan(x.grad).any(), "Gradients contain NaN values"


def test_vit_cls_token_effect():
    """Test that modifying the cls token changes the output."""
    B, C, H, W = 1, 3, 32, 32

    model = ViT(
        img_size=(H, W),
        patch_size=4,
        in_chans=C,
        num_classes=10,
        embed_dim=64,
        depth=1,
        num_heads=4,
    )
    x = torch.randn(B, C, H, W)

    with torch.no_grad():
        out_with_cls1 = model(x)

        # modify a single cls token feature (LayerNorm removes uniform shifts)
        model.cls_token[..., 0].add_(0.1)

        out_with_cls2 = model(x)

        assert not torch.allclose(out_with_cls1, out_with_cls2), (
            "Outputs with modified cls token should differ"
        )
