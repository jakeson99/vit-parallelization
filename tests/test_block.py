"""Unit tests for the Transformer Block module."""

import torch
from vit_scratch.models.vit import Block


def test_block_output_shape():
    """Test the output shape of the Transformer Block."""
    B, N, D, H = 2, 64, 192, 3
    x = torch.randn(B, N, D)
    blk = Block(dim=D, num_heads=H)
    out = blk(x)
    assert out.shape == (B, N, D), (
        f"Expected output shape {(B, N, D)}, but got {out.shape}"
    )


def test_block_backward():
    """Test that gradients can be computed through the Block."""
    B, N, D, H = 2, 16, 64, 4
    x = torch.randn(B, N, D, requires_grad=True)
    blk = Block(dim=D, num_heads=H)
    out = blk(x).sum()
    out.backward()
    assert x.grad is not None, "Gradients were not computed for the input"
    assert not torch.isnan(x.grad).any(), "Gradients contain NaN values"
