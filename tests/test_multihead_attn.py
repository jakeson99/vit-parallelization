"""Tests for the Multi-Head Self-Attention module."""

import torch
import pytest

from vit_scratch.models.vit import MultiHeadSelfAttention


@pytest.mark.parametrize(
    "batch, tokens, dim, heads",
    [
        (2, 65, 192, 3),
        (1, 10, 64, 4),
    ],
)
def test_multihead_self_attention_output_shape(batch, tokens, dim, heads):
    """Test the output shape of the multi-head self-attention module."""
    x = torch.randn(batch, tokens, dim)
    attn = MultiHeadSelfAttention(dim=dim, num_heads=heads)
    out = attn(x)
    assert out.shape == (batch, tokens, dim)


def test_multihead_self_attention_requires_divisible_dim():
    """Test that an assertion is raised if dim is not divisible by num_heads."""
    with pytest.raises(AssertionError):
        _ = MultiHeadSelfAttention(dim=100, num_heads=3)


def test_multihead_self_attention_backwards_pass():
    """Test that gradients can be computed through the attention module."""
    B, N, D, H = 2, 16, 64, 4
    x = torch.randn(B, N, D, requires_grad=True)
    attn = MultiHeadSelfAttention(dim=D, num_heads=H)
    out = attn(x).sum()
    out.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_sdpa_and_manual_paths_match():
    """Test that the outputs from SDPA and manual attention computation match.
    Due to differences in implementation, we use identical weights for both to test the codepath."""
    B, N, D, H = 2, 16, 64, 4
    x = torch.randn(B, N, D)

    attn_sdpa = MultiHeadSelfAttention(dim=D, num_heads=H, use_sdpa=True)
    attn_sdpa.eval()
    attn_manual = MultiHeadSelfAttention(dim=D, num_heads=H, use_sdpa=False)
    attn_manual.eval()

    # Copy weights exactly
    attn_manual.qkv.load_state_dict(attn_sdpa.qkv.state_dict())
    attn_manual.proj.load_state_dict(attn_sdpa.proj.state_dict())

    with torch.no_grad():
        out_sdpa = attn_sdpa(x)
        out_manual = attn_manual(x)

    max_diff = (out_sdpa - out_manual).abs().max().item()
    assert torch.allclose(out_sdpa, out_manual, atol=1e-5, rtol=1e-4), (
        f"Max difference: {max_diff}"
    )
