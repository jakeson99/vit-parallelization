import torch
from torch import nn
from typing import Optional, Tuple
from torch.nn import functional as F


class ViT:
    """
    Vision Transformer (ViT) model.

    This class implements the Vision Transformer architecture for image classification tasks.
    Usage:
        model = ViT()
        # Add methods and attributes as needed.
    """

    def forward(self, *args, **kwargs):
        """Forward pass for the ViT model (to be implemented)."""
        pass


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism.

    This class implements the multi-head self-attention mechanism used in transformer architectures.
    Usage:
        attention = MultiHeadSelfAttention()
        # Add methods and attributes as needed.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_sdpa: bool = False,
    ):
        super().__init__()

        assert dim % num_heads == 0, "Dimension must be divisible by number of heads."
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        # setup one set of q, k, v projections for all heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # by default use PyTorch's scaled dot-product attention if available
        if use_sdpa is None:
            self.use_sdpa = hasattr(F, "scaled_dot_product_attention")
        else:
            self.use_sdpa = use_sdpa

        self.reset_parameters()

    def reset_parameters(self):
        """Re-initialize learnable parameters."""
        # Initialise weights and biases
        # for ViT, standard is to use truncated normal initialisation with std=0.02
        nn.init.trunc_normal_(self.qkv.weight, std=0.02)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, N, D = x.shape

        assert D == self.dim, f"Expected input dim {self.dim}, instead got {D}."

        # qkv: (B, N, 3*D) -> split -> (3, B, H, N, head_dim)
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(dim=0)  # each: (B, H, N, head_dim)

        if self.use_sdpa:
            # use PyTorch's built-in scaled dot-product attention
            attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=False,
            )  # (B, H, N, head_dim)

        else:
            # manual implementation of scaled dot-product attention
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, N, N)

            if attn_mask is not None:
                # support both bool and additive masks
                if attn_mask.dtype == torch.bool:
                    scores = scores.masked_fill(~attn_mask, float("-inf"))
                else:
                    scores = scores + attn_mask
            attn = scores.softmax(dim=-1)  # (B, H, N, N)
            attn = self.attn_drop(attn)
            attn = torch.matmul(attn, v)  # (B, H, N, head_dim)

        # merge all heads
        attn = attn.transpose(1, 2).reshape(B, N, D)  # (B, N, D)
        out = self.proj(attn)
        out = self.proj_drop(out)
        return out


class PatchEmbed(nn.Module):
    """
    Patch embeding divides image up into smaller patches,
    flattens to 1D and then maps via a linear projection to a D dimension tensor.
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (32, 32),
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 192,
        flatten: bool = True,
        norm_layer: Optional[nn.Module] = None,
    ):
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        H, W = img_size

        assert H % patch_size == 0 and W % patch_size == 0, (
            f"Image size {img_size} must be divisable by patch size {patch_size}"
        )

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (H // patch_size, W // patch_size)  # (Grid H, Grid W)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.embed_dim = embed_dim
        self.flatten = flatten

        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

        # use the xavier uniform distribution for initalisation weights instead of default Pytorch ones.
        # helps preserve variance when projecting many pixels at once.
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        assert (H, W) == self.img_size, (
            f"Expected input size {self.img_size}, instead got {H, W}"
        )

        x = self.proj(x)  # (B, D, Grid H, Grid W)
        x = self.norm(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x
