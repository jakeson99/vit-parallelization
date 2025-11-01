import torch
from torch import nn
from typing import Optional, Tuple


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
