from hydra_zen import builds
from vit_scratch.models.vit import ViT

# wraps callable, capturing defaults in Hydra config object
ViT_tiny = builds(
    ViT,
    img_size=32,
    patch_size=4,
    embed_dim=192,
    depth=12,
    num_heads=3,
    mlp_ratio=4.0,
    drop_rate=0.0,
    drop_path_rate=0.1,
    num_classes=10,  # CIFAR-10
    populate_full_signature=True,
    zen_partial=False,
)

ViT_small = builds(
    ViT,
    img_size=64,
    patch_size=8,
    embed_dim=384,
    depth=12,
    num_heads=6,
    mlp_ratio=4.0,
    drop_rate=0.0,
    drop_path_rate=0.1,
    num_classes=200,  # Tiny-ImageNet
    populate_full_signature=True,
    zen_partial=False,
)
