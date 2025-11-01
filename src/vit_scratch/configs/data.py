from hydra_zen import builds
from vit_scratch.data.cifar import make_cifar10_dataloaders

CIFAR10 = builds(
    make_cifar10_dataloaders,
    batch_size=256,
    num_workers=8,
    aug="randaug",
    pin_memory=True,
    prefetch_factor=4,
    persistent_workers=True,
    populate_full_signature=True,
)
