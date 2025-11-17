import pydra

import torch.nn as nn


class ModelConfig(pydra.Config):
    def __init__(self):
        self.img_size: tuple[int, int] = (32, 32)
        self.patch_size: int = 4
        self.in_chans: int = 3
        self.embed_dim: int = 192
        self.depth: int = 12
        self.num_heads: int = 3
        self.mlp_ratio: float = 4.0
        self.qkv_bias: bool = True
        self.attn_drop: float = 0.0
        self.proj_drop: float = 0.0
        self.drop: float = 0.0
        self.num_classes: int = 10
        self.flatten: bool = True
        self.norm_layer: nn.Module | None = None


class TrainingConfig(pydra.Config):
    def __init__(self):
        self.batch_size: int = 256
        self.num_epochs: int = 100
        self.seed: int = 985743
        self.log_every: int = 50
        self.eval_every: int = 1
        self.save_every: int = 10
        self.out_dir: str = "./checkpoints"


class OptimizerConfig(pydra.Config):
    def __init__(self):
        self.optimizer: str = "adamw"
        self.learning_rate: float = 0.001
        self.weight_decay: float = 0.0001
        self.grad_clip: float | None = None
        self.use_amp: bool = True
        self.criterion: str = "cross_entropy"


class DataConfig(pydra.Config):
    def __init__(self):
        self.dataset: str = "cifar10"
        self.num_workers: int = 8
        self.aug: str | None = "randaug"
        self.pin_memory: bool = True
        self.prefetch_factor: int | None = None
        self.persistent_workers: bool = True
        self.root: str = "./data"


class ExperimentConfig(pydra.Config):
    def __init__(self):
        self.model: ModelConfig = ModelConfig()
        self.training: TrainingConfig = TrainingConfig()
        self.data: DataConfig = DataConfig()
        self.optimizer: OptimizerConfig = OptimizerConfig()
