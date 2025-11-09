from contextlib import nullcontext
from pathlib import Path
import random
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader

from vit_scratch.data.cifar10 import make_cifar10_dataloaders
from vit_scratch.models.vit import ViT


MODEL_REGISTRY = {
    "vit": ViT,
}

DATA_REGISTRY = {
    "cifar10": make_cifar10_dataloaders,
}

OPTIMIZER_REGISTRY = {
    "adamw": AdamW,
}


def setup_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion,
    optimizer: Optimizer,
    *,
    device: torch.device,
    scaler: GradScaler | None,
    use_amp: bool,
    grad_clip: float | None,
):
    model.train()
    total_loss = 0.0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        autocast_ctx = (
            torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16 if device.type == "cpu" else torch.float16,
            )
            if use_amp
            else nullcontext()
        )
        with autocast_ctx:
            logits = model(images)
            loss = criterion(logits, labels)

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion,
    *,
    device: torch.device,
):
    model.eval()
    correct = total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)

        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    return correct / total


def maybe_save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    scaler: GradScaler | None,
    epoch: int,
    out_dir: str,
):
    ckpt_dir = Path(out_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict() if scaler is not None else None,
        },
        ckpt_path,
    )


def _build_model(model_cfg: Dict[str, Any]) -> nn.Module:
    cfg = model_cfg or {}
    name = cfg.get("name", "vit").lower()
    params = cfg.get("params", {})
    img_size = params.get("img_size")
    if isinstance(img_size, list):
        params["img_size"] = tuple(img_size)
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Known models: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**params)


def _build_dataloaders(data_cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    cfg = data_cfg or {}
    name = cfg.get("name", "cifar10").lower()
    params = cfg.get("params", {})
    if name not in DATA_REGISTRY:
        raise ValueError(f"Unknown dataloader '{name}'. Known loaders: {list(DATA_REGISTRY)}")
    return DATA_REGISTRY[name](**params)


def _build_optimizer(
    opt_cfg: Dict[str, Any],
    model: nn.Module,
) -> Optimizer:
    cfg = opt_cfg or {}
    name = cfg.get("name", "adamw").lower()
    params = cfg.get("params", {})
    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Unknown optimizer '{name}'. Known optimizers: {list(OPTIMIZER_REGISTRY)}")
    return OPTIMIZER_REGISTRY[name](model.parameters(), **params)


def train_app(cfg: Dict[str, Any]):
    seed = cfg.get("seed", 42)
    train_cfg = cfg.get("train", {})
    log_interval = train_cfg.get("log_interval", 1) or 1
    save_every = train_cfg.get("save_every")
    output_dir = train_cfg.get("output_dir")

    setup_seed(seed)
    device = get_device()

    train_loader, test_loader = _build_dataloaders(cfg.get("dataloaders", {}))
    model = _build_model(cfg.get("model", {})).to(device)
    optimizer = _build_optimizer(cfg.get("optimizer", {}), model)

    criterion = nn.CrossEntropyLoss()
    use_amp = train_cfg.get("amp", True)
    scaler = GradScaler(device=device.type) if use_amp else None
    grad_clip = train_cfg.get("grad_clip")
    total_epochs = train_cfg.get("epochs", 1)

    for epoch in range(1, total_epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            grad_clip=grad_clip,
        )
        test_accuracy = evaluate(model, test_loader, criterion, device=device)

        if epoch % log_interval == 0 or epoch == 1:
            print(
                f"[{epoch}/{total_epochs}] "
                f"loss={train_loss:.4f} "
                f"acc={test_accuracy * 100:.2f}%"
            )

        if save_every and output_dir and epoch % save_every == 0:
            maybe_save_checkpoint(model, optimizer, scaler, epoch, output_dir)
