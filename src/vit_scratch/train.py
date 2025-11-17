from contextlib import nullcontext
from pathlib import Path
import random
from copy import deepcopy
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
import mlflow


MODEL_REGISTRY = {
    "vit": ViT,
}

DATA_REGISTRY = {
    "cifar10": make_cifar10_dataloaders,
}

OPTIMIZER_REGISTRY = {"adamw": AdamW}

CRITERION_REGISTRY = {
    "cross_entropy": nn.CrossEntropyLoss,
}


def setup_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


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
    device: torch.device,
    log_every: int = 10,
    epoch_idx: int = 0,
):
    model.train()
    total_loss = correct = 0.0

    for batch_idx, batch in enumerate(dataloader):
        batch = [t.to(device) for t in batch]
        images, labels = batch
        optimizer.zero_grad()

        logits = model(images)

        loss = criterion(logits, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * len(images)

        predictions = torch.argmax(logits, dim=1)

        correct += torch.sum(predictions == labels).item()

        if batch_idx % log_every == 0:
            mlflow.log_metric(
                "batch_loss",
                loss.item(),
                step=epoch_idx * len(dataloader) + batch_idx,
            )

    accuracy = correct / len(dataloader.dataset)
    avg_loss = total_loss / len(dataloader.dataset)

    return avg_loss, accuracy


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

    for batch in dataloader:
        batch = [t.to(device) for t in batch]
        images, labels = batch

        logits = model(images)

        loss = criterion(logits, labels)

        total += loss.item() * len(images)

        predictions = torch.argmax(logits, dim=1)
        correct += torch.sum(predictions == labels).item()

    accuracy = correct / len(dataloader.dataset)
    avg_loss = total / len(dataloader.dataset)

    return avg_loss, accuracy


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


def _to_plain_dict(obj: Any) -> Any:
    """Recursively convert Config/namespace objects to plain Python containers."""
    if isinstance(obj, dict):
        return {k: _to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return obj.__class__(_to_plain_dict(v) for v in obj)
    if hasattr(obj, "__dict__"):
        return {k: _to_plain_dict(v) for k, v in vars(obj).items()}
    return obj


def _build_model(model_cfg: Dict[str, Any]) -> nn.Module:
    cfg = model_cfg or {}
    name = cfg.get("name", "vit").lower()
    params = cfg.get("params", {})
    img_size = params.get("img_size")
    if isinstance(img_size, list):
        params["img_size"] = tuple(img_size)
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Known models: {list(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name](**params)


def _build_dataloaders(
    dataset,
    num_workers,
    aug,
    pin_memory,
    prefetch_factor,
    persistent_workers,
    root,
    batch_size,
):
    if dataset not in DATA_REGISTRY:
        raise ValueError(
            f"Unknown dataloader '{dataset}'. Known loaders: {list(DATA_REGISTRY)}"
        )

    else:
        params = {
            "num_workers": num_workers,
            "batch_size": batch_size,
            "aug": aug,
            "pin_memory": pin_memory,
            "prefetch_factor": prefetch_factor,
            "persistent_workers": persistent_workers,
            "root": root,
        }

        return DATA_REGISTRY[dataset](**params)


def _build_optimizer(
    optimizer_cfg: Any,
    model: nn.Module,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
) -> Optimizer:
    def _default_params():
        params: Dict[str, Any] = {}
        if learning_rate is not None:
            params["lr"] = learning_rate
        if weight_decay is not None:
            params["weight_decay"] = weight_decay
        return params

    name = "adamw"
    params: Dict[str, Any] = {}

    if optimizer_cfg is None:
        params = _default_params()
    elif isinstance(optimizer_cfg, dict):
        name = optimizer_cfg.get("name") or optimizer_cfg.get("optimizer") or "adamw"
        params = dict(optimizer_cfg.get("params") or {})
        if "lr" not in params and learning_rate is not None:
            params["lr"] = learning_rate
        if "weight_decay" not in params and weight_decay is not None:
            params["weight_decay"] = weight_decay
    else:
        name = getattr(optimizer_cfg, "name", None) or getattr(
            optimizer_cfg, "optimizer", "adamw"
        )
        params = _default_params()

    name = name.lower()
    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(
            f"Unknown optimizer '{name}'. Known optimizers: {list(OPTIMIZER_REGISTRY)}"
        )
    return OPTIMIZER_REGISTRY[name](model.parameters(), **params)


def _build_criterion(criterion_cfg: Any) -> nn.Module:
    name = "cross_entropy"
    params: Dict[str, Any] = {}

    if criterion_cfg is None:
        pass
    elif isinstance(criterion_cfg, dict):
        name = criterion_cfg.get("name") or "cross_entropy"
        params = dict(criterion_cfg.get("params") or {})
    else:
        name = getattr(criterion_cfg, "name", None) or "cross_entropy"

    name = name.lower()
    if name not in CRITERION_REGISTRY:
        raise ValueError(
            f"Unknown criterion '{name}'. Known criterions: {list(CRITERION_REGISTRY)}"
        )
    return CRITERION_REGISTRY[name](**params)


def _setup_output_dir(out_dir: str):
    ckpt_dir = Path(out_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


def train(
    trainloader,
    testloader,
    epochs,
    model,
    optimizer,
    criterion,
    device,
    save_every,
    log_every,
    eval_every,
    output_dir,
):
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(
            model,
            trainloader,
            criterion,
            optimizer,
            device=device,
            log_every=log_every,
            epoch_idx=epoch - 1,
        )
        if epoch % eval_every == 0:
            test_loss, test_acc = evaluate(
                model,
                testloader,
                criterion,
                device=device,
            )

        print(
            f"[{epoch}/{epochs}] train_loss={train_loss:.4f} test_loss={test_loss:.4f}"
        )

        mlflow.log_metric("epoch_train_loss", train_loss, step=epoch)
        mlflow.log_metric("epoch_train_acc", train_acc, step=epoch)
        mlflow.log_metric("epoch_test_loss", test_loss, step=epoch)
        mlflow.log_metric("epoch_test_acc", test_acc, step=epoch)

        if save_every and output_dir and epoch % save_every == 0:
            maybe_save_checkpoint(model, optimizer, None, epoch, output_dir)


def run_training(cfg: Any):
    with mlflow.start_run():
        mlflow.log_params(cfg.to_dict())

        seed = cfg.training.seed
        batch_size = cfg.training.batch_size
        setup_seed(seed)

        device = get_device()

        train_loader, test_loader = _build_dataloaders(
            dataset=cfg.data.dataset,
            num_workers=cfg.data.num_workers,
            aug=cfg.data.aug,
            pin_memory=cfg.data.pin_memory,
            prefetch_factor=cfg.data.prefetch_factor,
            persistent_workers=cfg.data.persistent_workers,
            root=cfg.data.root,
            batch_size=batch_size,
        )

        model = _build_model(cfg.model.to_dict()).to(device)

        optimizer = _build_optimizer(
            optimizer_cfg=cfg.optimizer,
            learning_rate=cfg.optimizer.learning_rate,
            weight_decay=cfg.optimizer.weight_decay,
            model=model,
        )

        criterion = _build_criterion(cfg.optimizer.criterion)

        output_dir = _setup_output_dir(cfg.training.out_dir)

        train(
            train_loader,
            test_loader,
            epochs=cfg.training.num_epochs,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            save_every=cfg.training.save_every,
            log_every=cfg.training.log_every,
            eval_every=cfg.training.eval_every,
            output_dir=output_dir,
        )
