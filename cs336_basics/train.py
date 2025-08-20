from dataclasses import dataclass
import os
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb

from cs336_basics.checkpoint import save_checkpoint
from cs336_basics.data_loader import RandomStartBatchSampler, TokensDataset
from cs336_basics.nn_model import TransformerLM
from cs336_basics.nn_utils import *

import logging
logger = logging.getLogger(__name__)

__MAJOR_METRIC_NAME = "val_loss"
__MAJOR_METRIC_GOAL = "minimize"

@dataclass
class TrainParams:
    train_dir_path: str
    valid_dir_path: str
    checkpoint_path: str

    seed: int = 42

    vocab_size: int = 10000
    context_length: int = 256
    d_model: int = 512
    d_ff: int = 1344
    num_layers: int = 4
    num_heads: int = 16
    rope_theta: float = 10000.0

    lr: float = 1e-2
    batch_size: int = 64
    loader_num_workers: int = 8
    max_norm: float = 0
    optimizer_beta1: float = 0.9
    optimizer_beta2: float = 0.999
    optimizer_weight_decay: float = 0.01
    scheduler_t: int = 5
    scheduler_t_mult: int = 1
    scheduler_min_lr: float = 0

    num_epochs: int = 10
    patience: int = 3
    min_delta: int = 1e-4

    device: str = "cuda"

@dataclass
class TrainState:
    epoch: int
    best_metric: float
    epochs_no_improve: int = 0

def train_model(params:TrainParams):
    set_seed(params.seed)
    state = TrainState(epoch=0, best_metric=float("inf"), epochs_no_improve=0)

    train_loader, valid_loader = get_dataloaders(params)
    model = TransformerLM(
        vocab_size=params.vocab_size,
        context_length=params.context_length,
        d_model=params.d_model,
        d_ff=params.d_ff,
        num_layers=params.num_layers,
        num_heads=params.num_heads,
        rope_theta=params.rope_theta,
        device=params.device
    )
    optimizer = AdamW(
        model.parameters(),
        lr=params.lr,
        betas=(params.optimizer_beta1, params.optimizer_beta2),
        weight_decay=params.optimizer_weight_decay
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=params.scheduler_t,
        T_mult=params.scheduler_t_mult,
        eta_min=params.scheduler_min_lr,
    )
    scaler = torch.amp.GradScaler()

    init_wandb(params)
    wandb.watch(model, log="all", log_freq=200)

    total_params, trainable_params = get_model_size(model)
    logger.info(f"Starting training with parameters: total_params={total_params}, trainable_params={trainable_params}")

    # ========= 训练循环（含早停与最优模型保存）=========
    best_path: Path | None = None

    for epoch in range(1, params.num_epochs + 1):
        train_metrics = train_one_epoch(
            epoch,
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            torch.device(params.device),
            params
        )
        val_metrics = validate(
            epoch,
            model,
            valid_loader,
            torch.device(params.device),
        )

        current_lr = optimizer.param_groups[0]["lr"]

        wandb.log({**train_metrics, **val_metrics, "epoch": epoch, "lr": current_lr})

        # 选择指标做早停与保存
        current_metric = val_metrics[__MAJOR_METRIC_NAME]
        improved = (-1 if __MAJOR_METRIC_GOAL == "minimize" else 1) * (current_metric - state.best_metric) > \
            params.min_delta * abs(current_metric)

        state.epoch = epoch
        if improved:
            state.best_metric = current_metric
            state.epochs_no_improve = 0

            best_path = save_checkpoint_only_best(
                save_dir=Path(params.checkpoint_path),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                params=params,
                state=state
            )

            wandb.summary["best_epoch"] = epoch
            wandb.summary["best_metric"] = state.best_metric
        else:
            state.epochs_no_improve += 1

        if state.epochs_no_improve >= params.patience:
            print(f"Early stopping at epoch {epoch}: no improvement in {params.patience} epochs.")
            break

    print(f"Training finished. Best {__MAJOR_METRIC_NAME}: {state.best_metric:.6f}")

    wandb.finish()


def init_random(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If you are using multi-GPU.

def init_wandb(params: TrainParams) -> wandb.Run:
    run_name = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    return wandb.init(project="cs336-ass1", name=run_name, config={
        "vocab_size": params.vocab_size,
        "context_length": params.context_length,
        "d_model": params.d_model,
        "d_ff": params.d_ff,
        "num_layers": params.num_layers,
        "num_heads": params.num_heads,
        "batch_size": params.batch_size,
        "learning_rate": params.lr,
        "max_norm": params.max_norm,
    })

# ========== 训练与验证 ==========
def train_one_epoch(
    epoch: int,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingWarmRestarts,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    train_params: TrainParams
) -> dict[str, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total_tokens = 0
    total_cases = 0
    max_grad = 0

    pbar = tqdm(loader, desc=f"Train | Epoch {epoch}", leave=False)
    for i, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda"):
            outputs = model(inputs)
            loss = cross_entropy(outputs, targets)
            loss = loss.mean()

        scaler.scale(loss).backward()
        norm_grad = gradient_clipping(model.parameters(), train_params.max_norm)
        scaler.step(optimizer)
        scaler.update()

        scheduler.step(epoch + (i + 1) / len(loader))

        running_loss += loss.item() * inputs.size(0)
        max_grad = max(max_grad, norm_grad)
        preds = outputs.argmax(dim=-1)
        correct += preds.eq(targets).sum().item()
        total_tokens += targets.numel()
        total_cases += inputs.size(0)

        pbar.set_postfix(loss=running_loss / total_cases, acc=correct / total_tokens, grad=norm_grad)

    return {
        "train_loss": running_loss / total_cases,
        "train_acc": correct / total_tokens,
        "train_max_grad": max_grad
    }


@torch.no_grad()
def validate(
    epoch: int,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    running_loss = 0.0
    running_ppl = 0.0
    correct = 0
    total_tokens = 0
    total_cases = 0

    pbar = tqdm(loader, desc=f"Valid | Epoch {epoch}", leave=False)
    for inputs, targets in pbar:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = cross_entropy(outputs, targets)
        running_loss += loss.sum().item()

        ppl = perplexity(outputs, targets)  
        running_ppl += ppl.sum().item()

        preds = outputs.argmax(dim=-1)
        correct += preds.eq(targets).sum().item()
        total_tokens += targets.numel()
        total_cases += inputs.size(0)

        pbar.set_postfix(
            loss=running_loss / total_cases,
            acc=correct / total_tokens,
            ppl=running_ppl / total_cases
        )

    return {
        "val_loss": running_loss / total_cases,
        "val_acc": correct / total_tokens,
        "val_ppl": running_ppl / total_cases
    }

# ========== 工具函数 ==========
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def get_dataloaders(
    params: TrainParams
) -> tuple[DataLoader, DataLoader]:
    train_ds = TokensDataset(data=np.load(f"{params.train_dir_path}/tokens.npy"), context_length=params.context_length)
    valid_ds = TokensDataset(data=np.load(f"{params.valid_dir_path}/tokens.npy"), context_length=params.context_length)
    train_loader = DataLoader(
        train_ds,
        batch_sampler=RandomStartBatchSampler(len(train_ds), batch_size=params.batch_size),
        pin_memory=True,
        num_workers=params.loader_num_workers,
        persistent_workers=(params.loader_num_workers > 0),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=params.batch_size,
        shuffle=False,
        num_workers=params.loader_num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(params.loader_num_workers > 0),
    )
    return train_loader, valid_loader

def save_checkpoint_only_best(
    save_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingWarmRestarts,
    params: TrainParams,
    state: TrainState
) -> Path:
    """
    只保留当前最优模型：保存新 best，删除旧 best。
    返回新的 best 路径。
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = f"best_epoch{state.epoch:03d}_{__MAJOR_METRIC_NAME}-{state.best_metric:.4f}.pt"
    ckpt_path = save_dir / ckpt_name
    save_checkpoint(
        ckpt_path,
        model,
        optimizer,
        scheduler,
        train_param=params,
        train_state=state
    )
    logger.info(f"Saved best checkpoint to {ckpt_path}")

    # 删除旧同一目录下其他.pt
    for old_ckpt in save_dir.glob("best_epoch*.pt"):
        if old_ckpt != ckpt_path:
            try:
                old_ckpt.unlink()
            except Exception as e:
                print(f"[WARN] Failed to delete old checkpoint: {old_ckpt} ({e})")

    return ckpt_path
