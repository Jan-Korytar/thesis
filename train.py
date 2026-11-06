import argparse
from pathlib import Path
import random
from typing import Any
import numpy as np
import torch
import torch.nn as nn
from data.dataset import build_dataloaders, create_dataloader
from data.generator import generate_bates_data, normalize_features, price_bates_call, sample_parameters
from models import build_model, predict_mc_dropout
from utils.config_parser import load_config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.set_grad_enabled(is_train):
        for bx, by in loader:
            bx = bx.to(device)
            by = by.to(device)

            if is_train:
                optimizer.zero_grad()

            pred = model(bx)
            loss = criterion(pred, by)

            if is_train:
                loss.backward()
                optimizer.step()

            batch_size = len(bx)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    assert total_samples > 0
    return total_loss / total_samples


def get_val_bounds(bounds: dict[str, list[float]]) -> dict[str, list[float]]:
    val_b = {k: list(v) for k, v in bounds.items()}
    # Restrict validation moneyness to realistic liquid market strikes K/S in [0.60, 1.65]
    val_b["moneyness"] = [-0.5, 0.5]
    return val_b


def save_checkpoint(model: nn.Module, cfg: dict[str, Any], val_loss: float, path: str) -> None:
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "cfg": cfg,
            "val_loss": val_loss,
        },
        ckpt_path,
    )


def train_passive(cfg: dict[str, Any]) -> nn.Module:
    set_seed(cfg["seed"])
    device_str = cfg["device"]
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")

    passive_cfg = cfg["passive"]
    bounds = cfg["data"]["bounds"]
    x, y = generate_bates_data(
        bounds=bounds,
        n_samples=passive_cfg["n_samples"],
        seed=cfg["seed"],
    )
    x_norm = normalize_features(x, bounds)

    train_loader, val_loader = build_dataloaders(
        x=x_norm,
        y=y,
        train_split=passive_cfg["train_split"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
        seed=cfg["seed"],
    )

    model = build_model(cfg).to(device)
    loss_map = {
        "l1": nn.L1Loss(),
        "mae": nn.L1Loss(),
        "mse": nn.MSELoss(),
    }
    criterion = loss_map[cfg["training"]["loss_fn"].lower()]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    best_val_loss = float("inf")
    epochs = passive_cfg["epochs"]
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = run_epoch(model, val_loader, criterion, None, device)
        print(f"[Passive] Epoch {epoch:03d}/{epochs:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, cfg, val_loss, "checkpoints/best_surrogate.pt")

    save_checkpoint(model, cfg, val_loss, "checkpoints/final_surrogate.pt")
    return model


def train_active(cfg: dict[str, Any]) -> nn.Module:
    set_seed(cfg["seed"])
    device_str = cfg["device"]
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")

    active_cfg = cfg["active"]
    bounds = cfg["data"]["bounds"]
    val_bounds = get_val_bounds(bounds)
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["training"]["num_workers"]

    x_val_raw, y_val = generate_bates_data(val_bounds, active_cfg["val_samples"], seed=cfg["seed"] + 1)
    x_val_norm = normalize_features(x_val_raw, bounds)
    val_loader = create_dataloader(x_val_norm, y_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    x_train_raw, y_train = generate_bates_data(bounds, active_cfg["initial_samples"], seed=cfg["seed"])
    x_train_norm = normalize_features(x_train_raw, bounds)
    train_loader = create_dataloader(x_train_norm, y_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    model = build_model(cfg).to(device)
    loss_map = {
        "l1": nn.L1Loss(),
        "mae": nn.L1Loss(),
        "mse": nn.MSELoss(),
    }
    criterion = loss_map[cfg["training"]["loss_fn"].lower()]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    epochs_per_round = active_cfg["epochs_per_round"]
    rounds = active_cfg["rounds"]
    pool_size = active_cfg["pool_size"]
    query_size = active_cfg["query_size"]
    mc_passes = active_cfg["mc_passes"]

    for epoch in range(1, epochs_per_round + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = run_epoch(model, val_loader, criterion, None, device)
    print(f"[Active Round 0] Samples: {len(x_train_raw)} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    best_val_loss = val_loss
    save_checkpoint(model, cfg, best_val_loss, "checkpoints/best_surrogate.pt")

    for r in range(1, rounds + 1):
        x_pool_raw = sample_parameters(bounds, pool_size, seed=cfg["seed"] + 100 * r)
        x_pool_norm = normalize_features(x_pool_raw, bounds)

        _, variance = predict_mc_dropout(model, x_pool_norm.to(device), mc_passes)
        uncertainty = variance.squeeze(1).cpu()

        top_indices = torch.topk(uncertainty, k=query_size).indices
        x_acquired_raw = x_pool_raw[top_indices]
        y_acquired = price_bates_call(x_acquired_raw).unsqueeze(1)

        x_train_raw = torch.cat([x_train_raw, x_acquired_raw], dim=0)
        y_train = torch.cat([y_train, y_acquired], dim=0)

        x_train_norm = normalize_features(x_train_raw, bounds)
        train_loader = create_dataloader(x_train_norm, y_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        for epoch in range(1, epochs_per_round + 1):
            train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = run_epoch(model, val_loader, criterion, None, device)
        mean_uncertainty = uncertainty[top_indices].mean().item()
        print(f"[Active Round {r}/{rounds}] Samples: {len(x_train_raw)} | Acquired Uncert: {mean_uncertainty:.6e} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, cfg, best_val_loss, "checkpoints/best_surrogate.pt")

    save_checkpoint(model, cfg, val_loss, "checkpoints/final_surrogate.pt")
    print(f"Model saved -> Best Val Loss: {best_val_loss:.6f} to checkpoints/best_surrogate.pt")
    return model


def train(cfg: dict[str, Any]) -> nn.Module:
    mode = cfg["mode"]
    if mode == "active":
        return train_active(cfg)
    elif mode == "passive":
        return train_passive(cfg)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pathto", "--config", dest="config_path", type=str, default="configs/base_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config_path)
    train(cfg)


if __name__ == "__main__":
    main()
