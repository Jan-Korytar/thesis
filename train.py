import argparse
from pathlib import Path
import random
from typing import Any
import numpy as np
import torch
import torch.nn as nn
from data.dataset import build_dataloaders, create_dataloader
from data.generator import (
    generate_bates_bsiv_data,
    generate_bates_data,
    invert_bsiv,
    normalize_features,
    price_bates_call,
    sample_parameters,
)
from models import build_model, predict_mc_dropout
from models.losses import BSIVLoss
from utils.config_parser import load_config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_vega_weights(x_raw: torch.Tensor, eps: float = 0.01, max_weight: float = 20.0) -> torch.Tensor:
    m = x_raw[:, 0]
    tau = torch.clamp(x_raw[:, 1] / 365.0, min=1e-4)
    v_t = torch.clamp(x_raw[:, 2], min=1e-4)
    r = x_raw[:, 3]
    q = x_raw[:, 4]

    sig = torch.sqrt(v_t)
    sqrt_tau = torch.sqrt(tau)
    d1 = (m + (r - q + 0.5 * sig**2) * tau) / (sig * sqrt_tau)
    phi = (1.0 / np.sqrt(2.0 * np.pi)) * torch.exp(-0.5 * d1**2)
    vega = torch.exp(-q * tau) * sqrt_tau * phi

    raw_weights = 1.0 / (vega + eps)
    norm_weights = raw_weights / raw_weights.mean()
    return torch.clamp(norm_weights, min=0.1, max=max_weight).unsqueeze(1).float()


def compute_pool_curvature(model: nn.Module, x_norm: torch.Tensor, batch_size: int = 1024) -> torch.Tensor:
    curvatures = []
    model.eval()
    for i in range(0, len(x_norm), batch_size):
        bx = x_norm[i : i + batch_size].clone().detach().requires_grad_(True)
        pred = model(bx)
        grad1 = torch.autograd.grad(pred.sum(), bx, create_graph=True)[0][:, 0]
        grad2 = torch.autograd.grad(grad1.sum(), bx, create_graph=False)[0][:, 0]
        curvatures.append(torch.abs(grad2).detach().cpu())
        del bx, pred, grad1, grad2

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return torch.cat(curvatures, dim=0)


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    use_weights: bool = False,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
) -> float:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_samples = 0
    device_type = "cuda" if device.type == "cuda" else "cpu"

    with torch.set_grad_enabled(is_train):
        for bx, by, bw in loader:
            bx = bx.to(device)
            by = by.to(device)
            bw = bw.to(device)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                pred = model(bx)
                if is_train and use_weights:
                    loss = torch.mean(bw * torch.abs(pred - by))
                else:
                    loss = criterion(pred, by)

            if is_train:
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            batch_size = len(bx)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    assert total_samples > 0
    return total_loss / total_samples


def get_val_bounds(bounds: dict[str, list[float]]) -> dict[str, list[float]]:
    val_b = {k: list(v) for k, v in bounds.items()}
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

    use_amp = cfg["training"]["use_amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    is_bsiv = cfg.get("target_type") == "bsiv"

    passive_cfg = cfg["passive"]
    bounds = cfg["data"]["bounds"]
    if is_bsiv:
        x, y = generate_bates_bsiv_data(bounds, passive_cfg["n_samples"], seed=cfg["seed"])
    else:
        x, y = generate_bates_data(bounds, passive_cfg["n_samples"], seed=cfg["seed"])
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
        "bsiv": BSIVLoss(),
    }
    criterion = loss_map[cfg["training"]["loss_fn"].lower()]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg["training"]["lr_factor"],
        patience=cfg["training"]["lr_patience"],
        min_lr=cfg["training"]["min_learning_rate"],
    )

    best_val_loss = float("inf")
    epochs = passive_cfg["epochs"]
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler, use_amp=use_amp)
        val_loss = run_epoch(model, val_loader, criterion, None, device, scaler=None, use_amp=False)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"[Passive] Epoch {epoch:03d}/{epochs:03d} | LR: {current_lr:.1e} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, cfg, val_loss, "checkpoints/best_surrogate.pt")

    save_checkpoint(model, cfg, val_loss, "checkpoints/final_surrogate.pt")
    return model


def train_active(cfg: dict[str, Any]) -> nn.Module:
    set_seed(cfg["seed"])
    device_str = cfg["device"]
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")

    use_amp = cfg["training"]["use_amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    is_bsiv = cfg.get("target_type") == "bsiv"

    active_cfg = cfg["active"]
    bounds = cfg["data"]["bounds"]
    val_bounds = get_val_bounds(bounds)
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["training"]["num_workers"]
    use_vega_loss = cfg["training"]["use_vega_loss"] and not is_bsiv

    if is_bsiv:
        x_val_raw, y_val = generate_bates_bsiv_data(val_bounds, active_cfg["val_samples"], seed=cfg["seed"] + 1)
        x_train_raw, y_train = generate_bates_bsiv_data(bounds, active_cfg["initial_samples"], seed=cfg["seed"])
    else:
        x_val_raw, y_val = generate_bates_data(val_bounds, active_cfg["val_samples"], seed=cfg["seed"] + 1)
        x_train_raw, y_train = generate_bates_data(bounds, active_cfg["initial_samples"], seed=cfg["seed"])

    x_val_norm = normalize_features(x_val_raw, bounds)
    val_loader = create_dataloader(x_val_norm, y_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    w_train = compute_vega_weights(x_train_raw) if use_vega_loss else None
    x_train_norm = normalize_features(x_train_raw, bounds)
    train_loader = create_dataloader(x_train_norm, y_train, w_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    model = build_model(cfg).to(device)
    loss_map = {
        "l1": nn.L1Loss(),
        "mae": nn.L1Loss(),
        "mse": nn.MSELoss(),
        "bsiv": BSIVLoss(),
    }
    criterion = loss_map[cfg["training"]["loss_fn"].lower()]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg["training"]["lr_factor"],
        patience=cfg["training"]["lr_patience"],
        min_lr=cfg["training"]["min_learning_rate"],
    )

    epochs_per_round = active_cfg["epochs_per_round"]
    rounds = active_cfg["rounds"]
    pool_size = active_cfg["pool_size"]
    query_size = active_cfg["query_size"]
    mc_passes = active_cfg["mc_passes"]
    target_val_loss = active_cfg["target_val_loss"]
    early_stopping_patience = active_cfg["early_stopping_patience"]
    curvature_weight = active_cfg["curvature_weight"]

    for epoch in range(1, epochs_per_round + 1):
        train_loss = run_epoch(
            model, train_loader, criterion, optimizer, device, use_weights=use_vega_loss, scaler=scaler, use_amp=use_amp
        )
    val_loss = run_epoch(model, val_loader, criterion, None, device, use_weights=False, scaler=None, use_amp=False)
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]["lr"]
    print(f"[Active Round 00] Samples: {len(x_train_raw):5d} | LR: {current_lr:.1e} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    best_val_loss = val_loss
    save_checkpoint(model, cfg, best_val_loss, "checkpoints/best_surrogate.pt")
    patience_counter = 0

    for r in range(1, rounds + 1):
        x_pool_raw = sample_parameters(bounds, pool_size, seed=cfg["seed"] + 100 * r)
        x_pool_norm = normalize_features(x_pool_raw, bounds)

        # 1. Epistemic uncertainty
        _, variance = predict_mc_dropout(model, x_pool_norm.to(device), mc_passes)
        uncertainty = variance.squeeze(1).cpu()

        # 2. Surface curvature
        curvature = compute_pool_curvature(model, x_pool_norm.to(device), batch_size=1024)

        # 3. Combined acquisition score
        u_norm = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min() + 1e-8)
        c_norm = (curvature - curvature.min()) / (curvature.max() - curvature.min() + 1e-8)
        score = u_norm + curvature_weight * c_norm

        top_indices = torch.topk(score, k=query_size).indices
        x_acquired_raw = x_pool_raw[top_indices]

        if is_bsiv:
            prices = price_bates_call(x_acquired_raw)
            m = x_acquired_raw[:, 0]
            tau = x_acquired_raw[:, 1] / 365.0
            r_rate = x_acquired_raw[:, 3]
            q_div = x_acquired_raw[:, 4]
            bsiv_acquired = invert_bsiv(prices, m, tau, r_rate, q_div).unsqueeze(1)
            valid = torch.isfinite(bsiv_acquired.squeeze(1)) & (bsiv_acquired.squeeze(1) > 0.01) & (bsiv_acquired.squeeze(1) < 4.0)
            x_acquired_raw = x_acquired_raw[valid]
            y_acquired = bsiv_acquired[valid]
        else:
            y_acquired = price_bates_call(x_acquired_raw).unsqueeze(1)

        x_train_raw = torch.cat([x_train_raw, x_acquired_raw], dim=0)
        y_train = torch.cat([y_train, y_acquired], dim=0)

        w_train = compute_vega_weights(x_train_raw) if use_vega_loss else None
        x_train_norm = normalize_features(x_train_raw, bounds)
        train_loader = create_dataloader(x_train_norm, y_train, w_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        for epoch in range(1, epochs_per_round + 1):
            train_loss = run_epoch(
                model, train_loader, criterion, optimizer, device, use_weights=use_vega_loss, scaler=scaler, use_amp=use_amp
            )
        val_loss = run_epoch(model, val_loader, criterion, None, device, use_weights=False, scaler=None, use_amp=False)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        mean_uncert = uncertainty[top_indices].mean().item()
        mean_curv = curvature[top_indices].mean().item()
        print(f"[Active Round {r:02d}/{rounds}] Samples: {len(x_train_raw):5d} | LR: {current_lr:.1e} | Uncert: {mean_uncert:.2e} | Curv: {mean_curv:.2e} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, cfg, best_val_loss, "checkpoints/best_surrogate.pt")
        else:
            patience_counter += 1

        if val_loss <= target_val_loss:
            print(f"\nTarget validation loss achieved ({val_loss:.6f} <= {target_val_loss:.6f})! Stopping training.")
            break

        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered: No validation improvement for {early_stopping_patience} consecutive rounds.")
            break

    save_checkpoint(model, cfg, val_loss, "checkpoints/final_surrogate.pt")
    print(f"Training complete. Best checkpoint saved to checkpoints/best_surrogate.pt (Best Val Loss: {best_val_loss:.6f})")
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
