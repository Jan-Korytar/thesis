import argparse
import copy
from typing import Any
import optuna
import torch
import torch.nn as nn
from data.dataset import build_dataloaders
from data.generator import generate_bates_data, normalize_features
from models import build_model
from train import run_epoch, set_seed
from utils.config_parser import load_config


def objective(trial: optuna.Trial, base_cfg: dict[str, Any], train_loader: Any, val_loader: Any, device: torch.device) -> float:
    cfg = copy.deepcopy(base_cfg)

    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    n_layers = trial.suggest_int("n_layers", 2, 4)
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
    activation = trial.suggest_categorical("activation", ["silu", "gelu"])

    cfg["training"]["learning_rate"] = lr
    cfg["model"]["hidden_layers"] = [hidden_dim] * n_layers
    cfg["model"]["activation"] = activation

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

    epochs = cfg["passive"]["epochs"] if "passive" in cfg else 20
    val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = run_epoch(model, val_loader, criterion, None, device)
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pathto", "--config", dest="config_path", type=str, default="configs/base_config.yaml")
    parser.add_argument("--n-trials", dest="n_trials", type=int, default=20)
    args = parser.parse_args()

    cfg = load_config(args.config_path)
    set_seed(cfg["seed"])

    device_str = cfg["device"]
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")

    bounds = cfg["data"]["bounds"]
    n_samples = cfg["passive"]["n_samples"] if "passive" in cfg else 20000
    train_split = cfg["passive"]["train_split"] if "passive" in cfg else 0.8

    x, y = generate_bates_data(
        bounds=bounds,
        n_samples=n_samples,
        seed=cfg["seed"],
    )
    x_norm = normalize_features(x, bounds)

    train_loader, val_loader = build_dataloaders(
        x=x_norm,
        y=y,
        train_split=train_split,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
        seed=cfg["seed"],
    )

    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, cfg, train_loader, val_loader, device),
        n_trials=args.n_trials,
    )

    print("Best Trial:")
    print(f"  Value: {study.best_trial.value:.6f}")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
