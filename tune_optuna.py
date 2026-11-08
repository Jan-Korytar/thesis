import argparse
import copy
from pathlib import Path
from typing import Any
import optuna
import torch
import torch.nn as nn
import yaml

from data.dataset import create_dataloader
from data.generator import (
    generate_bates_bsiv_data,
    invert_bsiv,
    normalize_features,
    price_bates_call,
    sample_parameters,
)
from models import build_model, predict_mc_dropout
from models.losses import BSIVLoss
from train import compute_pool_curvature, get_val_bounds, run_epoch, set_seed
from utils.config_parser import load_config


def evaluate_trial(trial: optuna.Trial, base_cfg: dict[str, Any], tune_rounds: int, epochs_per_round: int) -> float:
    cfg = copy.deepcopy(base_cfg)

    # 1. Hyperparameter Sampling
    hidden_depth = trial.suggest_int("hidden_depth", 3, 5)
    hidden_dim = trial.suggest_categorical("hidden_dim", [256, 384, 512, 768])
    activation = trial.suggest_categorical("activation", ["gelu", "silu"])
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.10, step=0.02)

    learning_rate = trial.suggest_float("learning_rate", 3e-4, 3e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    lr_factor = trial.suggest_float("lr_factor", 0.3, 0.7, step=0.1)
    batch_size = trial.suggest_categorical("batch_size", [512, 1024, 2048])

    curvature_weight = trial.suggest_float("curvature_weight", 0.5, 3.0, step=0.5)
    mc_passes = trial.suggest_int("mc_passes", 10, 25, step=5)

    # Apply to configuration
    cfg["model"]["hidden_layers"] = [hidden_dim] * hidden_depth
    cfg["model"]["activation"] = activation
    cfg["model"]["dropout_rate"] = dropout_rate

    cfg["training"]["learning_rate"] = learning_rate
    cfg["training"]["weight_decay"] = weight_decay
    cfg["training"]["lr_factor"] = lr_factor
    cfg["training"]["batch_size"] = batch_size

    cfg["active"]["curvature_weight"] = curvature_weight
    cfg["active"]["mc_passes"] = mc_passes
    cfg["active"]["rounds"] = tune_rounds
    cfg["active"]["epochs_per_round"] = epochs_per_round

    # 2. Setup environment & device
    set_seed(cfg["seed"])
    device_str = cfg["device"]
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    use_amp = cfg["training"]["use_amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    bounds = cfg["data"]["bounds"]
    val_bounds = get_val_bounds(bounds)
    active_cfg = cfg["active"]
    val_samples = min(active_cfg["val_samples"], 3000)

    # 3. Validation set
    x_val_raw, y_val = generate_bates_bsiv_data(val_bounds, val_samples, seed=cfg["seed"] + 1, device=device)
    x_val_norm = normalize_features(x_val_raw, bounds)
    val_loader = create_dataloader(x_val_norm, y_val, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initial training pool
    x_train_raw, y_train = generate_bates_bsiv_data(bounds, active_cfg["initial_samples"], seed=cfg["seed"], device=device)
    x_train_norm = normalize_features(x_train_raw, bounds)
    train_loader = create_dataloader(x_train_norm, y_train, batch_size=batch_size, shuffle=True, num_workers=0)

    model = build_model(cfg).to(device)
    criterion = BSIVLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=cfg["training"]["lr_patience"], min_lr=1e-5
    )

    best_val_loss = float("inf")

    # Initial round 0 training
    for _ in range(epochs_per_round):
        run_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler, use_amp=use_amp)
    val_loss = run_epoch(model, val_loader, criterion, None, device, scaler=None, use_amp=False)
    scheduler.step(val_loss)
    best_val_loss = min(best_val_loss, val_loss)
    trial.report(val_loss, step=0)

    # Active rounds
    pool_size = min(active_cfg["pool_size"], 10000)
    query_size = min(active_cfg["query_size"], 800)

    for r in range(1, tune_rounds + 1):
        x_pool_raw = sample_parameters(bounds, pool_size, seed=cfg["seed"] + 100 * r, device=device)
        x_pool_norm = normalize_features(x_pool_raw, bounds)

        # MC Dropout uncertainty
        _, variance = predict_mc_dropout(model, x_pool_norm, mc_passes)
        uncertainty = variance.squeeze(1)

        # Surface curvature
        curvature = compute_pool_curvature(model, x_pool_norm, batch_size=2048)

        # Composite score
        u_norm = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min() + 1e-8)
        c_norm = (curvature - curvature.min()) / (curvature.max() - curvature.min() + 1e-8)
        score = u_norm + curvature_weight * c_norm

        top_indices = torch.topk(score, k=query_size).indices
        x_acquired_raw = x_pool_raw[top_indices]

        # Price and invert BSIV on GPU
        prices = price_bates_call(x_acquired_raw)
        m = x_acquired_raw[:, 0]
        tau = x_acquired_raw[:, 1] / 365.0
        r_rate = x_acquired_raw[:, 3]
        q_div = x_acquired_raw[:, 4]
        bsiv_acquired = invert_bsiv(prices, m, tau, r_rate, q_div).unsqueeze(1)
        valid = torch.isfinite(bsiv_acquired.squeeze(1)) & (bsiv_acquired.squeeze(1) > 0.01) & (bsiv_acquired.squeeze(1) < 4.0)
        x_acquired_raw = x_acquired_raw[valid]
        y_acquired = bsiv_acquired[valid]

        x_train_raw = torch.cat([x_train_raw, x_acquired_raw], dim=0)
        y_train = torch.cat([y_train, y_acquired], dim=0)

        x_train_norm = normalize_features(x_train_raw, bounds)
        train_loader = create_dataloader(x_train_norm, y_train, batch_size=batch_size, shuffle=True, num_workers=0)

        for _ in range(epochs_per_round):
            run_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler, use_amp=use_amp)
        val_loss = run_epoch(model, val_loader, criterion, None, device, scaler=None, use_amp=False)
        scheduler.step(val_loss)

        best_val_loss = min(best_val_loss, val_loss)
        trial.report(val_loss, step=r)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_loss


def save_tuned_configs(
    best_params: dict[str, Any],
    base_cfg: dict[str, Any],
    out_cfg_path: str,
    out_model_cfg_path: str,
) -> None:
    # 1. Update model config
    hidden_depth = best_params["hidden_depth"]
    hidden_dim = best_params["hidden_dim"]
    model_cfg = {
        "model_name": "surrogate_mlp",
        "input_dim": base_cfg["model"]["input_dim"],
        "hidden_layers": [hidden_dim] * hidden_depth,
        "activation": best_params["activation"],
        "dropout_rate": round(best_params["dropout_rate"], 4),
        "output_dim": base_cfg["model"]["output_dim"],
    }
    Path(out_model_cfg_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_model_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(model_cfg, f, sort_keys=False)

    # 2. Update base config
    tuned_cfg = copy.deepcopy(base_cfg)
    tuned_cfg["model_config"] = out_model_cfg_path
    tuned_cfg.pop("model", None)

    tuned_cfg["training"]["learning_rate"] = float(f"{best_params['learning_rate']:.2e}")
    tuned_cfg["training"]["weight_decay"] = float(f"{best_params['weight_decay']:.2e}")
    tuned_cfg["training"]["lr_factor"] = round(best_params["lr_factor"], 2)
    tuned_cfg["training"]["batch_size"] = best_params["batch_size"]

    tuned_cfg["active"]["curvature_weight"] = round(best_params["curvature_weight"], 2)
    tuned_cfg["active"]["mc_passes"] = best_params["mc_passes"]

    Path(out_cfg_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(tuned_cfg, f, sort_keys=False)

    print(f"\n[Saved] Tuned model config saved to: {out_model_cfg_path}")
    print(f"[Saved] Tuned training config saved to: {out_cfg_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Tuning for Bates-Kou BSIV Surrogate")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--tune-rounds", type=int, default=6)
    parser.add_argument("--epochs-per-round", type=int, default=15)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--study-name", type=str, default="bates_bsiv_optuna")
    parser.add_argument("--out-config", type=str, default="configs/tuned_config.yaml")
    parser.add_argument("--out-model-config", type=str, default="configs/model/mlp_tuned.yaml")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    print("=" * 70)
    print("OPTUNA HYPERPARAMETER TUNING FOR BATES-KOU BSIV SURROGATE")
    print(f"Trials: {args.n_trials} | Rounds per trial: {args.tune_rounds} | Device: {base_cfg['device']}")
    print("=" * 70)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study = optuna.create_study(study_name=args.study_name, direction="minimize", pruner=pruner)

    def objective(trial: optuna.Trial) -> float:
        return evaluate_trial(trial, base_cfg, args.tune_rounds, args.epochs_per_round)

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

    print("\n" + "=" * 70)
    print(f"OPTUNA STUDY COMPLETE: Best Trial #{study.best_trial.number}")
    print(f"Best Validation Loss: {study.best_value:.6f}")
    print("Best Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("=" * 70)

    save_tuned_configs(study.best_params, base_cfg, args.out_config, args.out_model_config)


if __name__ == "__main__":
    main()
