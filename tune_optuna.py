import argparse
import copy
import math
from pathlib import Path
import time
from typing import Any
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import TrialState
import torch
import yaml

from data.dataset import create_dataloader
from data.generator import generate_bates_bsiv_data, normalize_features
from models import build_model
from models.losses import BSIVLoss
from train import build_optimizer, get_val_bounds, resolve_device, run_epoch, set_seed
from utils.config_parser import load_config


class Benchmark:
    def __init__(self, cfg: dict[str, Any], device: torch.device) -> None:
        opt_cfg = cfg["optuna"]
        bounds = cfg["data"]["bounds"]
        t0 = time.time()
        x_tr, self.y_train = generate_bates_bsiv_data(bounds, opt_cfg["train_samples"], seed=cfg["seed"] + 10, device=device)
        x_va, self.y_val = generate_bates_bsiv_data(
            get_val_bounds(bounds), opt_cfg["val_samples"], seed=cfg["seed"] + 11, device=device
        )
        self.x_train = normalize_features(x_tr, bounds)
        self.x_val = normalize_features(x_va, bounds)
        print(f"[Benchmark] train={len(self.x_train)} val={len(self.x_val)} on {device} in {time.time() - t0:.1f}s")


def cosine_with_warmup(epochs: int, warmup: int, floor: float = 0.01):
    def schedule(epoch: int) -> float:
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return schedule


def suggest(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_layers": trial.suggest_int("n_layers", 3, 6),
        "width": trial.suggest_categorical("width", [256, 384, 512, 768, 1024]),
        "activation": trial.suggest_categorical("activation", ["silu", "gelu", "tanh"]),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.02, 0.10, step=0.02),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [1024, 2048, 4096, 8192]),
    }


def objective(trial: optuna.Trial, cfg: dict[str, Any], bench: Benchmark, device: torch.device) -> float:
    p = suggest(trial)
    opt_cfg = cfg["optuna"]
    epochs = opt_cfg["epochs"]

    trial_cfg = copy.deepcopy(cfg)
    trial_cfg["model"]["hidden_layers"] = [p["width"]] * p["n_layers"]
    trial_cfg["model"]["activation"] = p["activation"]
    trial_cfg["model"]["dropout_rate"] = p["dropout_rate"]

    set_seed(cfg["seed"])
    model = build_model(trial_cfg).to(device)
    optimizer = build_optimizer(model, p["learning_rate"], p["weight_decay"], device)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_with_warmup(epochs, opt_cfg["warmup_epochs"]))
    use_amp = cfg["training"]["use_amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = BSIVLoss()

    train_loader = create_dataloader(bench.x_train, bench.y_train, batch_size=p["batch_size"], shuffle=True, device=device)
    val_loader = create_dataloader(
        bench.x_val, bench.y_val, batch_size=cfg["training"]["eval_batch_size"], shuffle=False, device=device
    )

    n_params = sum(t.numel() for t in model.parameters())
    trial.set_user_attr("n_params", n_params)
    t0 = time.time()
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        run_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler, use_amp=use_amp)
        scheduler.step()
        if epoch % opt_cfg["report_every"] != 0 and epoch != epochs:
            continue

        val_loss = run_epoch(model, val_loader, criterion, None, device)
        if not math.isfinite(val_loss):
            raise optuna.TrialPruned()
        best_val = min(best_val, val_loss)
        trial.report(val_loss, epoch)
        if trial.should_prune():
            trial.set_user_attr("seconds", time.time() - t0)
            raise optuna.TrialPruned()

    trial.set_user_attr("seconds", time.time() - t0)
    print(f"[Trial {trial.number:03d}] val={best_val:.6f} | params={n_params / 1e6:.2f}M | {time.time() - t0:.1f}s | {p}")
    return best_val


def export_best(study: optuna.Study, cfg: dict[str, Any], out_cfg: str, out_model_cfg: str) -> None:
    p = study.best_params
    model_cfg = {
        "model_name": cfg["model"]["model_name"],
        "input_dim": cfg["model"]["input_dim"],
        "hidden_layers": [p["width"]] * p["n_layers"],
        "activation": p["activation"],
        "dropout_rate": round(p["dropout_rate"], 4),
        "output_dim": cfg["model"]["output_dim"],
    }
    Path(out_model_cfg).parent.mkdir(parents=True, exist_ok=True)
    with open(out_model_cfg, "w", encoding="utf-8") as f:
        yaml.safe_dump(model_cfg, f, sort_keys=False)

    tuned = copy.deepcopy(cfg)
    tuned.pop("model")
    tuned["model_config"] = out_model_cfg
    tuned["training"]["learning_rate"] = float(f"{p['learning_rate']:.3e}")
    tuned["training"]["weight_decay"] = float(f"{p['weight_decay']:.3e}")
    tuned["training"]["batch_size"] = p["batch_size"]
    with open(out_cfg, "w", encoding="utf-8") as f:
        yaml.safe_dump(tuned, f, sort_keys=False)

    print(f"\nBest trial #{study.best_trial.number}: val={study.best_value:.6f}")
    for k, v in p.items():
        print(f"  {k}: {v}")
    print(f"Saved {out_model_cfg} and {out_cfg}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--out-config", type=str, default="configs/tuned_config.yaml")
    parser.add_argument("--out-model-config", type=str, default="configs/model/mlp_tuned.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    opt_cfg = cfg["optuna"]
    device = resolve_device(args.device or cfg["device"])

    storage_path = Path(opt_cfg["storage"])
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage = JournalStorage(JournalFileBackend(str(storage_path), lock_obj=JournalFileOpenLock(str(storage_path))))
    study = optuna.create_study(
        study_name=opt_cfg["study_name"],
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(multivariate=True, group=True, n_startup_trials=opt_cfg["startup_trials"]),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=opt_cfg["startup_trials"],
            n_warmup_steps=opt_cfg["prune_warmup_reports"] * opt_cfg["report_every"],
        ),
    )

    if not args.export_only:
        bench = Benchmark(cfg, device)
        stop_at_total = optuna.study.MaxTrialsCallback(opt_cfg["n_trials"], states=(TrialState.COMPLETE, TrialState.PRUNED))
        study.optimize(lambda t: objective(t, cfg, bench, device), callbacks=[stop_at_total])

    export_best(study, cfg, args.out_config, args.out_model_config)


if __name__ == "__main__":
    main()
