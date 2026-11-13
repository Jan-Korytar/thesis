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
from train import (
    build_optimizer,
    build_scheduler,
    get_val_bounds,
    resolve_device,
    run_epoch,
    select_active_samples,
    set_seed,
    update_ema_model,
)
from utils.config_parser import load_config


class ActiveBenchmark:
    """Pre-generates initial training set, validation set, and per-round candidate pools in VRAM.
    Pre-pricing candidate pools once means trials pay zero FFT/BSIV solver cost when acquiring samples.
    """

    def __init__(self, cfg: dict[str, Any], device: torch.device) -> None:
        opt_cfg = cfg["optuna"]
        bounds = cfg["data"]["bounds"]
        self.bounds = bounds
        t0 = time.time()

        self.x_init_raw, self.y_init = generate_bates_bsiv_data(
            bounds, opt_cfg["initial_samples"], seed=cfg["seed"] + 10, device=device
        )
        x_va_raw, self.y_val = generate_bates_bsiv_data(
            get_val_bounds(bounds), opt_cfg["val_samples"], seed=cfg["seed"] + 11, device=device
        )
        self.x_val_norm = normalize_features(x_va_raw, bounds)

        self.pools_raw: list[torch.Tensor] = []
        self.pools_norm: list[torch.Tensor] = []
        self.pools_y: list[torch.Tensor] = []
        for r in range(1, opt_cfg["rounds"] + 1):
            xp_raw, yp = generate_bates_bsiv_data(
                bounds, opt_cfg["pool_size"], seed=cfg["seed"] + 100 * r, device=device
            )
            self.pools_raw.append(xp_raw)
            self.pools_norm.append(normalize_features(xp_raw, bounds))
            self.pools_y.append(yp)

        total_pool = sum(len(p) for p in self.pools_raw)
        print(
            f"[ActiveBenchmark] init={len(self.x_init_raw)} | val={len(self.x_val_norm)} | "
            f"pools={opt_cfg['rounds']}x (~{total_pool} total) cached on {device} in {time.time() - t0:.1f}s"
        )


def suggest(trial: optuna.Trial) -> dict[str, Any]:
    return {
        # Architecture
        "n_layers": trial.suggest_int("n_layers", 4, 6),
        "width": trial.suggest_categorical("width", [256, 384, 512, 768, 1024]),
        "activation": trial.suggest_categorical("activation", ["silu", "gelu", "selu", "tanh"]),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.01, 0.05, step=0.01),
        # Optimizer & Scheduler
        "lr_scheduler": trial.suggest_categorical("lr_scheduler", ["cosine", "step", "warm_restarts"]),
        "learning_rate": trial.suggest_float("learning_rate", 5e-5, 2e-3, log=True),
        "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 1e-7, 1e-6, 1e-5]),
        "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024, 2048]),
        # Active Learning
        "acquisition_strategy": trial.suggest_categorical(
            "acquisition_strategy",
            ["uncertainty", "grad_norm", "vega_uncertainty", "delta_variance", "ema", "curvature"],
        ),
        "uniform_ratio": trial.suggest_float("uniform_ratio", 0.10, 0.40, step=0.05),
        "curvature_weight": trial.suggest_float("curvature_weight", 0.25, 2.0, log=True),
        "epochs_per_round": trial.suggest_categorical("epochs_per_round", [10, 15, 20, 25]),
    }


def objective(trial: optuna.Trial, cfg: dict[str, Any], bench: ActiveBenchmark, device: torch.device) -> float:
    p = suggest(trial)
    opt_cfg = cfg["optuna"]
    rounds = opt_cfg["rounds"]
    epochs_per_round = p["epochs_per_round"]
    query_size = opt_cfg["query_size"]
    mc_passes = opt_cfg["mc_passes"]
    eval_batch_size = cfg["training"]["eval_batch_size"]

    trial_cfg = copy.deepcopy(cfg)
    trial_cfg["model"]["hidden_layers"] = [p["width"]] * p["n_layers"]
    trial_cfg["model"]["activation"] = p["activation"]
    trial_cfg["model"]["dropout_rate"] = p["dropout_rate"]
    trial_cfg["training"]["lr_scheduler"] = p["lr_scheduler"]
    trial_cfg["training"]["learning_rate"] = p["learning_rate"]
    trial_cfg["training"]["weight_decay"] = p["weight_decay"]
    trial_cfg["training"]["batch_size"] = p["batch_size"]

    set_seed(cfg["seed"])
    model = build_model(trial_cfg).to(device)
    ema_model = copy.deepcopy(model).eval() if p["acquisition_strategy"] == "ema" else None

    total_epochs = (rounds + 1) * epochs_per_round
    optimizer = build_optimizer(model, p["learning_rate"], p["weight_decay"], device)
    scheduler = build_scheduler(optimizer, trial_cfg, total_epochs=total_epochs)

    use_amp = cfg["training"]["use_amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = BSIVLoss()

    val_loader = create_dataloader(bench.x_val_norm, bench.y_val, batch_size=eval_batch_size, shuffle=False, device=device)

    x_train_raw = bench.x_init_raw.clone()
    y_train = bench.y_init.clone()
    x_train_norm = normalize_features(x_train_raw, bench.bounds)
    train_loader = create_dataloader(x_train_norm, y_train, batch_size=p["batch_size"], shuffle=True, device=device)

    n_params = sum(t.numel() for t in model.parameters())
    trial.set_user_attr("n_params", n_params)
    t0 = time.time()
    best_val = float("inf")

    # Round 0: Initial training
    for _ in range(epochs_per_round):
        run_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler, use_amp=use_amp)
        if ema_model is not None:
            update_ema_model(model, ema_model, decay=0.95)
        scheduler.step()

    val_loss = run_epoch(model, val_loader, criterion, None, device)
    if not math.isfinite(val_loss):
        raise optuna.TrialPruned()
    best_val = min(best_val, val_loss)
    trial.report(val_loss, step=0)

    # Active Learning Rounds 1..rounds
    for r in range(1, rounds + 1):
        xp_raw = bench.pools_raw[r - 1]
        xp_norm = bench.pools_norm[r - 1]
        yp = bench.pools_y[r - 1]

        eff_query = min(query_size, len(xp_raw))
        top_indices, _, _ = select_active_samples(
            strategy=p["acquisition_strategy"],
            model=model,
            ema_model=ema_model,
            x_pool_raw=xp_raw,
            x_pool_norm=xp_norm,
            query_size=eff_query,
            mc_passes=mc_passes,
            curvature_weight=p["curvature_weight"],
            eval_batch_size=eval_batch_size,
            uniform_ratio=p["uniform_ratio"],
        )

        x_train_raw = torch.cat([x_train_raw, xp_raw[top_indices]], dim=0)
        y_train = torch.cat([y_train, yp[top_indices]], dim=0)
        x_train_norm = normalize_features(x_train_raw, bench.bounds)
        train_loader = create_dataloader(x_train_norm, y_train, batch_size=p["batch_size"], shuffle=True, device=device)

        for _ in range(epochs_per_round):
            run_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler, use_amp=use_amp)
            if ema_model is not None:
                update_ema_model(model, ema_model, decay=0.95)
            scheduler.step()

        val_loss = run_epoch(model, val_loader, criterion, None, device)
        if not math.isfinite(val_loss):
            raise optuna.TrialPruned()
        best_val = min(best_val, val_loss)
        trial.report(val_loss, step=r)
        if trial.should_prune():
            trial.set_user_attr("seconds", time.time() - t0)
            raise optuna.TrialPruned()

    trial.set_user_attr("seconds", time.time() - t0)
    print(
        f"[{device} | Trial {trial.number:03d}] val={best_val:.6f} | params={n_params / 1e6:.2f}M | "
        f"{time.time() - t0:.1f}s | strat={p['acquisition_strategy']} | sched={p['lr_scheduler']} | {p}"
    )
    return best_val


def build_study(opt_cfg: dict[str, Any], seed: int) -> optuna.Study:
    storage_path = Path(opt_cfg["storage"])
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage = JournalStorage(JournalFileBackend(str(storage_path), lock_obj=JournalFileOpenLock(str(storage_path))))
    return optuna.create_study(
        study_name=opt_cfg["study_name"],
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            multivariate=True, group=True, n_startup_trials=opt_cfg["startup_trials"], seed=seed
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=opt_cfg["startup_trials"],
            n_warmup_steps=opt_cfg["prune_warmup_rounds"],
        ),
    )


def run_worker(rank: int, cfg: dict[str, Any], device_str: str) -> None:
    device = torch.device(device_str)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    opt_cfg = cfg["optuna"]
    study = build_study(opt_cfg, seed=cfg["seed"] + rank)
    bench = ActiveBenchmark(cfg, device)
    stop_at_total = optuna.study.MaxTrialsCallback(opt_cfg["n_trials"], states=(TrialState.COMPLETE, TrialState.PRUNED))
    study.optimize(lambda t: objective(t, cfg, bench, device), callbacks=[stop_at_total])


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
    tuned["training"]["lr_scheduler"] = p["lr_scheduler"]
    tuned["training"]["learning_rate"] = float(f"{p['learning_rate']:.3e}")
    tuned["training"]["weight_decay"] = float(f"{p['weight_decay']:.3e}")
    tuned["training"]["batch_size"] = p["batch_size"]
    tuned["active"]["acquisition_strategy"] = p["acquisition_strategy"]
    tuned["active"]["uniform_ratio"] = round(p["uniform_ratio"], 3)
    tuned["active"]["curvature_weight"] = round(p["curvature_weight"], 4)
    tuned["active"]["epochs_per_round"] = p["epochs_per_round"]

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
    parser.add_argument("--n-gpus", type=int, default=0, help="0 = auto-use all available GPUs in parallel")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--out-config", type=str, default="configs/tuned_config.yaml")
    parser.add_argument("--out-model-config", type=str, default="configs/model/mlp_tuned.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    opt_cfg = cfg["optuna"]

    if not args.export_only:
        avail_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        n_gpus = args.n_gpus if args.n_gpus > 0 else avail_gpus

        if args.device is None and n_gpus > 1:
            print(f"[Optuna Multi-GPU] Launching {n_gpus} parallel trial workers across cuda:0..cuda:{n_gpus - 1}")
            ctx = torch.multiprocessing.get_context("spawn")
            procs = []
            for rank in range(n_gpus):
                p = ctx.Process(target=run_worker, args=(rank, cfg, f"cuda:{rank}"))
                p.start()
                procs.append(p)
            for p in procs:
                p.join()
        else:
            device = resolve_device(args.device or cfg["device"])
            run_worker(0, cfg, str(device))

    study = build_study(opt_cfg, seed=cfg["seed"])
    export_best(study, cfg, args.out_config, args.out_model_config)


if __name__ == "__main__":
    main()
