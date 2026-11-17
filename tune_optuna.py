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


class PassiveBenchmark:
    """Stage 1 Benchmark: Fixed difficulty-weighted training + validation set in VRAM for pure architecture/scheduler search."""

    def __init__(self, cfg: dict[str, Any], device: torch.device) -> None:
        opt_cfg = cfg["optuna"]
        bounds = cfg["data"]["bounds"]
        t0 = time.time()
        x_tr, self.y_train = generate_bates_bsiv_data(
            bounds, opt_cfg["stage1_train_samples"], seed=cfg["seed"] + 10, device=device
        )
        x_va, self.y_val = generate_bates_bsiv_data(
            get_val_bounds(bounds), opt_cfg["val_samples"], seed=cfg["seed"] + 11, device=device
        )
        self.x_train = normalize_features(x_tr, bounds)
        self.x_val = normalize_features(x_va, bounds)
        print(
            f"[Stage 1 Benchmark] train={len(self.x_train)} | val={len(self.x_val)} cached on {device} in {time.time() - t0:.1f}s"
        )


class ActiveBudgetBenchmark:
    """Stage 2 Benchmark: Pre-generates 50k initial set, 100k validation set, and 8x40k candidate pools in VRAM.
    Pre-pricing candidate pools once means trials pay zero FFT/BSIV solver cost during active acquisition.
    """

    def __init__(self, cfg: dict[str, Any], device: torch.device) -> None:
        opt_cfg = cfg["optuna"]
        bounds = cfg["data"]["bounds"]
        self.bounds = bounds
        self.rounds = opt_cfg["stage2_rounds"]
        t0 = time.time()

        self.x_init_master, self.y_init_master = generate_bates_bsiv_data(
            bounds, opt_cfg["stage2_initial_samples"], seed=cfg["seed"] + 20, device=device
        )
        x_va_raw, self.y_val = generate_bates_bsiv_data(
            get_val_bounds(bounds), opt_cfg["val_samples"], seed=cfg["seed"] + 11, device=device
        )
        self.x_val_norm = normalize_features(x_va_raw, bounds)

        self.pools_raw: list[torch.Tensor] = []
        self.pools_norm: list[torch.Tensor] = []
        self.pools_y: list[torch.Tensor] = []
        for r in range(1, self.rounds + 1):
            xp_raw, yp = generate_bates_bsiv_data(
                bounds, opt_cfg["stage2_pool_size"], seed=cfg["seed"] + 100 * r, device=device
            )
            self.pools_raw.append(xp_raw)
            self.pools_norm.append(normalize_features(xp_raw, bounds))
            self.pools_y.append(yp)

        print(
            f"[Stage 2 Benchmark] init={len(self.x_init_master)} | val={len(self.x_val_norm)} | "
            f"pools={self.rounds}x{opt_cfg['stage2_pool_size']} cached on {device} in {time.time() - t0:.1f}s"
        )


def suggest_stage1(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_layers": trial.suggest_int("n_layers", 4, 6),
        "width": trial.suggest_categorical("width", [256, 384, 512, 768, 1024]),
        "activation": trial.suggest_categorical("activation", ["silu", "gelu", "selu", "tanh"]),
        "use_residual": trial.suggest_categorical("use_residual", [True, False]),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.01, 0.04, step=0.01),
        "lr_scheduler": trial.suggest_categorical("lr_scheduler", ["cosine", "step"]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 6e-4, log=True),
        "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 1e-7, 1e-6, 1e-5]),
        "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
    }


def objective_stage1(trial: optuna.Trial, cfg: dict[str, Any], bench: PassiveBenchmark, device: torch.device) -> float:
    p = suggest_stage1(trial)
    opt_cfg = cfg["optuna"]
    epochs = opt_cfg["stage1_epochs"]
    report_every = opt_cfg["stage1_report_every"]

    trial_cfg = copy.deepcopy(cfg)
    trial_cfg["model"]["hidden_layers"] = [p["width"]] * p["n_layers"]
    trial_cfg["model"]["activation"] = p["activation"]
    trial_cfg["model"]["use_residual"] = p["use_residual"]
    trial_cfg["model"]["dropout_rate"] = p["dropout_rate"]
    trial_cfg["training"]["lr_scheduler"] = p["lr_scheduler"]
    trial_cfg["training"]["learning_rate"] = p["learning_rate"]
    trial_cfg["training"]["weight_decay"] = p["weight_decay"]
    trial_cfg["training"]["batch_size"] = p["batch_size"]

    set_seed(cfg["seed"])
    model = build_model(trial_cfg).to(device)
    optimizer = build_optimizer(model, p["learning_rate"], p["weight_decay"], device)
    scheduler = build_scheduler(optimizer, trial_cfg, total_epochs=epochs)

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
        if epoch % report_every != 0 and epoch != epochs:
            continue

        val_loss = run_epoch(model, val_loader, criterion, None, device)
        if not math.isfinite(val_loss):
            raise optuna.TrialPruned()
        best_val = min(best_val, val_loss)
        trial.report(val_loss, step=epoch)
        if trial.should_prune():
            trial.set_user_attr("seconds", time.time() - t0)
            raise optuna.TrialPruned()

    trial.set_user_attr("seconds", time.time() - t0)
    print(
        f"[Stage 1 | {device} | Trial {trial.number:03d}] val={best_val:.6f} | "
        f"params={n_params / 1e6:.2f}M | {time.time() - t0:.1f}s | {p}"
    )
    return best_val


def suggest_stage2(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "acquisition_strategy": trial.suggest_categorical(
            "acquisition_strategy",
            ["uncertainty", "grad_norm", "vega_uncertainty", "delta_variance", "ema", "curvature"],
        ),
        "uniform_ratio": trial.suggest_float("uniform_ratio", 0.15, 0.45, step=0.05),
        "curvature_weight": trial.suggest_float("curvature_weight", 0.15, 1.50, log=True),
        "epochs_per_round": trial.suggest_categorical("epochs_per_round", [10, 15, 20]),
    }


def objective_stage2(trial: optuna.Trial, cfg: dict[str, Any], bench: ActiveBudgetBenchmark, device: torch.device) -> float:
    p = suggest_stage2(trial)
    opt_cfg = cfg["optuna"]
    rounds = opt_cfg["stage2_rounds"]
    query_size = opt_cfg["stage2_query_size"]
    epochs_per_round = p["epochs_per_round"]
    mc_passes = opt_cfg["mc_passes"]
    eval_batch_size = cfg["training"]["eval_batch_size"]
    batch_size = cfg["training"]["batch_size"]

    set_seed(cfg["seed"])
    model = build_model(cfg).to(device)
    ema_model = copy.deepcopy(model).eval() if p["acquisition_strategy"] == "ema" else None

    total_epochs = (rounds + 1) * epochs_per_round
    optimizer = build_optimizer(model, cfg["training"]["learning_rate"], cfg["training"]["weight_decay"], device)
    scheduler = build_scheduler(optimizer, cfg, total_epochs=total_epochs)

    use_amp = cfg["training"]["use_amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = BSIVLoss()

    val_loader = create_dataloader(bench.x_val_norm, bench.y_val, batch_size=eval_batch_size, shuffle=False, device=device)

    x_train_raw = bench.x_init_master.clone()
    y_train = bench.y_init_master.clone()
    x_train_norm = normalize_features(x_train_raw, bench.bounds)
    train_loader = create_dataloader(x_train_norm, y_train, batch_size=batch_size, shuffle=True, device=device)

    t0 = time.time()
    best_val = float("inf")

    # Round 0
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

    # Active Rounds 1..rounds
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
        train_loader = create_dataloader(x_train_norm, y_train, batch_size=batch_size, shuffle=True, device=device)

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
        f"[Stage 2 | {device} | Trial {trial.number:03d}] val={best_val:.6f} | "
        f"samples={len(x_train_raw)} | {time.time() - t0:.1f}s | {p}"
    )
    return best_val


def build_study(study_name: str, storage_file: str, startup_trials: int, warmup_steps: int, seed: int) -> optuna.Study:
    storage_path = Path(storage_file)
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage = JournalStorage(JournalFileBackend(str(storage_path), lock_obj=JournalFileOpenLock(str(storage_path))))
    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(multivariate=True, n_startup_trials=startup_trials, seed=seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=startup_trials,
            n_warmup_steps=warmup_steps,
        ),
    )


def count_finished_trials(study: optuna.Study) -> int:
    return sum(1 for t in study.trials if t.state in (TrialState.COMPLETE, TrialState.PRUNED))


def clear_storage_if_fresh(storage_file: str) -> None:
    p = Path(storage_file)
    for candidate in [p, Path(f"{storage_file}.lock")]:
        if candidate.is_file():
            candidate.unlink()
            print(f"[Fresh Start] Removed existing journal file: {candidate}")


def save_study_plots(study: optuna.Study, stage_tag: str, out_dir: str = "optuna/plots") -> None:
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE and t.value is not None]
    if not completed:
        print(f"[Visualize] No completed trials yet for {stage_tag}; skipping plots.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = Path(out_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # 1. Optimization History Plot (Trial Values + Running Best)
    nums = [t.number for t in completed]
    vals = [t.value * 1e4 for t in completed]  # in bps
    best_so_far = []
    cur_best = float("inf")
    for v in vals:
        cur_best = min(cur_best, v)
        best_so_far.append(cur_best)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    ax.scatter(nums, vals, alpha=0.65, label="Completed Trial (bps)")
    ax.plot(nums, best_so_far, color="crimson", linewidth=2.0, label="Best So Far (bps)")
    ax.set_title(f"Optuna Optimization History — {stage_tag}")
    ax.set_xlabel("Trial Number")
    ax.set_ylabel("Validation BSIV MAE (bps)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    hist_path = plot_dir / f"{stage_tag}_history.png"
    fig.savefig(hist_path)
    plt.close(fig)

    # 2. Hyperparameter Importance Bar Chart
    if len(completed) >= 4:
        try:
            importances = optuna.importance.get_param_importances(study)
            if importances:
                names = list(importances.keys())[::-1]
                scores = [importances[k] for k in names]
                fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
                ax.barh(names, scores, color="steelblue")
                ax.set_title(f"Hyperparameter Importance — {stage_tag}")
                ax.set_xlabel("Relative Importance")
                ax.grid(True, axis="x", alpha=0.3)
                fig.tight_layout()
                imp_path = plot_dir / f"{stage_tag}_importances.png"
                fig.savefig(imp_path)
                plt.close(fig)
        except Exception as e:
            print(f"[Visualize] Could not compute importances for {stage_tag}: {e}")

    # 3. Parameter Slice / Marginal Effect Subplots
    param_keys = list(completed[0].params.keys())
    n_p = len(param_keys)
    cols = min(3, n_p)
    rows = math.ceil(n_p / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), dpi=150)
    axes_flat = [axes] if n_p == 1 else list(axes.flat)

    for idx, pk in enumerate(param_keys):
        ax = axes_flat[idx]
        x_vals = [t.params[pk] for t in completed if pk in t.params]
        y_vals = [t.value * 1e4 for t in completed if pk in t.params]
        if isinstance(x_vals[0], (str, bool)):
            str_vals = [str(v) for v in x_vals]
            cats = sorted(set(str_vals))
            cat_to_idx = {c: i for i, c in enumerate(cats)}
            ax.scatter([cat_to_idx[c] for c in str_vals], y_vals, alpha=0.6)
            ax.set_xticks(range(len(cats)))
            ax.set_xticklabels(cats, rotation=25, ha="right")
        else:
            ax.scatter(x_vals, y_vals, alpha=0.6)
            if pk in ("learning_rate", "curvature_weight"):
                ax.set_xscale("log")
        ax.set_title(pk)
        ax.set_ylabel("Val MAE (bps)")
        ax.grid(True, alpha=0.3)

    for idx in range(n_p, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle(f"Parameter Slice Plots — {stage_tag}", y=1.01)
    fig.tight_layout()
    slice_path = plot_dir / f"{stage_tag}_slice.png"
    fig.savefig(slice_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[Visualize] Saved plots to {plot_dir}/ ({stage_tag}_history.png, {stage_tag}_importances.png, {stage_tag}_slice.png)")


def run_stage1_worker(rank: int, cfg: dict[str, Any], device_str: str) -> None:
    device = torch.device(device_str)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    opt_cfg = cfg["optuna"]
    study = build_study(
        study_name=opt_cfg["stage1_study_name"],
        storage_file=opt_cfg["stage1_storage"],
        startup_trials=opt_cfg["startup_trials"],
        warmup_steps=opt_cfg["stage1_prune_warmup_reports"] * opt_cfg["stage1_report_every"],
        seed=cfg["seed"] + rank,
    )
    if count_finished_trials(study) >= opt_cfg["stage1_trials"]:
        return
    bench = PassiveBenchmark(cfg, device)
    stop_cb = optuna.study.MaxTrialsCallback(opt_cfg["stage1_trials"], states=(TrialState.COMPLETE, TrialState.PRUNED))
    study.optimize(lambda t: objective_stage1(t, cfg, bench, device), callbacks=[stop_cb])


def run_stage2_worker(rank: int, cfg: dict[str, Any], device_str: str) -> None:
    device = torch.device(device_str)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    opt_cfg = cfg["optuna"]
    study = build_study(
        study_name=opt_cfg["stage2_study_name"],
        storage_file=opt_cfg["stage2_storage"],
        startup_trials=opt_cfg["startup_trials"],
        warmup_steps=opt_cfg["stage2_prune_warmup_rounds"],
        seed=cfg["seed"] + 100 + rank,
    )
    if count_finished_trials(study) >= opt_cfg["stage2_trials"]:
        return
    bench = ActiveBudgetBenchmark(cfg, device)
    stop_cb = optuna.study.MaxTrialsCallback(opt_cfg["stage2_trials"], states=(TrialState.COMPLETE, TrialState.PRUNED))
    study.optimize(lambda t: objective_stage2(t, cfg, bench, device), callbacks=[stop_cb])


def apply_stage1_best(study: optuna.Study, cfg: dict[str, Any], out_cfg: str, out_model_cfg: str) -> dict[str, Any]:
    p = study.best_params
    model_cfg = {
        "model_name": cfg["model"]["model_name"],
        "input_dim": cfg["model"]["input_dim"],
        "hidden_layers": [p["width"]] * p["n_layers"],
        "activation": p["activation"],
        "use_residual": bool(p["use_residual"]),
        "dropout_rate": round(p["dropout_rate"], 4),
        "output_dim": cfg["model"]["output_dim"],
    }
    Path(out_model_cfg).parent.mkdir(parents=True, exist_ok=True)
    with open(out_model_cfg, "w", encoding="utf-8") as f:
        yaml.safe_dump(model_cfg, f, sort_keys=False)

    updated = copy.deepcopy(cfg)
    updated["model"] = copy.deepcopy(model_cfg)
    updated["model_config"] = out_model_cfg
    updated["training"]["lr_scheduler"] = p["lr_scheduler"]
    updated["training"]["learning_rate"] = float(f"{p['learning_rate']:.3e}")
    updated["training"]["weight_decay"] = float(f"{p['weight_decay']:.3e}")
    updated["training"]["batch_size"] = p["batch_size"]

    to_save = copy.deepcopy(updated)
    to_save.pop("model", None)
    with open(out_cfg, "w", encoding="utf-8") as f:
        yaml.safe_dump(to_save, f, sort_keys=False)

    print(f"\n[Stage 1 Winner] Trial #{study.best_trial.number}: val={study.best_value:.6f}")
    for k, v in p.items():
        print(f"  {k}: {v}")
    print(f"Saved architecture to {out_model_cfg} and updated {out_cfg}\n")
    return updated


def apply_stage2_best(study: optuna.Study, cfg: dict[str, Any], out_cfg: str) -> None:
    p = study.best_params
    updated = copy.deepcopy(cfg)
    updated.pop("model", None)

    updated["active"]["acquisition_strategy"] = p["acquisition_strategy"]
    updated["active"]["uniform_ratio"] = round(p["uniform_ratio"], 3)
    updated["active"]["curvature_weight"] = round(p["curvature_weight"], 4)
    updated["active"]["epochs_per_round"] = p["epochs_per_round"]

    with open(out_cfg, "w", encoding="utf-8") as f:
        yaml.safe_dump(updated, f, sort_keys=False)

    full_budget = updated["active"]["initial_samples"] + updated["active"]["rounds"] * updated["active"]["query_size"]
    print(f"\n[Stage 2 Winner] Trial #{study.best_trial.number}: val={study.best_value:.6f}")
    for k, v in p.items():
        print(f"  {k}: {v}")
    print(
        f"Preserved Full Marathon Schedule ({full_budget:,} total samples): "
        f"initial_samples={updated['active']['initial_samples']:,}, "
        f"rounds={updated['active']['rounds']}, "
        f"query_size={updated['active']['query_size']:,}"
    )
    print(f"Saved final tuned config to {out_cfg}")


def run_parallel_or_single(worker_fn, cfg: dict[str, Any], args: argparse.Namespace) -> None:
    avail_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    n_gpus = args.n_gpus if args.n_gpus > 0 else avail_gpus

    if args.device is None and n_gpus > 1:
        print(f"[Optuna Multi-GPU] Launching {n_gpus} parallel workers across cuda:0..cuda:{n_gpus - 1}")
        ctx = torch.multiprocessing.get_context("spawn")
        procs = []
        for rank in range(n_gpus):
            p = ctx.Process(target=worker_fn, args=(rank, cfg, f"cuda:{rank}"))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
    else:
        device = resolve_device(args.device or cfg["device"])
        worker_fn(0, cfg, str(device))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--stage", type=str, choices=["1", "2", "both"], default="both")
    parser.add_argument("--fresh", action="store_true", help="Delete existing Optuna journal(s) for the selected stage(s) and start fresh")
    parser.add_argument("--plot-only", action="store_true", help="Generate visualization plots from existing journals without running trials")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-gpus", type=int, default=0, help="0 = auto-use all available GPUs in parallel")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--out-config", type=str, default="configs/tuned_config.yaml")
    parser.add_argument("--out-model-config", type=str, default="configs/model/mlp_tuned.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    opt_cfg = cfg["optuna"]

    if args.stage in ("1", "both"):
        if args.fresh:
            clear_storage_if_fresh(opt_cfg["stage1_storage"])
        study1 = build_study(
            study_name=opt_cfg["stage1_study_name"],
            storage_file=opt_cfg["stage1_storage"],
            startup_trials=opt_cfg["startup_trials"],
            warmup_steps=opt_cfg["stage1_prune_warmup_reports"] * opt_cfg["stage1_report_every"],
            seed=cfg["seed"],
        )
        done1 = count_finished_trials(study1)
        target1 = opt_cfg["stage1_trials"]
        if not args.export_only and not args.plot_only:
            if done1 >= target1:
                print(f"=== OPTUNA STAGE 1: Already completed ({done1}/{target1} trials). Skipping to export. ===")
            else:
                print(f"=== OPTUNA STAGE 1: Resuming at {done1}/{target1} trials ({target1 - done1} remaining) ===")
                run_parallel_or_single(run_stage1_worker, cfg, args)
                study1 = build_study(
                    study_name=opt_cfg["stage1_study_name"],
                    storage_file=opt_cfg["stage1_storage"],
                    startup_trials=opt_cfg["startup_trials"],
                    warmup_steps=opt_cfg["stage1_prune_warmup_reports"] * opt_cfg["stage1_report_every"],
                    seed=cfg["seed"],
                )
        save_study_plots(study1, "stage1_arch")
        if not args.plot_only:
            cfg = apply_stage1_best(study1, cfg, args.out_config, args.out_model_config)

    if args.stage in ("2", "both"):
        if args.fresh:
            clear_storage_if_fresh(opt_cfg["stage2_storage"])
        if args.stage == "2" and Path(args.out_config).is_file():
            cfg = load_config(args.out_config)
        study2 = build_study(
            study_name=opt_cfg["stage2_study_name"],
            storage_file=opt_cfg["stage2_storage"],
            startup_trials=opt_cfg["startup_trials"],
            warmup_steps=opt_cfg["stage2_prune_warmup_rounds"],
            seed=cfg["seed"] + 100,
        )
        done2 = count_finished_trials(study2)
        target2 = opt_cfg["stage2_trials"]
        if not args.export_only and not args.plot_only:
            if done2 >= target2:
                print(f"=== OPTUNA STAGE 2: Already completed ({done2}/{target2} trials). Skipping to export. ===")
            else:
                print(f"=== OPTUNA STAGE 2: Resuming at {done2}/{target2} trials ({target2 - done2} remaining) ===")
                run_parallel_or_single(run_stage2_worker, cfg, args)
                study2 = build_study(
                    study_name=opt_cfg["stage2_study_name"],
                    storage_file=opt_cfg["stage2_storage"],
                    startup_trials=opt_cfg["startup_trials"],
                    warmup_steps=opt_cfg["stage2_prune_warmup_rounds"],
                    seed=cfg["seed"] + 100,
                )
        save_study_plots(study2, "stage2_active")
        if not args.plot_only:
            apply_stage2_best(study2, cfg, args.out_config)


if __name__ == "__main__":
    main()
