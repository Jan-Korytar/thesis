from pathlib import Path
from typing import Any
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    assert config_path.is_file(), f"Config file not found: {config_path}"

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert isinstance(cfg, dict)
    assert "experiment_name" in cfg
    assert "seed" in cfg
    assert "device" in cfg
    assert "data" in cfg
    assert "training" in cfg

    if "model_config" in cfg:
        model_cfg_path = Path(cfg["model_config"])
        if not model_cfg_path.is_absolute():
            resolved = (config_path.parent / model_cfg_path).resolve()
            if not resolved.is_file():
                resolved = (config_path.parent.parent / model_cfg_path).resolve()
            model_cfg_path = resolved

        assert model_cfg_path.is_file(), f"Model config file not found: {model_cfg_path}"
        with open(model_cfg_path, "r", encoding="utf-8") as mf:
            model_cfg = yaml.safe_load(mf)
        assert isinstance(model_cfg, dict)
        cfg["model"] = model_cfg

    assert "model" in cfg
    assert "model_name" in cfg["model"]
    assert "input_dim" in cfg["model"]
    assert "hidden_layers" in cfg["model"]
    assert "activation" in cfg["model"]
    assert "output_dim" in cfg["model"]
    assert "dropout_rate" in cfg["model"]

    assert "bounds" in cfg["data"]

    assert "learning_rate" in cfg["training"]
    assert "weight_decay" in cfg["training"]
    assert "loss_fn" in cfg["training"]

    if "batch_size" not in cfg["training"] and "batch_size" in cfg["data"]:
        cfg["training"]["batch_size"] = cfg["data"]["batch_size"]
    if "eval_batch_size" not in cfg["training"]:
        cfg["training"]["eval_batch_size"] = 16384
    if "use_vega_loss" not in cfg["training"]:
        cfg["training"]["use_vega_loss"] = True
    if "use_amp" not in cfg["training"]:
        cfg["training"]["use_amp"] = True
    if "min_learning_rate" not in cfg["training"]:
        cfg["training"]["min_learning_rate"] = 1.0e-5
    if "lr_patience" not in cfg["training"]:
        cfg["training"]["lr_patience"] = 2
    if "lr_factor" not in cfg["training"]:
        cfg["training"]["lr_factor"] = 0.5

    assert "batch_size" in cfg["training"]

    if "mode" not in cfg:
        cfg["mode"] = "active"
    if "target_type" not in cfg:
        cfg["target_type"] = "price"

    mode = cfg["mode"]
    assert mode in ("active", "passive")

    if mode == "active":
        assert "active" in cfg
        active_cfg = cfg["active"]
        assert "initial_samples" in active_cfg
        assert "rounds" in active_cfg
        assert "query_size" in active_cfg
        assert "pool_size" in active_cfg
        assert "mc_passes" in active_cfg
        assert "val_samples" in active_cfg
        assert "epochs_per_round" in active_cfg

        if "target_val_loss" not in active_cfg:
            active_cfg["target_val_loss"] = 0.001
        if "early_stopping_patience" not in active_cfg:
            active_cfg["early_stopping_patience"] = 5
        if "curvature_weight" not in active_cfg:
            active_cfg["curvature_weight"] = 1.0
    else:
        if "passive" not in cfg:
            cfg["passive"] = {
                "n_samples": cfg["data"]["n_samples"],
                "train_split": cfg["data"]["train_split"],
                "epochs": cfg["training"]["epochs"],
            }
        passive_cfg = cfg["passive"]
        assert "n_samples" in passive_cfg
        assert "train_split" in passive_cfg
        assert "epochs" in passive_cfg

    return cfg
