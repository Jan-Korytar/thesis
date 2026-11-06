from typing import Any
import torch.nn as nn
from models.surrogate_net import SurrogateMLP, predict_mc_dropout


def build_model(cfg: dict[str, Any]) -> nn.Module:
    model_cfg = cfg["model"] if "model" in cfg else cfg
    model_name = model_cfg["model_name"]
    registry = {
        "surrogate_mlp": SurrogateMLP,
    }
    model_cls = registry[model_name]
    return model_cls(
        input_dim=model_cfg["input_dim"],
        hidden_layers=model_cfg["hidden_layers"],
        output_dim=model_cfg["output_dim"],
        activation=model_cfg["activation"],
        dropout_rate=model_cfg["dropout_rate"],
        model_name=model_name,
    )


__all__ = ["build_model", "SurrogateMLP", "predict_mc_dropout"]
