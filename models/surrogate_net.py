import torch
import torch.nn as nn


class SurrogateMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        output_dim: int = 1,
        activation: str = "silu",
        dropout_rate: float = 0.0,
        model_name: str = "surrogate_mlp",
    ) -> None:
        super().__init__()
        assert input_dim > 0
        assert output_dim > 0
        assert len(hidden_layers) > 0
        assert dropout_rate >= 0.0

        activations = {
            "silu": nn.SiLU,
            "gelu": nn.GELU,
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
        }
        act_cls = activations[activation.lower()]

        layers: list[nn.Module] = []
        in_dim = input_dim
        for hidden_dim in hidden_layers:
            assert hidden_dim > 0
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act_cls())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2
        assert x.shape[1] == self.input_dim
        out = self.net(x)
        assert out.shape[1] == self.output_dim
        return out


def predict_mc_dropout(model: nn.Module, x: torch.Tensor, n_passes: int) -> tuple[torch.Tensor, torch.Tensor]:
    assert n_passes > 1
    assert x.ndim == 2

    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()

    with torch.no_grad():
        preds = torch.stack([model(x) for _ in range(n_passes)], dim=0)

    mean = preds.mean(dim=0)
    var = preds.var(dim=0)
    return mean, var
