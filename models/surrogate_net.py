import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, act_cls: type[nn.Module], dropout_rate: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(dim, dim), act_cls()]
        if dropout_rate > 0.0:
            layers.append(nn.Dropout(dropout_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class SurrogateMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        output_dim: int = 1,
        activation: str = "silu",
        dropout_rate: float = 0.0,
        use_residual: bool = False,
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
            "selu": nn.SELU,
        }
        act_cls = activations[activation.lower()]

        layers: list[nn.Module] = []
        in_dim = input_dim
        for hidden_dim in hidden_layers:
            assert hidden_dim > 0
            if use_residual and in_dim == hidden_dim:
                layers.append(ResidualBlock(hidden_dim, act_cls, dropout_rate))
            else:
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(act_cls())
                if dropout_rate > 0.0:
                    layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_residual = use_residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2
        assert x.shape[1] == self.input_dim
        out = self.net(x)
        assert out.shape[1] == self.output_dim
        return out


def predict_mc_dropout(
    model: nn.Module, x: torch.Tensor, n_passes: int, batch_size: int = 16384
) -> tuple[torch.Tensor, torch.Tensor]:
    assert n_passes > 1
    assert x.ndim == 2

    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()

    means: list[torch.Tensor] = []
    vars_: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            bx = x[i : i + batch_size]
            preds = torch.stack([model(bx).float() for _ in range(n_passes)], dim=0)
            means.append(preds.mean(dim=0))
            vars_.append(preds.var(dim=0))

    model.eval()
    return torch.cat(means, dim=0), torch.cat(vars_, dim=0)

