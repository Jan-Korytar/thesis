from typing import Any
import numpy as np
import torch


PARAM_NAMES = [
    "moneyness",
    "maturity_days",
    "spot_var",
    "risk_free_rate",
    "dividend_yield",
    "kappa",
    "long_run_var",
    "sigma",
    "rho",
    "jump_intensity",
    "nu_u",
    "nu_d",
    "p_up",
]


def price_bates_call(x: torch.Tensor, n_quad: int = 128) -> torch.Tensor:
    assert x.ndim == 2
    assert x.shape[1] == 13

    device = x.device
    dtype = torch.float64
    cdtype = torch.complex128

    m_val = x[:, 0].to(dtype).unsqueeze(-1)
    tau = (x[:, 1] / 365.0).to(dtype).unsqueeze(-1)
    v0 = x[:, 2].to(dtype).unsqueeze(-1)
    r = x[:, 3].to(dtype).unsqueeze(-1)
    q = x[:, 4].to(dtype).unsqueeze(-1)
    kap = x[:, 5].to(dtype).unsqueeze(-1)
    theta = x[:, 6].to(dtype).unsqueeze(-1)
    sig = x[:, 7].to(dtype).unsqueeze(-1)
    rh = x[:, 8].to(dtype).unsqueeze(-1)
    l_val = x[:, 9].to(dtype).unsqueeze(-1)
    nu1 = x[:, 10].to(dtype).unsqueeze(-1)
    nu2 = x[:, 11].to(dtype).unsqueeze(-1)
    prob = x[:, 12].to(dtype).unsqueeze(-1)

    u_max = 80.0
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    nodes_t = torch.as_tensor(0.5 * u_max * (nodes + 1.0), dtype=dtype, device=device).unsqueeze(0)
    weights_t = torch.as_tensor(0.5 * u_max * weights, dtype=dtype, device=device).unsqueeze(0)

    k_over_s = torch.exp(-m_val)
    k_log = torch.log(k_over_s)
    k_bar = prob / (1.0 - nu1) + (1.0 - prob) / (1.0 + nu2) - 1.0

    def char_func(u_eval: torch.Tensor) -> torch.Tensor:
        gamma = kap - 1j * rh * sig * u_eval
        d = torch.sqrt(gamma**2 + (sig**2) * (u_eval**2 + 1j * u_eval))
        g = (gamma - d) / (gamma + d)
        exp_neg_d_tau = torch.exp(-d * tau)

        c_part = (r - q) * 1j * u_eval * tau + (kap * theta / (sig**2)) * (
            (gamma - d) * tau - 2.0 * torch.log((1.0 - g * exp_neg_d_tau) / (1.0 - g))
        )
        d_part = ((gamma - d) / (sig**2)) * ((1.0 - exp_neg_d_tau) / (1.0 - g * exp_neg_d_tau))
        phi_heston = torch.exp(c_part + d_part * v0)

        e_exp_iu_y = prob / (1.0 - 1j * u_eval * nu1) + (1.0 - prob) / (1.0 + 1j * u_eval * nu2)
        psi_jump = l_val * (e_exp_iu_y - 1.0 - 1j * u_eval * k_bar)
        phi_jump = torch.exp(tau * psi_jump)

        return phi_heston * phi_jump

    u_c = nodes_t.to(cdtype)
    phi_eval = char_func(u_c - 0.5j)
    integrand = torch.real(torch.exp(-1j * u_c * k_log) * phi_eval / (u_c**2 + 0.25))
    integral = torch.sum(weights_t * integrand, dim=-1)

    c_norm = torch.exp(-q.squeeze(-1) * tau.squeeze(-1)) - (torch.sqrt(k_over_s.squeeze(-1)) / np.pi) * integral

    lower_bound = torch.clamp(
        torch.exp(-q.squeeze(-1) * tau.squeeze(-1)) - k_over_s.squeeze(-1) * torch.exp(-r.squeeze(-1) * tau.squeeze(-1)),
        min=0.0,
    )
    upper_bound = torch.exp(-q.squeeze(-1) * tau.squeeze(-1))

    return torch.clamp(c_norm, min=lower_bound, max=upper_bound).float()


def sample_parameters(bounds: dict[str, list[float]], n_samples: int, seed: int | None = None) -> torch.Tensor:
    assert n_samples > 0
    generator = torch.Generator().manual_seed(seed) if seed is not None else None

    cols = []
    for name in PARAM_NAMES:
        low, high = bounds[name]
        assert low <= high
        col = torch.empty(n_samples).uniform_(low, high, generator=generator)
        cols.append(col)

    features = torch.stack(cols, dim=1)
    assert features.shape == (n_samples, 13)
    return features


def normalize_features(x: torch.Tensor, bounds: dict[str, list[float]]) -> torch.Tensor:
    assert x.ndim == 2
    assert x.shape[1] == len(PARAM_NAMES)
    lows = torch.tensor([bounds[name][0] for name in PARAM_NAMES], dtype=x.dtype, device=x.device)
    highs = torch.tensor([bounds[name][1] for name in PARAM_NAMES], dtype=x.dtype, device=x.device)
    return 2.0 * (x - lows) / (highs - lows) - 1.0


def generate_bates_data(
    bounds: dict[str, list[float]],
    n_samples: int,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = sample_parameters(bounds, n_samples, seed)
    y = price_bates_call(x).unsqueeze(1)

    assert not torch.isnan(x).any()
    assert not torch.isnan(y).any()
    assert y.shape == (n_samples, 1)
    return x, y
