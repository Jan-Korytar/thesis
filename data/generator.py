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


def bs_call_price(m: torch.Tensor, tau: torch.Tensor, r: torch.Tensor, q: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    k = torch.exp(-m)
    sqrt_tau = torch.sqrt(tau)
    d1 = (m + (r - q + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau

    c1 = 0.5 * (1.0 + torch.erf(d1 / np.sqrt(2.0)))
    c2 = 0.5 * (1.0 + torch.erf(d2 / np.sqrt(2.0)))
    return torch.exp(-q * tau) * c1 - k * torch.exp(-r * tau) * c2


def invert_bsiv(
    price: torch.Tensor,
    m: torch.Tensor,
    tau: torch.Tensor,
    r: torch.Tensor,
    q: torch.Tensor,
    max_iter: int = 25,
    tol: float = 1e-5,
) -> torch.Tensor:
    k = torch.exp(-m)
    lower = torch.clamp(torch.exp(-q * tau) - k * torch.exp(-r * tau), min=0.0)
    upper = torch.exp(-q * tau)
    valid = (price > lower + 1e-5) & (price < upper - 1e-5)

    sigma = torch.clamp(torch.sqrt(2.0 * np.pi / tau) * price, min=0.05, max=2.0)
    for _ in range(max_iter):
        p_guess = bs_call_price(m, tau, r, q, sigma)
        diff = p_guess - price

        sqrt_tau = torch.sqrt(tau)
        d1 = (m + (r - q + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
        vega = torch.exp(-q * tau) * sqrt_tau * (1.0 / np.sqrt(2.0 * np.pi)) * torch.exp(-0.5 * d1**2)
        vega = torch.clamp(vega, min=1e-8)

        step = diff / vega
        sigma = torch.clamp(sigma - step, min=0.001, max=5.0)
        if torch.max(torch.abs(diff[valid])) < tol:
            break

    sigma[~valid] = float("nan")
    return sigma


def sample_parameters(
    bounds: dict[str, list[float]],
    n_samples: int,
    seed: int | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    assert n_samples > 0
    generator = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None

    lows = torch.tensor([bounds[name][0] for name in PARAM_NAMES], dtype=torch.float32)
    highs = torch.tensor([bounds[name][1] for name in PARAM_NAMES], dtype=torch.float32)
    u = torch.rand(n_samples, len(PARAM_NAMES), generator=generator)
    features = (lows + u * (highs - lows)).to(device)
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
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    x = sample_parameters(bounds, n_samples, seed, device=device)
    y = price_bates_call(x).unsqueeze(1)

    assert not torch.isnan(x).any()
    assert not torch.isnan(y).any()
    assert y.shape == (n_samples, 1)
    return x, y


def generate_bates_bsiv_data(
    bounds: dict[str, list[float]],
    n_samples: int,
    seed: int | None = None,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    collected_x: list[torch.Tensor] = []
    collected_y: list[torch.Tensor] = []
    collected_count = 0
    cur_seed = seed

    while collected_count < n_samples:
        needed = n_samples - collected_count
        batch_n = max(needed * 2, 1000)
        x_cand = sample_parameters(bounds, batch_n, cur_seed, device=device)
        if cur_seed is not None:
            cur_seed += 1000

        prices = price_bates_call(x_cand)
        m = x_cand[:, 0]
        tau = x_cand[:, 1] / 365.0
        r = x_cand[:, 3]
        q = x_cand[:, 4]

        bsiv = invert_bsiv(prices, m, tau, r, q)
        valid = torch.isfinite(bsiv) & (bsiv > 0.01) & (bsiv < 4.0)

        x_val = x_cand[valid]
        y_val = bsiv[valid].unsqueeze(1)

        collected_x.append(x_val)
        collected_y.append(y_val)
        collected_count += len(x_val)

    x_all = torch.cat(collected_x, dim=0)[:n_samples]
    y_all = torch.cat(collected_y, dim=0)[:n_samples]
    assert x_all.shape == (n_samples, 13)
    assert y_all.shape == (n_samples, 1)
    return x_all, y_all
