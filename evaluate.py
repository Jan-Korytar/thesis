import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from data.generator import generate_bates_data, normalize_features, price_bates_call, sample_parameters
from models import build_model
from utils.config_parser import load_config


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


def compute_surrogate_greeks(
    model: nn.Module,
    x_raw: torch.Tensor,
    bounds: dict[str, list[float]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    x_in = x_raw.clone().to(device)

    low_m, high_m = bounds["moneyness"]
    scale_m = 2.0 / (high_m - low_m)

    x_norm = normalize_features(x_in, bounds).detach().requires_grad_(True)
    pred = model(x_norm)

    grad_outputs = torch.ones_like(pred)
    first_grads = torch.autograd.grad(
        outputs=pred,
        inputs=x_norm,
        grad_outputs=grad_outputs,
        create_graph=True,
    )[0]

    d_c_dm = first_grads[:, 0:1] * scale_m
    delta_pred = pred + d_c_dm

    second_grads = torch.autograd.grad(
        outputs=first_grads[:, 0],
        inputs=x_norm,
        grad_outputs=torch.ones_like(first_grads[:, 0]),
        create_graph=False,
    )[0]
    d2_c_dm2 = second_grads[:, 0:1] * (scale_m**2)
    gamma_pred = d_c_dm + d2_c_dm2

    return pred.detach(), delta_pred.detach(), gamma_pred.detach()


def compute_true_greeks(x_raw: torch.Tensor, h: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_base = x_raw.clone()
    x_plus = x_raw.clone()
    x_minus = x_raw.clone()

    x_plus[:, 0] += h
    x_minus[:, 0] -= h

    c_base = price_bates_call(x_base).unsqueeze(1)
    c_plus = price_bates_call(x_plus).unsqueeze(1)
    c_minus = price_bates_call(x_minus).unsqueeze(1)

    dc_dm = (c_plus - c_minus) / (2.0 * h)
    d2c_dm2 = (c_plus - 2.0 * c_base + c_minus) / (h**2)

    delta_true = c_base + dc_dm
    gamma_true = dc_dm + d2c_dm2
    return c_base, delta_true, gamma_true


def evaluate(
    checkpoint_path: str = "checkpoints/best_surrogate.pt",
    n_test_samples: int = 2000,
    seed: int = 999,
) -> dict[str, float]:
    ckpt_path = Path(checkpoint_path)
    assert ckpt_path.is_file(), f"Checkpoint not found: {ckpt_path}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    bounds = cfg["data"]["bounds"]
    test_bounds = {k: list(v) for k, v in bounds.items()}
    # Standard liquid market test range
    test_bounds["moneyness"] = [-0.5, 0.5]

    x_test = sample_parameters(test_bounds, n_test_samples, seed=seed)
    y_true, delta_true, gamma_true = compute_true_greeks(x_test)

    pred, delta_pred, gamma_pred = compute_surrogate_greeks(model, x_test, bounds, device)

    y_true = y_true.to(device)
    delta_true = delta_true.to(device)
    gamma_true = gamma_true.to(device)

    # 1. Price MAE
    price_mae = torch.mean(torch.abs(pred - y_true)).item()

    # 2. BSIV Inversion & RMSE
    m_test = x_test[:, 0].to(device)
    tau_test = (x_test[:, 1] / 365.0).to(device)
    r_test = x_test[:, 3].to(device)
    q_test = x_test[:, 4].to(device)

    bsiv_true = invert_bsiv(y_true.squeeze(-1), m_test, tau_test, r_test, q_test)
    bsiv_pred = invert_bsiv(pred.squeeze(-1), m_test, tau_test, r_test, q_test)

    valid_mask = torch.isfinite(bsiv_true) & torch.isfinite(bsiv_pred)
    bsiv_rmse = torch.sqrt(torch.mean((bsiv_pred[valid_mask] - bsiv_true[valid_mask]) ** 2)).item()

    # 3. Greeks MAE
    delta_mae = torch.mean(torch.abs(delta_pred - delta_true)).item()
    gamma_mae = torch.mean(torch.abs(gamma_pred - gamma_true)).item()

    print("\n" + "=" * 60)
    print("      DEEP BATES-KOU SURROGATE: CHEN ET AL. TEST SUITE      ")
    print("=" * 60)
    print(f"Test Samples:              {n_test_samples}")
    print(f"Moneyness Range (K/S):     [-0.50, +0.50] (0.60x to 1.65x spot)")
    print("-" * 60)
    print(f"1. Price MAE:              {price_mae:.6f} ({price_mae * 100:.3f}% of spot)")
    print(f"2. Out-of-Sample BSIV RMSE:{bsiv_rmse:.6f} ({bsiv_rmse * 100:.3f}% vol / {bsiv_rmse * 10000:.1f} bps)")
    print(f"3. Delta Sensitivity (MAE):{delta_mae:.6f}")
    print(f"4. Gamma Sensitivity (MAE):{gamma_mae:.6f}")
    print("=" * 60 + "\n")

    return {
        "price_mae": price_mae,
        "bsiv_rmse": bsiv_rmse,
        "delta_mae": delta_mae,
        "gamma_mae": gamma_mae,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Deep Surrogate on Chen et al. Test Suite")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_surrogate.pt")
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=999)
    args = parser.parse_args()

    evaluate(checkpoint_path=args.checkpoint, n_test_samples=args.n_test, seed=args.seed)


if __name__ == "__main__":
    main()
