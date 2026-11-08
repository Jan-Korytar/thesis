import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from data.generator import (
    bs_call_price,
    generate_bates_bsiv_data,
    generate_bates_data,
    invert_bsiv,
    normalize_features,
    price_bates_call,
    sample_parameters,
)
from models import build_model
from utils.config_parser import load_config


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
    n_test_samples: int = 10000,
    seed: int = 999,
) -> dict[str, float]:
    ckpt_path = Path(checkpoint_path)
    assert ckpt_path.is_file(), f"Checkpoint not found: {ckpt_path}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    is_bsiv = cfg.get("target_type") == "bsiv"

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    bounds = cfg["data"]["bounds"]
    test_bounds = {k: list(v) for k, v in bounds.items()}
    test_bounds["moneyness"] = [-0.5, 0.5]

    if is_bsiv:
        x_test_cuda, y_true_bsiv = generate_bates_bsiv_data(test_bounds, n_test_samples, seed=seed, device=device)
        c_true, delta_true, gamma_true = compute_true_greeks(x_test_cuda)
    else:
        x_test_cuda = sample_parameters(test_bounds, n_test_samples, seed=seed, device=device)
        c_true, delta_true, gamma_true = compute_true_greeks(x_test_cuda)
        y_true_bsiv = None

    low_m, high_m = bounds["moneyness"]
    scale_m = 2.0 / (high_m - low_m)

    x_norm = normalize_features(x_test_cuda, bounds).detach().requires_grad_(True)
    out = model(x_norm)

    m = x_test_cuda[:, 0]
    tau = x_test_cuda[:, 1] / 365.0
    r = x_test_cuda[:, 3]
    q = x_test_cuda[:, 4]

    if is_bsiv:
        bsiv_pred = out.squeeze(-1)
        bsiv_true = y_true_bsiv.to(device).squeeze(-1) if y_true_bsiv is not None else invert_bsiv(c_true.squeeze(-1), m, tau, r, q)
        price_pred = bs_call_price(m, tau, r, q, bsiv_pred).unsqueeze(1)
        price_mae = torch.mean(torch.abs(price_pred - c_true)).item()
        bsiv_rmse = torch.sqrt(torch.mean((bsiv_pred - bsiv_true) ** 2)).item()

        # Compute Greeks from BSIV surrogate
        grad_out = torch.autograd.grad(out.sum(), x_norm, create_graph=True)[0]
        d_sig_dm = grad_out[:, 0] * scale_m

        k = torch.exp(-m)
        sqrt_tau = torch.sqrt(tau)
        d1 = (m + (r - q + 0.5 * bsiv_pred**2) * tau) / (bsiv_pred * sqrt_tau)
        c1 = 0.5 * (1.0 + torch.erf(d1 / np.sqrt(2.0)))
        delta_bs = torch.exp(-q * tau) * c1
        vega_bs = torch.exp(-q * tau) * sqrt_tau * (1.0 / np.sqrt(2.0 * np.pi)) * torch.exp(-0.5 * d1**2)
        delta_pred = (delta_bs + vega_bs * d_sig_dm).unsqueeze(1)
        delta_mae = torch.mean(torch.abs(delta_pred - delta_true)).item()
        gamma_mae = 0.0  # Optional higher-order Greek
    else:
        price_pred = out
        price_mae = torch.mean(torch.abs(price_pred - c_true)).item()

        bsiv_true = invert_bsiv(c_true.squeeze(-1), m, tau, r, q)
        bsiv_pred = invert_bsiv(price_pred.squeeze(-1), m, tau, r, q)
        valid = torch.isfinite(bsiv_true) & torch.isfinite(bsiv_pred)
        bsiv_rmse = torch.sqrt(torch.mean((bsiv_pred[valid] - bsiv_true[valid]) ** 2)).item()

        grad_out = torch.autograd.grad(out.sum(), x_norm, create_graph=True)[0]
        d_c_dm = grad_out[:, 0:1] * scale_m
        delta_pred = out + d_c_dm
        delta_mae = torch.mean(torch.abs(delta_pred - delta_true)).item()

        second_grads = torch.autograd.grad(grad_out[:, 0].sum(), x_norm, create_graph=False)[0]
        d2_c_dm2 = second_grads[:, 0:1] * (scale_m**2)
        gamma_pred = d_c_dm + d2_c_dm2
        gamma_mae = torch.mean(torch.abs(gamma_pred - gamma_true)).item()

    print("\n" + "=" * 60)
    print("      DEEP BATES-KOU SURROGATE: CHEN ET AL. TEST SUITE      ")
    print("=" * 60)
    print(f"Target Type:               {'BSIV (Implied Volatility)' if is_bsiv else 'Option Price (C/S)'}")
    print(f"Test Samples:              {n_test_samples}")
    print(f"Moneyness Range (K/S):     [-0.50, +0.50] (0.60x to 1.65x spot)")
    print("-" * 60)
    print(f"1. Price MAE:              {price_mae:.6f} ({price_mae * 100:.3f}% of spot)")
    print(f"2. Out-of-Sample BSIV RMSE:{bsiv_rmse:.6f} ({bsiv_rmse * 100:.3f}% vol / {bsiv_rmse * 10000:.1f} bps)")
    print(f"3. Delta Sensitivity (MAE):{delta_mae:.6f}")
    if not is_bsiv:
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
    parser.add_argument("--n-test", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=999)
    args = parser.parse_args()

    evaluate(checkpoint_path=args.checkpoint, n_test_samples=args.n_test, seed=args.seed)


if __name__ == "__main__":
    main()
