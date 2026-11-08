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

    total_price_ae = 0.0
    total_bsiv_se = 0.0
    valid_bsiv_count = 0
    total_delta_ae = 0.0
    total_gamma_ae = 0.0
    n_total = len(x_test_cuda)
    eval_batch_size = cfg.get("training", {}).get("eval_batch_size", 16384)

    for i in range(0, n_total, eval_batch_size):
        bx_test = x_test_cuda[i : i + eval_batch_size]
        bc_true = c_true[i : i + eval_batch_size]
        bdelta_true = delta_true[i : i + eval_batch_size]
        bgamma_true = gamma_true[i : i + eval_batch_size] if gamma_true is not None else None
        by_true_bsiv = y_true_bsiv[i : i + eval_batch_size] if y_true_bsiv is not None else None

        bx_norm = normalize_features(bx_test, bounds).detach().requires_grad_(True)
        bout = model(bx_norm)

        bm = bx_test[:, 0]
        btau = bx_test[:, 1] / 365.0
        br = bx_test[:, 3]
        bq = bx_test[:, 4]

        if is_bsiv:
            bbsiv_pred = bout.squeeze(-1)
            bbsiv_true = by_true_bsiv.to(device).squeeze(-1) if by_true_bsiv is not None else invert_bsiv(bc_true.squeeze(-1), bm, btau, br, bq)
            bprice_pred = bs_call_price(bm, btau, br, bq, bbsiv_pred).unsqueeze(1)
            total_price_ae += torch.sum(torch.abs(bprice_pred - bc_true)).item()

            valid_bsiv = torch.isfinite(bbsiv_true) & torch.isfinite(bbsiv_pred)
            total_bsiv_se += torch.sum((bbsiv_pred[valid_bsiv] - bbsiv_true[valid_bsiv]) ** 2).item()
            valid_bsiv_count += valid_bsiv.sum().item()

            bgrad_out = torch.autograd.grad(bout.sum(), bx_norm, create_graph=False)[0]
            bd_sig_dm = bgrad_out[:, 0] * scale_m

            bsqrt_tau = torch.sqrt(btau)
            bd1 = (bm + (br - bq + 0.5 * bbsiv_pred**2) * btau) / (bbsiv_pred * bsqrt_tau)
            bc1 = 0.5 * (1.0 + torch.erf(bd1 / np.sqrt(2.0)))
            bdelta_bs = torch.exp(-bq * btau) * bc1
            bvega_bs = torch.exp(-bq * btau) * bsqrt_tau * (1.0 / np.sqrt(2.0 * np.pi)) * torch.exp(-0.5 * bd1**2)
            bdelta_pred = (bdelta_bs + bvega_bs * bd_sig_dm).unsqueeze(1)
            total_delta_ae += torch.sum(torch.abs(bdelta_pred - bdelta_true)).item()
        else:
            bprice_pred = bout
            total_price_ae += torch.sum(torch.abs(bprice_pred - bc_true)).item()

            bbsiv_true = invert_bsiv(bc_true.squeeze(-1), bm, btau, br, bq)
            bbsiv_pred = invert_bsiv(bprice_pred.squeeze(-1), bm, btau, br, bq)
            valid = torch.isfinite(bbsiv_true) & torch.isfinite(bbsiv_pred)
            total_bsiv_se += torch.sum((bbsiv_pred[valid] - bbsiv_true[valid]) ** 2).item()
            valid_bsiv_count += valid.sum().item()

            bgrad_out = torch.autograd.grad(bout.sum(), bx_norm, create_graph=True)[0]
            bd_c_dm = bgrad_out[:, 0:1] * scale_m
            bdelta_pred = bout + bd_c_dm
            total_delta_ae += torch.sum(torch.abs(bdelta_pred - bdelta_true)).item()

            bsecond_grads = torch.autograd.grad(bgrad_out[:, 0].sum(), bx_norm, create_graph=False)[0]
            bd2_c_dm2 = bsecond_grads[:, 0:1] * (scale_m**2)
            bgamma_pred = bd_c_dm + bd2_c_dm2
            total_gamma_ae += torch.sum(torch.abs(bgamma_pred - bgamma_true)).item()

    price_mae = total_price_ae / n_total
    bsiv_rmse = float(np.sqrt(total_bsiv_se / max(valid_bsiv_count, 1)))
    delta_mae = total_delta_ae / n_total
    gamma_mae = (total_gamma_ae / n_total) if not is_bsiv else 0.0

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
