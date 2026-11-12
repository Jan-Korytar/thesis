import argparse
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn as nn
from data.generator import (
    bs_call_price,
    invert_bsiv,
    normalize_features,
    price_bates_call,
    sample_parameters,
)
from models import build_model


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


def compute_quantiles_safe(err_tensor: torch.Tensor, max_q_samples: int = 2_000_000) -> list[float]:
    if len(err_tensor) == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    if len(err_tensor) > max_q_samples:
        step = len(err_tensor) // max_q_samples
        sub = err_tensor[::step][:max_q_samples].float()
    else:
        sub = err_tensor.float()
    q_levels = torch.tensor([0.05, 0.25, 0.50, 0.75, 0.95], dtype=torch.float32, device=sub.device)
    return torch.quantile(sub, q_levels).tolist()


def evaluate_batch_chunk(
    model: nn.Module,
    bx_test: torch.Tensor,
    bounds: dict[str, list[float]],
    is_bsiv: bool,
    compute_greeks: bool = False,
) -> tuple[float, float, float, torch.Tensor]:
    low_m, high_m = bounds["moneyness"]
    scale_m = 2.0 / (high_m - low_m)

    if compute_greeks:
        bc_true, bdelta_true, bgamma_true = compute_true_greeks(bx_test)
    else:
        bc_true = price_bates_call(bx_test).unsqueeze(1)
        bdelta_true = None
        bgamma_true = None

    bx_norm = normalize_features(bx_test, bounds).detach().requires_grad_(compute_greeks)
    bout = model(bx_norm)

    bm = bx_test[:, 0]
    btau = bx_test[:, 1] / 365.0
    br = bx_test[:, 3]
    bq = bx_test[:, 4]

    bbsiv_true = invert_bsiv(bc_true.squeeze(-1), bm, btau, br, bq)

    price_ae = 0.0
    delta_ae = 0.0
    gamma_ae = 0.0

    if is_bsiv:
        bbsiv_pred = bout.squeeze(-1)
        bprice_pred = bs_call_price(bm, btau, br, bq, bbsiv_pred).unsqueeze(1)
        price_ae = torch.sum(torch.abs(bprice_pred - bc_true)).item()

        valid_bsiv = torch.isfinite(bbsiv_true) & torch.isfinite(bbsiv_pred)
        bsiv_err = torch.abs(bbsiv_pred[valid_bsiv] - bbsiv_true[valid_bsiv]).detach()

        if compute_greeks and bdelta_true is not None:
            bgrad_out = torch.autograd.grad(bout.sum(), bx_norm, create_graph=False)[0]
            bd_sig_dm = bgrad_out[:, 0] * scale_m

            bsqrt_tau = torch.sqrt(btau)
            bd1 = (bm + (br - bq + 0.5 * bbsiv_pred**2) * btau) / (bbsiv_pred * bsqrt_tau)
            bc1 = 0.5 * (1.0 + torch.erf(bd1 / np.sqrt(2.0)))
            bdelta_bs = torch.exp(-bq * btau) * bc1
            bvega_bs = torch.exp(-bq * btau) * bsqrt_tau * (1.0 / np.sqrt(2.0 * np.pi)) * torch.exp(-0.5 * bd1**2)
            bdelta_pred = (bdelta_bs + bvega_bs * bd_sig_dm).unsqueeze(1)
            delta_ae = torch.sum(torch.abs(bdelta_pred - bdelta_true)).item()
    else:
        bprice_pred = bout
        price_ae = torch.sum(torch.abs(bprice_pred - bc_true)).item()

        bbsiv_pred = invert_bsiv(bprice_pred.squeeze(-1), bm, btau, br, bq)
        valid_bsiv = torch.isfinite(bbsiv_true) & torch.isfinite(bbsiv_pred)
        bsiv_err = torch.abs(bbsiv_pred[valid_bsiv] - bbsiv_true[valid_bsiv]).detach()

        if compute_greeks and bdelta_true is not None and bgamma_true is not None:
            bgrad_out = torch.autograd.grad(bout.sum(), bx_norm, create_graph=True)[0]
            bd_c_dm = bgrad_out[:, 0:1] * scale_m
            bdelta_pred = bout + bd_c_dm
            delta_ae = torch.sum(torch.abs(bdelta_pred - bdelta_true)).item()

            bsecond_grads = torch.autograd.grad(bgrad_out[:, 0].sum(), bx_norm, create_graph=False)[0]
            bd2_c_dm2 = bsecond_grads[:, 0:1] * (scale_m**2)
            bgamma_pred = bd_c_dm + bd2_c_dm2
            gamma_ae = torch.sum(torch.abs(bgamma_pred - bgamma_true)).item()

    return price_ae, delta_ae, gamma_ae, bsiv_err.cpu()


def evaluate_hypercube_streaming(
    model: nn.Module,
    bounds: dict[str, list[float]],
    is_bsiv: bool,
    n_samples: int,
    seed: int,
    device: torch.device,
    chunk_size: int = 100_000,
    greek_samples: int = 100_000,
) -> dict[str, Any]:
    """Streams n_samples (e.g. 10,000,000) across the full 13D hypercube in constant VRAM."""
    collected = 0
    greek_collected = 0
    cur_seed = seed

    total_price_ae = 0.0
    total_delta_ae = 0.0
    total_gamma_ae = 0.0
    total_bsiv_ae = 0.0
    total_bsiv_se = 0.0
    total_bsiv_count = 0
    bsiv_err_chunks: list[torch.Tensor] = []

    while collected < n_samples:
        needed = n_samples - collected
        batch_n = min(max(needed, 16384), chunk_size)
        x_cand = sample_parameters(bounds, batch_n, seed=cur_seed, device=device)
        cur_seed += 1000

        if is_bsiv:
            prices = price_bates_call(x_cand)
            bsiv = invert_bsiv(prices, x_cand[:, 0], x_cand[:, 1] / 365.0, x_cand[:, 3], x_cand[:, 4])
            valid = torch.isfinite(bsiv) & (bsiv > 0.01) & (bsiv < 4.0)
            x_val = x_cand[valid]
            if len(x_val) == 0:
                continue
            if len(x_val) > needed:
                x_val = x_val[:needed]
        else:
            x_val = x_cand[:needed]

        do_greeks = greek_collected < greek_samples
        p_ae, d_ae, g_ae, b_err = evaluate_batch_chunk(
            model=model,
            bx_test=x_val,
            bounds=bounds,
            is_bsiv=is_bsiv,
            compute_greeks=do_greeks,
        )

        n_cur = len(x_val)
        collected += n_cur
        total_price_ae += p_ae
        if do_greeks:
            greek_collected += n_cur
            total_delta_ae += d_ae
            total_gamma_ae += g_ae

        total_bsiv_ae += torch.sum(b_err.double()).item()
        total_bsiv_se += torch.sum(b_err.double() ** 2).item()
        total_bsiv_count += len(b_err)
        bsiv_err_chunks.append(b_err)

    all_err = torch.cat(bsiv_err_chunks, dim=0)
    quantiles = compute_quantiles_safe(all_err)

    return {
        "price_mae": total_price_ae / max(collected, 1),
        "bsiv_mae": total_bsiv_ae / max(total_bsiv_count, 1),
        "bsiv_rmse": float(np.sqrt(total_bsiv_se / max(total_bsiv_count, 1))),
        "bsiv_p05": quantiles[0],
        "bsiv_p25": quantiles[1],
        "bsiv_p50": quantiles[2],
        "bsiv_p75": quantiles[3],
        "bsiv_p95": quantiles[4],
        "delta_mae": total_delta_ae / max(greek_collected, 1),
        "gamma_mae": (total_gamma_ae / max(greek_collected, 1)) if not is_bsiv else 0.0,
    }


def get_interior_bounds(bounds: dict[str, list[float]], margin: float = 0.05) -> dict[str, list[float]]:
    """Shrinks parameter ranges to [x_low + delta, x_high - delta] where delta = margin * |x_high - x_low|."""
    interior: dict[str, list[float]] = {}
    for k, (low, high) in bounds.items():
        delta = margin * abs(high - low)
        interior[k] = [low + delta, high - delta]
    return interior


def evaluate_chen_bin_streaming(
    model: nn.Module,
    bounds: dict[str, list[float]],
    interior_bounds: dict[str, list[float]],
    tau_range: tuple[float, float],
    m_norm_range: tuple[float, float],
    is_bsiv: bool,
    n_samples: int,
    seed: int,
    device: torch.device,
    chunk_size: int = 100_000,
) -> dict[str, Any]:
    """Streams n_samples for a single Chen et al. maturity-moneyness bin in constant GPU memory."""
    bin_bounds = {k: list(v) for k, v in interior_bounds.items()}
    bin_bounds["maturity_days"] = [float(tau_range[0]), float(tau_range[1])]
    m_low, m_high = bounds["moneyness"]

    collected = 0
    cur_seed = seed
    total_bsiv_ae = 0.0
    total_bsiv_se = 0.0
    total_bsiv_count = 0
    err_chunks: list[torch.Tensor] = []

    while collected < n_samples:
        needed = n_samples - collected
        batch_n = min(max(needed * 3, 16384), chunk_size)

        x_cand = sample_parameters(bin_bounds, batch_n, seed=cur_seed, device=device)
        gen = torch.Generator(device=device).manual_seed(cur_seed + 500000)
        cur_seed += 1000

        tau_years = x_cand[:, 1] / 365.0
        v_t = x_cand[:, 2]
        u_m = torch.rand((batch_n,), generator=gen, dtype=torch.float32, device=device)
        m_tilde = m_norm_range[0] + u_m * (m_norm_range[1] - m_norm_range[0])

        m_log = m_tilde * torch.sqrt(v_t) * torch.sqrt(tau_years)
        x_cand[:, 0] = m_log

        valid_m = (m_log >= m_low) & (m_log <= m_high)
        if not valid_m.any():
            continue

        x_val = x_cand[valid_m]
        prices = price_bates_call(x_val)
        bsiv_true = invert_bsiv(prices, x_val[:, 0], x_val[:, 1] / 365.0, x_val[:, 3], x_val[:, 4])
        valid_iv = torch.isfinite(bsiv_true) & (bsiv_true > 0.01) & (bsiv_true < 4.0)
        if not valid_iv.any():
            continue

        x_val = x_val[valid_iv]
        bsiv_true = bsiv_true[valid_iv]
        if len(x_val) > needed:
            x_val = x_val[:needed]
            bsiv_true = bsiv_true[:needed]

        with torch.no_grad():
            bx_norm = normalize_features(x_val, bounds)
            bout = model(bx_norm).squeeze(-1)
            if is_bsiv:
                bsiv_pred = bout
            else:
                bsiv_pred = invert_bsiv(bout, x_val[:, 0], x_val[:, 1] / 365.0, x_val[:, 3], x_val[:, 4])

            valid_pred = torch.isfinite(bsiv_pred)
            b_err = torch.abs(bsiv_pred[valid_pred] - bsiv_true[valid_pred]).cpu()

        collected += len(x_val)
        total_bsiv_ae += torch.sum(b_err.double()).item()
        total_bsiv_se += torch.sum(b_err.double() ** 2).item()
        total_bsiv_count += len(b_err)
        err_chunks.append(b_err)

    all_err = torch.cat(err_chunks, dim=0)
    quantiles = compute_quantiles_safe(all_err)
    # Keep a subsampled error tensor for overall 12-bin quantile aggregation
    sub_err = all_err[:: max(len(all_err) // 100_000, 1)]

    return {
        "bsiv_mae": total_bsiv_ae / max(total_bsiv_count, 1),
        "bsiv_rmse": float(np.sqrt(total_bsiv_se / max(total_bsiv_count, 1))),
        "bsiv_p05": quantiles[0],
        "bsiv_p25": quantiles[1],
        "bsiv_p50": quantiles[2],
        "bsiv_p75": quantiles[3],
        "bsiv_p95": quantiles[4],
        "total_ae": total_bsiv_ae,
        "total_se": total_bsiv_se,
        "total_count": total_bsiv_count,
        "sub_errors": sub_err,
    }


def run_chen_12bin_suite(
    title: str,
    model: nn.Module,
    bounds: dict[str, list[float]],
    is_bsiv: bool,
    n_bin_samples: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    interior_bounds = get_interior_bounds(bounds, margin=0.05)

    tau_bins = [
        ("1 < tau <= 7", (1.0, 7.0)),
        ("7 < tau <= 30", (7.0, 30.0)),
        ("30 < tau <= 90", (30.0, 90.0)),
        ("90 < tau <= 365", (90.0, 365.0)),
    ]
    m_bins = [
        ("OTM (-9 <= m <= -1.8816)", (-9.0, -1.8816)),
        ("ATM (-1.8816 < m <= -0.2381)", (-1.8816, -0.2381)),
        ("ITM (-0.2381 < m <= 5)", (-0.2381, 5.0)),
    ]

    total_options = n_bin_samples * 12
    print("\n" + "=" * 90)
    print(f"  {title} ({n_bin_samples:,} / BIN = {total_options:,} TOTAL OPTIONS)")
    print("=" * 90)
    header = f"{'Maturity Bin':<18} | {'Moneyness Bin (Eq. 14)':<28} | {'MAE (bps)':>9} | {'P05':>6} | {'P25':>6} | {'Med':>6} | {'P75':>6} | {'P95':>6}"
    print(header)
    print("-" * 90)

    overall_ae = 0.0
    overall_se = 0.0
    overall_count = 0
    sub_errors_list: list[torch.Tensor] = []
    bin_results: dict[str, dict[str, float]] = {}

    bin_idx = 0
    for tau_label, tau_range in tau_bins:
        for m_label, m_range in m_bins:
            bin_idx += 1
            b_metrics = evaluate_chen_bin_streaming(
                model=model,
                bounds=bounds,
                interior_bounds=interior_bounds,
                tau_range=tau_range,
                m_norm_range=m_range,
                is_bsiv=is_bsiv,
                n_samples=n_bin_samples,
                seed=seed + 10000 * bin_idx,
                device=device,
            )
            overall_ae += b_metrics["total_ae"]
            overall_se += b_metrics["total_se"]
            overall_count += b_metrics["total_count"]
            sub_errors_list.append(b_metrics["sub_errors"])

            key = f"{tau_label} | {m_label}"
            bin_results[key] = {k: v for k, v in b_metrics.items() if k != "sub_errors"}

            print(
                f"{tau_label:<18} | {m_label:<28} | "
                f"{b_metrics['bsiv_mae']*1e4:9.1f} | "
                f"{b_metrics['bsiv_p05']*1e4:6.1f} | "
                f"{b_metrics['bsiv_p25']*1e4:6.1f} | "
                f"{b_metrics['bsiv_p50']*1e4:6.1f} | "
                f"{b_metrics['bsiv_p75']*1e4:6.1f} | "
                f"{b_metrics['bsiv_p95']*1e4:6.1f}"
            )
        print("-" * 90)

    chen_mae = overall_ae / max(overall_count, 1)
    chen_rmse = float(np.sqrt(overall_se / max(overall_count, 1)))
    chen_q = compute_quantiles_safe(torch.cat(sub_errors_list, dim=0))

    print(
        f"{'OVERALL (12 BINS)':<18} | {f'{overall_count:,} Total Options':<28} | "
        f"{chen_mae*1e4:9.1f} | "
        f"{chen_q[0]*1e4:6.1f} | "
        f"{chen_q[1]*1e4:6.1f} | "
        f"{chen_q[2]*1e4:6.1f} | "
        f"{chen_q[3]*1e4:6.1f} | "
        f"{chen_q[4]*1e4:6.1f}"
    )
    print("=" * 90 + "\n")

    return {
        "bins": bin_results,
        "overall_bsiv_mae": chen_mae,
        "overall_bsiv_rmse": chen_rmse,
    }


def evaluate(
    checkpoint_path: str = "checkpoints/best_surrogate.pt",
    n_hypercube_samples: int = 10_000_000,
    n_chen_standard_total: int = 240_000,
    n_chen_large_total: int = 24_000_000,
    seed: int = 999,
) -> dict[str, Any]:
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

    # =========================================================================
    # TEST 1: 10,000,000 SAMPLES ON FULL UNCONSTRAINED 13D HYPERCUBE
    # =========================================================================
    cube_metrics = evaluate_hypercube_streaming(
        model=model,
        bounds=bounds,
        is_bsiv=is_bsiv,
        n_samples=n_hypercube_samples,
        seed=seed,
        device=device,
    )

    print("\n" + "=" * 90)
    print("  TEST 1: FULL 13D TRAINING HYPERCUBE (100% BOUNDARY-TO-BOUNDARY SUPPORT)")
    print("=" * 90)
    print(f"Target Type:               {'BSIV (Implied Volatility)' if is_bsiv else 'Option Price (C/S)'}")
    print(f"Hypercube Samples:         {n_hypercube_samples:,}")
    print(f"Log-Moneyness Range:       [{bounds['moneyness'][0]:.2f}, {bounds['moneyness'][1]:+.2f}]")
    print(f"Maturity Range (Days):     [{bounds['maturity_days'][0]:.1f}, {bounds['maturity_days'][1]:.1f}]")
    print("-" * 90)
    print(f"1. Price MAE (C/S):        {cube_metrics['price_mae']:.6f} ({cube_metrics['price_mae'] * 100:.3f}% of spot)")
    print(f"2. BSIV MAE:               {cube_metrics['bsiv_mae']:.6f} ({cube_metrics['bsiv_mae'] * 10000:.1f} bps)")
    print(f"3. BSIV RMSE:              {cube_metrics['bsiv_rmse']:.6f} ({cube_metrics['bsiv_rmse'] * 10000:.1f} bps)")
    print(
        f"4. BSIV Error Quantiles:   P05={cube_metrics['bsiv_p05']*1e4:.1f} | "
        f"P25={cube_metrics['bsiv_p25']*1e4:.1f} | Median={cube_metrics['bsiv_p50']*1e4:.1f} | "
        f"P75={cube_metrics['bsiv_p75']*1e4:.1f} | P95={cube_metrics['bsiv_p95']*1e4:.1f} bps"
    )
    print(f"5. Delta Sensitivity MAE:  {cube_metrics['delta_mae']:.6f}")
    if not is_bsiv:
        print(f"6. Gamma Sensitivity MAE:  {cube_metrics['gamma_mae']:.6f}")
    print("=" * 90)

    # =========================================================================
    # TEST 2: CHEN ET AL. (2026) 12-BIN BENCHMARK — 240,000 TOTAL (20k / BIN)
    # =========================================================================
    chen_240k = run_chen_12bin_suite(
        title="TEST 2: CHEN ET AL. (JFE 2026) 12-BIN PAPER BENCHMARK",
        model=model,
        bounds=bounds,
        is_bsiv=is_bsiv,
        n_bin_samples=n_chen_standard_total // 12,
        seed=seed + 1_000_000,
        device=device,
    )

    # =========================================================================
    # TEST 3: CHEN ET AL. (2026) 12-BIN BENCHMARK — 24,000,000 TOTAL (2M / BIN)
    # =========================================================================
    chen_24m = run_chen_12bin_suite(
        title="TEST 3: CHEN ET AL. (JFE 2026) 12-BIN LARGE-SCALE BENCHMARK",
        model=model,
        bounds=bounds,
        is_bsiv=is_bsiv,
        n_bin_samples=n_chen_large_total // 12,
        seed=seed + 5_000_000,
        device=device,
    )

    return {
        "hypercube_10m": cube_metrics,
        "chen_240k": chen_240k,
        "chen_24m": chen_24m,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Deep Surrogate on 10M Hypercube, 240k Chen 12-Bin, and 24M Chen 12-Bin")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_surrogate.pt")
    parser.add_argument("--n-test", type=int, default=10_000_000, help="Total samples for Test 1: Full 13D Hypercube (default: 10M)")
    parser.add_argument("--n-chen", type=int, default=240_000, help="Total samples for Test 2: Chen et al. standard 12-bin suite (default: 240k = 20k/bin)")
    parser.add_argument("--n-chen-large", type=int, default=24_000_000, help="Total samples for Test 3: Chen et al. large 12-bin suite (default: 24M = 2M/bin)")
    parser.add_argument("--seed", type=int, default=999)
    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        n_hypercube_samples=args.n_test,
        n_chen_standard_total=args.n_chen,
        n_chen_large_total=args.n_chen_large,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
