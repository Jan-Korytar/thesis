import argparse
from pathlib import Path
import torch
from data.generator import generate_bates_data
from utils.config_parser import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--n-samples", type=int, default=50000)
    parser.add_argument("--output", type=str, default="data/bates_kou_dataset.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config_path if hasattr(args, "config_path") else args.config)
    bounds = cfg["data"]["bounds"]

    print(f"Generating {args.n_samples} Bates-Kou samples...")
    x, y = generate_bates_data(bounds, args.n_samples, seed=args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"x": x, "y": y, "bounds": bounds}, output_path)
    print(f"Saved dataset to {output_path} (x: {x.shape}, y: {y.shape})")


if __name__ == "__main__":
    main()
