"""
Step 2: train a world model on the collected offline data.

    python train_wm.py --model diamond --steps 5000
    python train_wm.py --model iris    --steps 5000

The script is intentionally thin — all the work lives in wm_iris/ and wm_diamond/.
To plug in your own model, implement the WorldModel Protocol and add a branch here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data import TrajectoryDataset


def build_model(name: str, obs_dim: int, action_dim: int, device: str):
    if name == "iris":
        from wm_iris import MiniIRIS

        return MiniIRIS(obs_dim=obs_dim, action_dim=action_dim, device=device)
    if name == "diamond":
        from wm_diamond import MiniDIAMOND

        return MiniDIAMOND(obs_dim=obs_dim, action_dim=action_dim, device=device)
    raise ValueError(f"unknown model: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["iris", "diamond"], default="diamond")
    ap.add_argument("--data", type=str, default="data/raw")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", type=str, default="checkpoints/wm.pt")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dataset = TrajectoryDataset(args.data, device=args.device)
    print(f"Dataset: N={len(dataset)}  obs_dim={dataset.obs_dim}  action_dim={dataset.action_dim}")

    wm = build_model(args.model, dataset.obs_dim, dataset.action_dim, args.device)
    n_params = sum(p.numel() for p in wm.parameters())
    print(f"Model: {args.model}  params={n_params/1e6:.2f}M  device={args.device}")

    hist = wm.fit(dataset, steps=args.steps, batch_size=args.batch_size, lr=args.lr)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wm.save(str(out))
    print(f"saved → {out}")

    # Quick evaluation metrics
    mse1 = wm.eval_next_obs_mse(dataset)
    mseH = wm.eval_rollout_mse(dataset, horizon=15)
    print(f"eval:  1-step MSE = {mse1:.4f}   15-step MSE = {mseH:.4f}")


if __name__ == "__main__":
    main()
