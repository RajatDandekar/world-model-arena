"""
World model quality evaluation on held-out data.

Computes:
  - 1-step next-obs MSE
  - 15-step open-loop rollout MSE
  - Reward prediction R²
  - Done prediction F1
  - Dream score (policy return inside WM dreams)

Usage:
    from eval_wm import evaluate_wm
    metrics = evaluate_wm("checkpoints/wm.pt", "diamond", "data/held_out")
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

from data import TrajectoryDataset


@torch.no_grad()
def _compute_reward_r2(wm, dataset, n_batches: int = 16) -> float:
    """R² for reward prediction."""
    all_true, all_pred = [], []
    for _ in range(n_batches):
        batch = dataset.sample_transitions(32)
        obs, action, reward_true = batch["obs"], batch["action"], batch["reward"]

        ctx = obs.unsqueeze(1).expand(-1, wm.context_len, -1)
        act_ctx = action.unsqueeze(1).expand(-1, wm.context_len, -1)
        wm.reset_dream(ctx, act_ctx)
        _, r_hat, _, _ = wm.step_dream(obs, action)

        all_true.append(reward_true.cpu().numpy())
        all_pred.append(r_hat.cpu().numpy())

    true = np.concatenate(all_true)
    pred = np.concatenate(all_pred)
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    if ss_tot < 1e-8:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


@torch.no_grad()
def _compute_done_f1(wm, dataset, n_batches: int = 16) -> float:
    """F1 score for done prediction."""
    all_true, all_pred = [], []
    for _ in range(n_batches):
        batch = dataset.sample_transitions(32)
        obs, action, done_true = batch["obs"], batch["action"], batch["done"]

        ctx = obs.unsqueeze(1).expand(-1, wm.context_len, -1)
        act_ctx = action.unsqueeze(1).expand(-1, wm.context_len, -1)
        wm.reset_dream(ctx, act_ctx)
        _, _, d_hat, _ = wm.step_dream(obs, action)

        all_true.append((done_true > 0.5).cpu().numpy().astype(int))
        all_pred.append((d_hat > 0.5).cpu().numpy().astype(int))

    true = np.concatenate(all_true)
    pred = np.concatenate(all_pred)

    tp = np.sum((pred == 1) & (true == 1))
    fp = np.sum((pred == 1) & (true == 0))
    fn = np.sum((pred == 0) & (true == 1))

    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


@torch.no_grad()
def _dream_eval(wm, dataset, policy_path: str, n_rollouts: int = 16, horizon: int = 15) -> float | None:
    """Evaluate the student's policy inside the WM's dreams."""
    try:
        from dream_train_policy import ActorCritic
        policy = ActorCritic.load(policy_path, device="cpu")
        policy.eval()
    except Exception:
        return None

    init_obs, init_act = dataset.sample_context(n_rollouts, ctx_len=wm.context_len)
    wm.reset_dream(init_obs, init_act)
    obs_hat = init_obs[:, -1, :].clone()

    total_rewards = torch.zeros(n_rollouts)
    for _ in range(horizon):
        obs_t = obs_hat
        with torch.no_grad():
            dist, _ = policy.forward(obs_t)
            action_batch = dist.mean.clamp(-1, 1)
        _, r_hat, _, obs_hat = wm.step_dream(obs_hat, action_batch)
        total_rewards += r_hat.cpu()

    return float(total_rewards.mean().item())


def evaluate_wm(
    wm_path: str,
    wm_type: str,
    held_out_dir: str = "data/held_out",
    policy_path: str | None = None,
) -> dict:
    """Run all WM quality metrics.

    Args:
        wm_path: Path to the world model checkpoint (wm.pt).
        wm_type: "iris", "diamond", or "custom".
        held_out_dir: Directory containing held-out .npz episodes.
        policy_path: Path to policy.pt for dream evaluation. Optional.

    Returns:
        Dict with keys: wm_mse_1step, wm_mse_rollout, wm_reward_r2,
        wm_done_f1, dream_score, error.
    """
    results = {"error": None}

    wm_path = Path(wm_path)
    if not wm_path.exists():
        results["error"] = f"WM checkpoint not found: {wm_path}"
        return results

    try:
        dataset = TrajectoryDataset(held_out_dir, device="cpu")

        # Load the appropriate world model
        if wm_type == "iris":
            from wm_iris import MiniIRIS
            wm = MiniIRIS.load(str(wm_path), device="cpu")
        elif wm_type == "diamond":
            from wm_diamond import MiniDIAMOND
            wm = MiniDIAMOND.load(str(wm_path), device="cpu")
        else:
            results["error"] = f"Unknown wm_type: {wm_type}"
            return results

        # 1-step MSE
        results["wm_mse_1step"] = float(wm.eval_next_obs_mse(dataset))

        # 15-step rollout MSE
        results["wm_mse_rollout"] = float(wm.eval_rollout_mse(dataset, horizon=15))

        # Reward R²
        results["wm_reward_r2"] = float(_compute_reward_r2(wm, dataset))

        # Done F1
        results["wm_done_f1"] = float(_compute_done_f1(wm, dataset))

        # Dream evaluation
        if policy_path and Path(policy_path).exists():
            dream_score = _dream_eval(wm, dataset, policy_path)
            results["dream_score"] = float(dream_score) if dream_score is not None else None
        else:
            results["dream_score"] = None

    except Exception as e:
        import traceback
        results["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    return results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Evaluate world model quality on held-out data")
    ap.add_argument("--wm", type=str, default="checkpoints/wm.pt")
    ap.add_argument("--wm-type", choices=["iris", "diamond", "custom"], default="diamond")
    ap.add_argument("--held-out", type=str, default="data/held_out")
    ap.add_argument("--policy", type=str, default="checkpoints/policy.pt")
    args = ap.parse_args()

    metrics = evaluate_wm(args.wm, args.wm_type, args.held_out, args.policy)

    if metrics["error"]:
        print(f"ERROR: {metrics['error']}")
    else:
        print("\nWorld Model Quality Metrics:")
        print(f"  1-step MSE:       {metrics['wm_mse_1step']:.6f}")
        print(f"  15-step MSE:      {metrics['wm_mse_rollout']:.6f}")
        print(f"  Reward R²:        {metrics['wm_reward_r2']:.4f}")
        print(f"  Done F1:          {metrics['wm_done_f1']:.4f}")
        if metrics["dream_score"] is not None:
            print(f"  Dream score:      {metrics['dream_score']:.2f}")
