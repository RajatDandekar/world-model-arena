"""
World Model Protocol — the single interface every student's WM must satisfy.

The dream RL loop (dream_train_policy.py) is written entirely against this
interface, so swapping `mini-IRIS` ↔ `mini-DIAMOND` ↔ `your-custom-WM` is
a one-line change.

Design notes:
- obs_dim is the real MetaDrive racing obs dimension (~91).
- action_dim is 2 for [steering, throttle] in [-1, 1].
- reset_dream() seeds imagination from a short context of *real* observations.
- step_dream() does one imagined step. It must return the decoded observation
  because the policy consumes observations, not internal latents.
"""

from __future__ import annotations

from typing import Protocol, Tuple, runtime_checkable
import torch


@runtime_checkable
class WorldModel(Protocol):
    """Structural type that every world model implementation must match."""

    device: torch.device
    obs_dim: int
    action_dim: int
    context_len: int  # how many past real obs reset_dream() expects

    # ---------- persistence ----------
    def save(self, path: str) -> None:
        ...

    @classmethod
    def load(cls, path: str, device: torch.device | str = "cpu") -> "WorldModel":
        ...

    # ---------- training ----------
    def fit(
        self,
        dataset: "TrajectoryDataset",  # noqa: F821 (forward ref to data.py)
        steps: int,
        batch_size: int = 64,
        log_every: int = 100,
    ) -> dict:
        """Offline-train the WM. Returns {'train_loss': [...], ...}."""
        ...

    # ---------- imagination ----------
    def reset_dream(
        self,
        init_obs: torch.Tensor,  # (B, context_len, obs_dim)
        init_actions: torch.Tensor | None = None,  # (B, context_len, action_dim) for DIAMOND
    ) -> torch.Tensor:
        """Seed a dream. Returns initial latent state of shape (B, latent_dim)."""
        ...

    def step_dream(
        self,
        latent: torch.Tensor,  # (B, latent_dim)
        action: torch.Tensor,  # (B, action_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One imagined step.

        Returns:
            next_latent: (B, latent_dim)    — pass to next step_dream()
            reward_hat:  (B,)               — predicted scalar reward
            done_hat:    (B,)               — predicted termination prob in [0, 1]
            obs_hat:     (B, obs_dim)       — decoded observation (policy input)
        """
        ...

    # ---------- evaluation helpers ----------
    @torch.no_grad()
    def eval_next_obs_mse(self, dataset, n_batches: int = 16) -> float:
        """1-step prediction MSE on held-out data."""
        ...

    @torch.no_grad()
    def eval_rollout_mse(self, dataset, horizon: int = 15, n_batches: int = 8) -> float:
        """H-step open-loop rollout MSE."""
        ...
