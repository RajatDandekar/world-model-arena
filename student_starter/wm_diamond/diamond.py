"""
mini-DIAMOND — a stripped-down DIAMOND for lidar-vector observations.

DIAMOND paper: "Diffusion for World Modeling: Visual Details Matter in Atari"
(Alonso et al., NeurIPS 2024)  https://arxiv.org/abs/2405.12399

What we keep from the paper:
    - EDM preconditioning (c_skip, c_out, c_in, c_noise) from Karras 2022
    - Training noise schedule: ln(sigma) ~ N(-0.4, 1.2²), sigma_data=0.5
    - Offset noise (sigma_offset = 0.3) for stability
    - 3-step Euler sampler at inference (paper's sweet spot)
    - Past-frame + action conditioning via AdaGN

What we change:
    - Observations are 91-D lidar vectors, not images → "U-Net" is an MLP
      ResNet with AdaGN conditioning (same structure, 1D instead of 2D)
    - L_context = 4 past obs instead of 4 past frames

Everything else is a faithful reduction of the paper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# EDM preconditioning (Karras 2022)
# ----------------------------------------------------------------------


SIGMA_DATA = 0.5
SIGMA_OFFSET = 0.3
LN_SIGMA_MEAN = -0.4
LN_SIGMA_STD = 1.2


def edm_coefs(sigma: torch.Tensor):
    """c_skip, c_out, c_in, c_noise (= 0.25 * log sigma)."""
    sigma2 = sigma.pow(2)
    c_skip = SIGMA_DATA**2 / (sigma2 + SIGMA_DATA**2)
    c_out = sigma * SIGMA_DATA / (sigma2 + SIGMA_DATA**2).sqrt()
    c_in = 1.0 / (sigma2 + SIGMA_DATA**2).sqrt()
    c_noise = 0.25 * torch.log(sigma)
    return c_skip, c_out, c_in, c_noise


def edm_loss_weight(sigma: torch.Tensor):
    return (sigma**2 + SIGMA_DATA**2) / (sigma * SIGMA_DATA).pow(2)


def sample_training_sigma(batch_size: int, device: torch.device) -> torch.Tensor:
    ln_sigma = LN_SIGMA_MEAN + LN_SIGMA_STD * torch.randn(batch_size, device=device)
    return ln_sigma.exp()


def edm_sample_schedule(n_steps: int, sigma_min: float = 0.02, sigma_max: float = 5.0, rho: float = 7.0):
    t = torch.linspace(0, 1, n_steps + 1)
    sigmas = (sigma_max ** (1 / rho) + t * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return sigmas  # length n_steps+1, starts at sigma_max, ends at sigma_min


# ----------------------------------------------------------------------
# AdaGN ResBlock (1D version for vector obs)
# ----------------------------------------------------------------------


class AdaGNResBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, dim)
        self.lin1 = nn.Linear(dim, dim)
        self.norm2 = nn.GroupNorm(groups, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.cond_proj = nn.Linear(cond_dim, 2 * dim)  # scale + shift

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        # x: (B, dim), cond: (B, cond_dim)
        h = self.norm1(x.unsqueeze(-1)).squeeze(-1)
        h = F.silu(h)
        h = self.lin1(h)
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
        h = self.norm2(h.unsqueeze(-1)).squeeze(-1)
        h = h * (1 + scale) + shift
        h = F.silu(h)
        h = self.lin2(h)
        return x + h


# ----------------------------------------------------------------------
# The denoiser MLP-UNet
# ----------------------------------------------------------------------


class DenoiserMLP(nn.Module):
    def __init__(
        self,
        obs_dim: int = 91,
        action_dim: int = 2,
        context_len: int = 4,
        hidden: int = 384,
        n_blocks: int = 6,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.context_len = context_len
        self.hidden = hidden

        # Conditioning: past obs (flattened) + past actions (flattened) + sigma
        cond_in = context_len * (obs_dim + action_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_in + hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        # Sigma (c_noise) embedding
        self.sigma_mlp = nn.Sequential(
            nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )

        self.in_proj = nn.Linear(obs_dim, hidden)
        self.blocks = nn.ModuleList(
            [AdaGNResBlock(hidden, cond_dim=hidden) for _ in range(n_blocks)]
        )
        self.out_proj = nn.Linear(hidden, obs_dim)

    def forward(
        self,
        noisy_obs: torch.Tensor,      # (B, obs_dim)
        c_noise: torch.Tensor,        # (B,)
        past_obs: torch.Tensor,       # (B, ctx, obs_dim)
        past_actions: torch.Tensor,   # (B, ctx, action_dim)
    ):
        B = noisy_obs.shape[0]
        sig = self.sigma_mlp(c_noise[:, None])                        # (B, H)
        cond_flat = torch.cat(
            [past_obs.reshape(B, -1), past_actions.reshape(B, -1)], dim=-1
        )
        cond = self.cond_mlp(torch.cat([cond_flat, sig], dim=-1))     # (B, H)

        h = self.in_proj(noisy_obs)
        for blk in self.blocks:
            h = blk(h, cond)
        return self.out_proj(h)


# ----------------------------------------------------------------------
# Reward / done head — small MLP on the denoised obs + conditioning
# ----------------------------------------------------------------------


class RewardDoneHead(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.r = nn.Linear(hidden, 1)
        self.d = nn.Linear(hidden, 1)

    def forward(self, obs_hat: torch.Tensor):
        h = self.net(obs_hat)
        return self.r(h).squeeze(-1), self.d(h).squeeze(-1)


# ----------------------------------------------------------------------
# Full mini-DIAMOND (Protocol implementation)
# ----------------------------------------------------------------------


class MiniDIAMOND(nn.Module):
    def __init__(
        self,
        obs_dim: int = 91,
        action_dim: int = 2,
        context_len: int = 4,
        hidden: int = 384,
        n_blocks: int = 6,
        n_sample_steps: int = 3,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.context_len = context_len
        self.n_sample_steps = n_sample_steps
        self.device = torch.device(device)

        self.denoiser = DenoiserMLP(
            obs_dim=obs_dim,
            action_dim=action_dim,
            context_len=context_len,
            hidden=hidden,
            n_blocks=n_blocks,
        )
        self.rdhead = RewardDoneHead(obs_dim)

        # dream buffers
        self._past_obs: torch.Tensor | None = None
        self._past_actions: torch.Tensor | None = None

        self.to(self.device)

    # ----------------------------- save / load -----------------------------
    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "obs_dim": self.obs_dim,
                    "action_dim": self.action_dim,
                    "context_len": self.context_len,
                    "hidden": self.denoiser.hidden,
                    "n_blocks": len(self.denoiser.blocks),
                    "n_sample_steps": self.n_sample_steps,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str | torch.device = "cpu") -> "MiniDIAMOND":
        ckpt = torch.load(path, map_location=device)
        m = cls(device=device, **ckpt["config"])
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        return m

    # ----------------------------- core EDM -----------------------------

    def denoise(
        self,
        x_noisy: torch.Tensor,
        sigma: torch.Tensor,
        past_obs: torch.Tensor,
        past_actions: torch.Tensor,
    ):
        """Apply EDM preconditioning around the raw denoiser."""
        c_skip, c_out, c_in, c_noise = edm_coefs(sigma)
        F_out = self.denoiser(
            c_in[:, None] * x_noisy,
            c_noise,
            past_obs,
            past_actions,
        )
        return c_skip[:, None] * x_noisy + c_out[:, None] * F_out

    # ----------------------------- training -----------------------------
    def fit(
        self,
        dataset,
        steps: int,
        batch_size: int = 64,
        log_every: int = 100,
        lr: float = 3e-4,
    ) -> dict:
        opt = torch.optim.AdamW(self.parameters(), lr=lr)
        history = {"loss": [], "loss_denoise": [], "loss_reward": [], "loss_done": []}

        H = self.context_len
        self.train()
        for step in range(steps):
            # Sample rollouts of length ctx+1 so we have a target obs_{t+1}
            obs, act, rew = dataset.sample_rollout(batch_size, horizon=H)
            obs = obs.to(self.device)    # (B, H+1, obs_dim)
            act = act.to(self.device)    # (B, H, action_dim)
            rew = rew.to(self.device)    # (B, H)

            past_obs = obs[:, :H, :]     # (B, H, obs_dim)
            past_act = act                # (B, H, action_dim)
            target = obs[:, -1, :]        # predict obs at end of window (B, obs_dim)

            B = target.shape[0]
            sigma = sample_training_sigma(B, self.device)
            # Offset noise: add a sample-level constant
            offset = SIGMA_OFFSET * torch.randn(B, 1, device=self.device)
            noise = torch.randn_like(target) + offset
            x_noisy = target + sigma[:, None] * noise

            pred = self.denoise(x_noisy, sigma, past_obs, past_act)
            weight = edm_loss_weight(sigma)[:, None]
            loss_denoise = (weight * (pred - target).pow(2)).mean()

            # Reward/done from the denoised obs
            r_hat, d_hat = self.rdhead(pred.detach())
            loss_reward = F.mse_loss(r_hat, rew[:, -1])
            loss_done = F.binary_cross_entropy_with_logits(
                d_hat, torch.zeros_like(d_hat)
            )

            loss = loss_denoise + loss_reward + 0.1 * loss_done

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            opt.step()

            history["loss"].append(loss.item())
            history["loss_denoise"].append(loss_denoise.item())
            history["loss_reward"].append(loss_reward.item())
            history["loss_done"].append(loss_done.item())

            if (step + 1) % log_every == 0:
                print(
                    f"[diamond {step+1}/{steps}] "
                    f"loss={loss.item():.3f} "
                    f"denoise={loss_denoise.item():.3f} "
                    f"rew={loss_reward.item():.4f}"
                )
        self.eval()
        return history

    # ----------------------------- sampler -----------------------------
    @torch.no_grad()
    def sample(
        self,
        past_obs: torch.Tensor,
        past_actions: torch.Tensor,
        n_steps: int | None = None,
    ) -> torch.Tensor:
        n_steps = n_steps or self.n_sample_steps
        B = past_obs.shape[0]
        sigmas = edm_sample_schedule(n_steps).to(self.device)      # (n_steps+1,)
        x = torch.randn(B, self.obs_dim, device=self.device) * sigmas[0]
        for i in range(n_steps):
            sig = sigmas[i].expand(B)
            sig_next = sigmas[i + 1].expand(B)
            denoised = self.denoise(x, sig, past_obs, past_actions)
            # Euler step (DIAMOND uses first-order deterministic sampler)
            d = (x - denoised) / sig[:, None]
            x = x + (sig_next - sig)[:, None] * d
        return x

    # ----------------------------- dream protocol -----------------------------
    @torch.no_grad()
    def reset_dream(
        self,
        init_obs: torch.Tensor,
        init_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        init_obs = init_obs.to(self.device)
        B, C, _ = init_obs.shape
        # Pad/truncate to context_len
        if C < self.context_len:
            pad = torch.zeros(
                B, self.context_len - C, self.obs_dim, device=self.device
            )
            init_obs = torch.cat([pad, init_obs], dim=1)
        else:
            init_obs = init_obs[:, -self.context_len :, :]

        if init_actions is None:
            init_actions = torch.zeros(
                B, self.context_len, self.action_dim, device=self.device
            )
        else:
            init_actions = init_actions.to(self.device)
            if init_actions.shape[1] < self.context_len:
                pad = torch.zeros(
                    B,
                    self.context_len - init_actions.shape[1],
                    self.action_dim,
                    device=self.device,
                )
                init_actions = torch.cat([pad, init_actions], dim=1)
            else:
                init_actions = init_actions[:, -self.context_len :, :]

        self._past_obs = init_obs
        self._past_actions = init_actions

        # latent is just the most recent obs (for the policy; dream loop uses obs_hat)
        return init_obs[:, -1, :].clone()

    @torch.no_grad()
    def step_dream(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = action.shape[0]
        action = action.to(self.device)

        # Slide the action window: drop oldest, append new
        self._past_actions = torch.cat(
            [self._past_actions[:, 1:, :], action[:, None, :]], dim=1
        )
        # Sample next obs via diffusion
        obs_hat = self.sample(self._past_obs, self._past_actions)

        # Slide the obs window
        self._past_obs = torch.cat(
            [self._past_obs[:, 1:, :], obs_hat[:, None, :]], dim=1
        )

        r_hat, d_logits = self.rdhead(obs_hat)
        d_hat = torch.sigmoid(d_logits)

        return obs_hat, r_hat, d_hat, obs_hat  # latent == obs for this WM

    # ----------------------------- eval helpers -----------------------------
    @torch.no_grad()
    def eval_next_obs_mse(self, dataset, n_batches: int = 16) -> float:
        self.eval()
        total, n = 0.0, 0
        H = self.context_len
        for _ in range(n_batches):
            obs, act, _rew = dataset.sample_rollout(64, horizon=H)
            obs, act = obs.to(self.device), act.to(self.device)
            past_obs = obs[:, :H, :]
            past_act = act
            target = obs[:, -1, :]
            pred = self.sample(past_obs, past_act)
            total += F.mse_loss(pred, target).item()
            n += 1
        return total / n

    @torch.no_grad()
    def eval_rollout_mse(self, dataset, horizon: int = 15, n_batches: int = 8) -> float:
        self.eval()
        total, n = 0.0, 0
        H = self.context_len
        for _ in range(n_batches):
            obs, act, _rew = dataset.sample_rollout(16, horizon=H + horizon)
            obs, act = obs.to(self.device), act.to(self.device)
            self.reset_dream(obs[:, :H, :], act[:, :H, :])
            mse = 0.0
            for t in range(horizon):
                _lat, _r, _d, obs_hat = self.step_dream(
                    torch.zeros(obs.shape[0], 1, device=self.device),
                    act[:, H + t, :],
                )
                mse += F.mse_loss(obs_hat, obs[:, H + t + 1, :]).item()
            total += mse / horizon
            n += 1
        return total / n
