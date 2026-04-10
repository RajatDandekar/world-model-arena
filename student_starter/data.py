"""
TrajectoryDataset — loads episodes collected by data_collect.py.

Each episode is a single .npz file with keys:
    obs:    (T+1, obs_dim)   float32
    action: (T,   action_dim) float32
    reward: (T,)              float32
    done:   (T,)              bool

The dataset concatenates all episodes and supports two sampling modes:

  sample_transitions(B)
      Returns (obs_t, action_t, reward_t, done_t, obs_tp1) minibatches
      for training the world model's single-step prediction.

  sample_context(B, ctx_len)
      Returns (B, ctx_len, obs_dim) + (B, ctx_len, action_dim) windows
      for seeding dreams during policy training.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    def __init__(self, root: str | Path, device: str | torch.device = "cpu"):
        self.root = Path(root)
        self.device = torch.device(device)
        self.files = sorted(glob.glob(str(self.root / "*.npz")))
        if not self.files:
            raise FileNotFoundError(f"no .npz episodes found in {self.root}")

        # Load everything into memory — fine for 500 episodes × ~1000 steps
        all_obs, all_act, all_rew, all_done, all_ep_id = [], [], [], [], []
        for i, f in enumerate(self.files):
            ep = np.load(f)
            obs, act, rew, done = ep["obs"], ep["action"], ep["reward"], ep["done"]
            T = act.shape[0]
            all_obs.append(obs[:-1])          # (T, obs_dim)  — obs_t
            all_act.append(act)               # (T, action_dim)
            all_rew.append(rew)               # (T,)
            all_done.append(done.astype(np.float32))
            all_ep_id.append(np.full(T, i, dtype=np.int32))

        self.obs = np.concatenate(all_obs, axis=0)        # (N, obs_dim)
        self.act = np.concatenate(all_act, axis=0)
        self.rew = np.concatenate(all_rew, axis=0)
        self.done = np.concatenate(all_done, axis=0)
        self.ep_id = np.concatenate(all_ep_id, axis=0)

        # Store next-obs side-by-side to avoid index tricks later
        next_obs = []
        offset = 0
        for f in self.files:
            ep = np.load(f)
            next_obs.append(ep["obs"][1:])  # (T, obs_dim)
            offset += ep["action"].shape[0]
        self.next_obs = np.concatenate(next_obs, axis=0)

        self.obs_dim = self.obs.shape[1]
        self.action_dim = self.act.shape[1]

    def __len__(self) -> int:
        return len(self.obs)

    def __getitem__(self, idx: int):
        return {
            "obs": torch.as_tensor(self.obs[idx], dtype=torch.float32),
            "action": torch.as_tensor(self.act[idx], dtype=torch.float32),
            "reward": torch.as_tensor(self.rew[idx], dtype=torch.float32),
            "done": torch.as_tensor(self.done[idx], dtype=torch.float32),
            "next_obs": torch.as_tensor(self.next_obs[idx], dtype=torch.float32),
        }

    # ---------------- batched samplers used by world models ----------------

    def sample_transitions(self, batch_size: int):
        idx = np.random.randint(0, len(self), size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[idx], device=self.device, dtype=torch.float32),
            "action": torch.as_tensor(self.act[idx], device=self.device, dtype=torch.float32),
            "reward": torch.as_tensor(self.rew[idx], device=self.device, dtype=torch.float32),
            "done": torch.as_tensor(self.done[idx], device=self.device, dtype=torch.float32),
            "next_obs": torch.as_tensor(
                self.next_obs[idx], device=self.device, dtype=torch.float32
            ),
        }

    def sample_context(self, batch_size: int, ctx_len: int = 4):
        """Sample (B, ctx_len, obs_dim) and (B, ctx_len, action_dim) windows.

        Only returns windows that stay within a single episode.
        """
        obs_list, act_list = [], []
        max_tries = batch_size * 20
        tries = 0
        while len(obs_list) < batch_size and tries < max_tries:
            tries += 1
            i = np.random.randint(0, len(self) - ctx_len)
            if self.ep_id[i] != self.ep_id[i + ctx_len - 1]:
                continue
            obs_list.append(self.obs[i : i + ctx_len])
            act_list.append(self.act[i : i + ctx_len])
        if len(obs_list) < batch_size:
            raise RuntimeError("could not sample enough intra-episode windows")
        obs = torch.as_tensor(
            np.stack(obs_list), device=self.device, dtype=torch.float32
        )
        act = torch.as_tensor(
            np.stack(act_list), device=self.device, dtype=torch.float32
        )
        return obs, act

    def sample_rollout(self, batch_size: int, horizon: int):
        """Sample consecutive (obs_0..H, action_0..H-1, reward_0..H-1) windows."""
        obs_list, act_list, rew_list = [], [], []
        max_tries = batch_size * 20
        tries = 0
        while len(obs_list) < batch_size and tries < max_tries:
            tries += 1
            i = np.random.randint(0, len(self) - horizon - 1)
            if self.ep_id[i] != self.ep_id[i + horizon]:
                continue
            obs_list.append(self.obs[i : i + horizon + 1])
            act_list.append(self.act[i : i + horizon])
            rew_list.append(self.rew[i : i + horizon])
        if len(obs_list) < batch_size:
            raise RuntimeError("could not sample enough rollout windows")
        return (
            torch.as_tensor(np.stack(obs_list), device=self.device, dtype=torch.float32),
            torch.as_tensor(np.stack(act_list), device=self.device, dtype=torch.float32),
            torch.as_tensor(np.stack(rew_list), device=self.device, dtype=torch.float32),
        )
