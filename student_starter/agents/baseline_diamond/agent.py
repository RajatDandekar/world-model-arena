"""
Instructor DIAMOND baseline — dream-trained policy using DIAMOND WM.
Trained with 500 episodes data, 10K WM steps, 200 policy iterations.
"""

import os
import numpy as np
import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, obs):
        h = self.trunk(obs)
        mean = torch.tanh(self.mean_head(h))
        std = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        value = self.value_head(h).squeeze(-1)
        return dist, value

    @classmethod
    def load(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        m = cls(ckpt["obs_dim"], ckpt["action_dim"]).to(device)
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        return m


class Policy:
    CREATOR_NAME = "Instructor (DIAMOND)"
    CREATOR_UID = "instructor"

    def __init__(self):
        here = os.path.dirname(os.path.abspath(__file__))
        self.model = ActorCritic.load(os.path.join(here, "model.pt"), device="cpu")
        self.model.eval()

    def reset(self):
        pass

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32)[None]
            dist, _ = self.model.forward(obs_t)
            action = dist.mean.clamp(-1, 1).squeeze(0).numpy()
        return action.astype(np.float32)
