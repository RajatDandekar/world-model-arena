"""
Step 3: train the driving policy entirely inside the world model's dreams.

REINFORCE with λ-returns — the same estimator DIAMOND uses for its CSGO
experiments. The policy never sees the real environment during training;
it only sees obs_hat from wm.step_dream().

Swap `--wm diamond` ↔ `--wm iris` (or point at your own checkpoint) with
no other code changes — that's the whole point of the WorldModel Protocol.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import TrajectoryDataset


# ----------------------------------------------------------------------
# Actor-critic
# ----------------------------------------------------------------------


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

    def forward(self, obs: torch.Tensor):
        h = self.trunk(obs)
        mean = torch.tanh(self.mean_head(h))
        std = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        value = self.value_head(h).squeeze(-1)
        return dist, value

    def act(self, obs: torch.Tensor):
        dist, value = self.forward(obs)
        action = dist.sample()
        logp = dist.log_prob(action).sum(-1)
        action = action.clamp(-1.0, 1.0)
        return action, logp, value

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self.state_dict(),
             "obs_dim": self.trunk[0].in_features,
             "action_dim": self.mean_head.out_features},
            path,
        )

    @classmethod
    def load(cls, path: str, device="cpu") -> "ActorCritic":
        ckpt = torch.load(path, map_location=device)
        m = cls(ckpt["obs_dim"], ckpt["action_dim"]).to(device)
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        return m


# ----------------------------------------------------------------------
# λ-returns (Sutton 1988 / Dreamer style)
# ----------------------------------------------------------------------


def lambda_returns(
    rewards: torch.Tensor,  # (H, B)
    values: torch.Tensor,   # (H+1, B)  — values at t=0..H inclusive
    gamma: float = 0.985,
    lam: float = 0.95,
) -> torch.Tensor:
    H, B = rewards.shape
    returns = torch.zeros_like(rewards)
    gae = values[-1]
    for t in reversed(range(H)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        returns[t] = gae + values[t]
    return returns


# ----------------------------------------------------------------------
# Main dream RL loop
# ----------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", type=str, default="checkpoints/wm.pt")
    ap.add_argument("--wm-type", choices=["iris", "diamond"], default="diamond")
    ap.add_argument("--data", type=str, default="data/raw")
    ap.add_argument("--iterations", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--gamma", type=float, default=0.985)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--out", type=str, default="checkpoints/policy.pt")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dataset = TrajectoryDataset(args.data, device=args.device)

    if args.wm_type == "iris":
        from wm_iris import MiniIRIS
        wm = MiniIRIS.load(args.wm, device=args.device)
    else:
        from wm_diamond import MiniDIAMOND
        wm = MiniDIAMOND.load(args.wm, device=args.device)

    policy = ActorCritic(obs_dim=dataset.obs_dim, action_dim=dataset.action_dim).to(args.device)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    for it in range(args.iterations):
        # 1. Seed dreams from real states
        init_obs, init_act = dataset.sample_context(args.batch_size, ctx_len=wm.context_len)
        _ = wm.reset_dream(init_obs, init_act)
        obs_hat = init_obs[:, -1, :].clone()   # the policy's "current obs"

        logps = []
        rewards = []
        values = []
        entropies = []
        obs_list = []

        # 2. Imagine H steps
        for t in range(args.horizon):
            dist, value = policy.forward(obs_hat)
            action = dist.rsample()
            logp = dist.log_prob(action).sum(-1)
            entropies.append(dist.entropy().sum(-1))
            action_c = action.clamp(-1.0, 1.0)

            _lat, r_hat, _d_hat, next_obs_hat = wm.step_dream(obs_hat, action_c)

            logps.append(logp)
            values.append(value)
            rewards.append(r_hat)
            obs_list.append(obs_hat)
            obs_hat = next_obs_hat

        # Bootstrap value at t=H
        with torch.no_grad():
            _, v_last = policy.forward(obs_hat)
        rewards_t = torch.stack(rewards, dim=0)   # (H, B)
        values_t = torch.stack(values + [v_last], dim=0)  # (H+1, B)

        returns = lambda_returns(rewards_t, values_t, args.gamma, args.lam)
        logp_t = torch.stack(logps, dim=0)
        ent_t = torch.stack(entropies, dim=0)

        advantages = (returns - values_t[:-1]).detach()
        policy_loss = -(logp_t * advantages).mean()
        value_loss = 0.5 * (values_t[:-1] - returns.detach()).pow(2).mean()
        entropy_bonus = ent_t.mean()

        loss = policy_loss + value_loss - args.ent_coef * entropy_bonus

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        if (it + 1) % 20 == 0:
            print(
                f"[dream {it+1}/{args.iterations}]  "
                f"loss={loss.item():.3f}  "
                f"pi={policy_loss.item():.3f}  "
                f"v={value_loss.item():.3f}  "
                f"ent={entropy_bonus.item():.3f}  "
                f"mean_ret={returns.mean().item():.3f}"
            )

    policy.save(args.out)
    print(f"saved policy → {args.out}")


if __name__ == "__main__":
    main()
