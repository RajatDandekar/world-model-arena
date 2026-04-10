"""
Step 4: deploy the trained policy on the REAL MetaDrive simulator.

This is the moment of truth. The policy has only ever seen obs_hat from its
world model. Now it has to drive a real car on a real track.

If there's a gap between dream reward and real reward, that gap IS the lesson.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from dream_train_policy import ActorCritic
from env import RacingEnv


@torch.no_grad()
def run_episode(policy, env, deterministic: bool = True):
    obs, _ = env.reset()
    total_r = 0.0
    steps = 0
    while True:
        obs_t = torch.as_tensor(obs, dtype=torch.float32)[None]
        dist, _value = policy.forward(obs_t)
        action = dist.mean if deterministic else dist.sample()
        action = action.clamp(-1, 1).squeeze(0).numpy()
        obs, r, done, trunc, info = env.step(action.astype(np.float32))
        total_r += r
        steps += 1
        if done or trunc:
            break
    return total_r, steps, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=str, default="checkpoints/policy.pt")
    ap.add_argument("--map", type=str, default="circuit")
    ap.add_argument("--opponent", type=str, default="still")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    policy = ActorCritic.load(args.policy, device="cpu")
    env = RacingEnv(map_name=args.map, opponent_policy=args.opponent, render=args.render)

    returns, lengths, completions = [], [], []
    for ep in range(args.episodes):
        ret, steps, info = run_episode(policy, env)
        returns.append(ret)
        lengths.append(steps)
        completions.append(info.get("route_completion", info.get("progress", 0.0)))
        print(f"  ep {ep+1:>2}: return={ret:7.1f}  length={steps:4d}  completion={completions[-1]:.3f}")

    env.close()

    returns = np.array(returns)
    print(
        f"\nREAL ENV SUMMARY ({args.episodes} episodes on {args.map}):\n"
        f"  mean return:     {returns.mean():7.1f} ± {returns.std():5.1f}\n"
        f"  mean length:     {np.mean(lengths):7.1f}\n"
        f"  mean completion: {np.mean(completions):.3f}\n"
    )


if __name__ == "__main__":
    main()
