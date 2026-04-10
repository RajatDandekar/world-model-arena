"""
Step 1: collect offline driving data from MetaDrive.

Runs a mix of opponent policies and a mix of ego policies so the logged
trajectories cover both good and bad driving behaviour — which is what
the world model actually needs to learn transitions in.

Example:
    python data_collect.py --n-episodes 500 --out data/raw
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from env import RacingEnv


def scripted_ego(obs: np.ndarray, step: int) -> np.ndarray:
    """Simple rule-based driver — drives forward, small random steer."""
    steer = np.clip(np.random.randn() * 0.2, -1.0, 1.0)
    throttle = 0.6 + 0.2 * np.sin(step * 0.01)
    return np.array([steer, throttle], dtype=np.float32)


def random_ego(_obs: np.ndarray, _step: int) -> np.ndarray:
    return np.random.uniform(-1, 1, size=(2,)).astype(np.float32)


def collect_one(env: RacingEnv, ego_fn, max_steps: int = 1000):
    obs, _ = env.reset()
    obs_buf = [obs.copy()]
    act_buf, rew_buf, done_buf = [], [], []
    for t in range(max_steps):
        action = ego_fn(obs, t)
        obs, reward, done, trunc, _info = env.step(action)
        obs_buf.append(obs.copy())
        act_buf.append(action)
        rew_buf.append(reward)
        done_buf.append(done or trunc)
        if done or trunc:
            break
    return (
        np.array(obs_buf, dtype=np.float32),
        np.array(act_buf, dtype=np.float32),
        np.array(rew_buf, dtype=np.float32),
        np.array(done_buf, dtype=bool),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-episodes", type=int, default=500)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--out", type=str, default="data/raw")
    ap.add_argument("--opponent", type=str, default="still")
    ap.add_argument("--map", type=str, default="circuit")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    env = RacingEnv(map_name=args.map, opponent_policy=args.opponent)
    print(f"obs_dim={env.observation_space.shape}  action_dim={env.action_space.shape}")

    for i in range(args.n_episodes):
        # Alternate: 60% scripted, 40% random for coverage
        ego_fn = scripted_ego if (i % 5) < 3 else random_ego
        obs, act, rew, done = collect_one(env, ego_fn, args.max_steps)
        np.savez_compressed(
            out / f"ep_{i:05d}.npz",
            obs=obs,
            action=act,
            reward=rew,
            done=done,
        )
        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{args.n_episodes}]  last ep len={len(rew)}  ret={rew.sum():.1f}")

    env.close()
    print(f"done — wrote {args.n_episodes} episodes to {out}")


if __name__ == "__main__":
    main()
