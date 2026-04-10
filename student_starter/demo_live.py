"""
Live MetaDrive demo — opens a pygame window showing the car driving in
real time. Watch your policy move right in front of you.

    python demo_live.py --policy scripted --steps 1500
    python demo_live.py --policy trained --ckpt checkpoints/policy.pt
    python demo_live.py --policy random

Uses MetaDrive's top-down pygame renderer (SDL-based, works on all Macs).
Close the window or Ctrl-C to stop.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from env import RacingEnv


# ----------------------------------------------------------------- policies


def scripted(step: int) -> np.ndarray:
    """Full throttle + gentle sinusoidal steer. Keeps the car circling."""
    steer = 0.08 * np.sin(step * 0.03)
    throttle = 1.0
    return np.array([steer, throttle], dtype=np.float32)


def random_policy(rng) -> np.ndarray:
    return rng.uniform(-1, 1, size=(2,)).astype(np.float32)


def build_policy(name: str, ckpt: str | None):
    rng = np.random.default_rng(0)
    if name == "random":
        return lambda obs, t: random_policy(rng)
    if name == "scripted":
        return lambda obs, t: scripted(t)
    if name == "trained":
        if not ckpt:
            raise ValueError("--ckpt required when --policy trained")
        import torch
        from dream_train_policy import ActorCritic

        policy = ActorCritic.load(ckpt, device="cpu")
        policy.eval()

        def act(obs, _t):
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32)[None]
                dist, _v = policy.forward(obs_t)
                a = dist.mean.clamp(-1, 1).squeeze(0).numpy()
            return a.astype(np.float32)

        return act
    raise ValueError(f"unknown policy: {name}")


# --------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--policy",
        choices=["random", "scripted", "trained"],
        default="scripted",
    )
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--map", type=str, default="circuit")
    ap.add_argument("--opponent", type=str, default="still")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="target framerate for the live viewer",
    )
    args = ap.parse_args()

    print(
        f"Live viewer — policy='{args.policy}' "
        f"map='{args.map}' steps={args.steps}"
    )

    # Use headless engine (no Panda3D window) — we'll show a pygame window
    # via the top-down renderer instead.
    env = RacingEnv(
        map_name=args.map,
        opponent_policy=args.opponent,
        render=False,
    )

    act_fn = build_policy(args.policy, args.ckpt)
    dt = 1.0 / max(args.fps, 1.0)

    obs, _ = env.reset()

    # The top-down renderer needs engine.main_camera to track the ego car.
    # When use_render=False there's no Panda3D camera, so we fake it.
    class _FakeMainCamera:
        def __init__(self, agent):
            self.current_track_agent = agent
    agent0 = env._env.agents["agent0"]
    env._env.engine.main_camera = _FakeMainCamera(agent0)

    # First render call creates the TopDownRenderer with a pygame window.
    # window=True (default) opens a live SDL window on screen.
    import pygame

    render_kwargs = dict(
        mode="topdown",
        film_size=(3000, 3000),
        screen_size=(900, 900),
        semantic_map=True,
        draw_target_vehicle_trajectory=True,
        draw_contour=True,
    )

    print("Opening pygame window...")
    try:
        total_r = 0.0
        ep_steps = 0
        ep_idx = 0

        for t in range(args.steps):
            action = act_fn(obs, t)
            obs, r, done, trunc, info = env.step(action)
            total_r += r
            ep_steps += 1

            # Render to the live pygame window
            env._env.render(**render_kwargs)

            # Pump pygame events so the window stays responsive and
            # we can detect if the user closed it.
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Window closed by user.")
                    env.close()
                    return

            # Pace to target FPS
            time.sleep(dt)

            if done or trunc:
                print(
                    f"  episode {ep_idx} ended at step {ep_steps}  "
                    f"return={total_r:.1f}"
                )
                obs, _ = env.reset()
                # Re-attach fake camera to the (possibly new) agent0
                agent0 = env._env.agents["agent0"]
                env._env.engine.main_camera = _FakeMainCamera(agent0)
                total_r = 0.0
                ep_steps = 0
                ep_idx += 1

        print(f"Finished {args.steps} steps. Total episodes: {ep_idx + 1}")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        env.close()
        pygame.quit()


if __name__ == "__main__":
    main()
