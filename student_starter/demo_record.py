"""
Headless demo that records a top-down video of a MetaDrive episode and saves
it as MP4. Works even when the process can't open an on-screen window (which
is the case when launched from a non-GUI shell on macOS).

    python demo_record.py --out videos/demo.mp4

Options:
    --policy   random | scripted | trained
    --ckpt     path to policy.pt (only used when --policy trained)
    --steps    episode length
    --map      circuit | hairpin
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import numpy as np

# Reuse the same env wrapper students use, but force render=False (we render
# frames manually via the underlying metadrive renderer).
from env import RacingEnv


def scripted(step: int) -> np.ndarray:
    # Full throttle + gentle sinusoidal steer. Strong enough forward motion
    # that the car clearly circles the track and never looks idle.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="videos/demo.mp4")
    ap.add_argument("--policy", choices=["random", "scripted", "trained"], default="scripted")
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--map", type=str, default="circuit")
    ap.add_argument("--opponent", type=str, default="still")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--no-open", action="store_true", help="don't auto-open the mp4")
    args = ap.parse_args()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = RacingEnv(map_name=args.map, opponent_policy=args.opponent, render=False)
    act_fn = build_policy(args.policy, args.ckpt)

    print(f"Recording {args.steps} steps of '{args.policy}' policy on map='{args.map}'")
    obs, _ = env.reset()

    # The top-down renderer needs engine.main_camera.current_track_agent set.
    # When use_render=False there's no main_camera. Fake it.
    class _FakeMainCamera:
        def __init__(self, agent):
            self.current_track_agent = agent
    agent0 = env._env.agents["agent0"]
    env._env.engine.main_camera = _FakeMainCamera(agent0)

    frames = []
    total_r = 0.0
    terminated_at = None
    for t in range(args.steps):
        action = act_fn(obs, t)
        obs, r, done, trunc, info = env.step(action)
        total_r += r

        # Use MetaDrive's built-in top-down renderer (pygame, works headless).
        # Params mirror metadrive/examples/generate_video_for_bev_and_interface.py:
        # semantic_map=True paints colored lane surfaces so the track is
        # actually visible (default is white-on-white); draw_contour gives
        # the ego car a visible outline; swapaxes converts pygame's (W,H,C)
        # surface layout to the (H,W,C) that mediapy / ffmpeg expect.
        frame = env._env.render(
            mode="topdown",
            film_size=(3000, 3000),
            screen_size=(900, 900),
            semantic_map=True,
            draw_target_vehicle_trajectory=True,
            draw_contour=True,
        )
        if frame is not None:
            frame = np.asarray(frame).swapaxes(0, 1)
            frames.append(frame)

        if done or trunc:
            terminated_at = t
            print(f"  episode ended at step {t}  return={total_r:.1f}")
            break

    env.close()

    if not frames:
        print("ERROR: no frames captured — renderer returned None")
        return

    print(f"Captured {len(frames)} frames  ({frames[0].shape})")
    # Write video via mediapy (metadrive already installs it)
    import mediapy
    mediapy.write_video(str(out_path), frames, fps=30)
    print(f"saved → {out_path}")

    if not args.no_open:
        try:
            subprocess.Popen(["open", str(out_path)])
            print("opened in QuickTime")
        except Exception as e:
            print(f"(could not auto-open: {e})")


if __name__ == "__main__":
    main()
