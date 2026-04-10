"""
Smoke-test demo: opens MetaDrive with a 3D window and drives for 500 steps
with a "forward-with-light-steering" scripted policy so you can SEE the car
moving on the track.

This is the warmup demo — no world model, no policy training, just proof that
the simulator renders on your machine.

    python demo_random.py
"""

from __future__ import annotations

import time

import numpy as np

from env import RacingEnv


def scripted(step: int) -> np.ndarray:
    """Forward with a gentle sinusoidal steer so the car follows the track."""
    steer = 0.15 * np.sin(step * 0.03)
    throttle = 0.5
    return np.array([steer, throttle], dtype=np.float32)


def main(max_steps: int = 500):
    print("Opening MetaDrive window... (first launch can take ~10-20s)")
    env = RacingEnv(map_name="circuit", opponent_policy="still", render=True)
    obs, _ = env.reset()
    print(f"Simulator ready. obs_dim={obs.shape[0]}. Driving for {max_steps} steps.")

    total_r = 0.0
    for t in range(max_steps):
        action = scripted(t)
        obs, r, done, trunc, info = env.step(action)
        total_r += r
        if done or trunc:
            print(f"episode ended at step {t}  reward={total_r:.1f}")
            obs, _ = env.reset()
            total_r = 0.0
        time.sleep(0.01)  # slow it down so the motion is visible

    env.close()
    print("demo finished — car parked.")


if __name__ == "__main__":
    main()
