"""
Instructor scripted baseline — drives forward with random steering.
No world model, no learned policy. Students must beat this with dream RL.
"""

import numpy as np


class Policy:
    CREATOR_NAME = "Instructor (Scripted)"
    CREATOR_UID = "instructor"

    def __init__(self):
        self.step = 0
        self.rng = np.random.default_rng(42)

    def reset(self):
        self.step = 0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        steer = float(np.clip(self.rng.standard_normal() * 0.15, -0.5, 0.5))
        throttle = 0.5 + 0.1 * np.sin(self.step * 0.01)
        self.step += 1
        return np.array([steer, throttle], dtype=np.float32)
