"""
MetaDrive racing environment wrapper.

Thin single-agent interface over MultiAgentRacingEnv. This is a close port of
MetaDrive-Arena's env.py, stripped of the opponent-self-play features we
don't need for the world-model pipeline (we only want ego trajectories).

Usage:
    env = RacingEnv(map_name="circuit", opponent_policy="still")
    obs, _ = env.reset()
    for _ in range(100):
        obs, r, done, trunc, info = env.step(env.action_space.sample())
        if done or trunc:
            obs, _ = env.reset()
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym


def _make_opponent(name: str):
    """Returns a callable (obs) -> action for the other agent."""
    rng = np.random.default_rng(0)

    def random_policy(_obs):
        return rng.uniform(-1, 1, size=(2,)).astype(np.float32)

    def aggressive_policy(_obs):
        return np.array([0.0, 1.0], dtype=np.float32)

    def still_policy(_obs):
        return np.array([0.0, 0.0], dtype=np.float32)

    table = {
        "random": random_policy,
        "aggressive": aggressive_policy,
        "still": still_policy,
    }
    if name not in table:
        raise ValueError(f"unknown opponent policy: {name}")
    return table[name]


class RacingEnv(gym.Env):
    """Single-agent gymnasium wrapper around MetaDrive's 2-agent racing env."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        map_name: str = "circuit",
        opponent_policy: str = "still",
        render: bool = False,
    ):
        # Import inside __init__ so the starter kit imports quickly even when
        # metadrive isn't installed yet (for unit tests on CI, etc.).
        from metadrive.envs.marl_envs.marl_racing_env import MultiAgentRacingEnv

        self._opponent = _make_opponent(opponent_policy)
        self._map_name = map_name

        self._env = MultiAgentRacingEnv(
            config={
                "num_agents": 2,
                "use_render": render,
                "horizon": 3000,
                "out_of_road_done": True,
                # The racing env's default idle detector terminates episodes
                # when the car moves <10 cm in 100 steps. That's a racing
                # anti-exploit and it's counterproductive for a teaching kit:
                # slow / stalling student policies should keep accumulating
                # reward + video frames, not trip a hidden termination.
                "idle_done": False,
            }
        )

        # Probe observation / action space from the underlying env
        sample_obs_space = self._env.observation_space["agent0"]
        sample_act_space = self._env.action_space["agent0"]
        self.observation_space = sample_obs_space
        self.action_space = sample_act_space

        self._last_opp_obs = None

    # ------------------------------------------------------------------ API

    def reset(self, *, seed=None, options=None):
        obs_dict, info_dict = self._env.reset(seed=seed)
        self._last_opp_obs = obs_dict.get("agent1")
        return obs_dict["agent0"], info_dict.get("agent0", {})

    def step(self, action):
        opp_action = self._opponent(self._last_opp_obs)
        obs_dict, r_dict, d_dict, t_dict, info_dict = self._env.step(
            {"agent0": action, "agent1": opp_action}
        )
        self._last_opp_obs = obs_dict.get("agent1", self._last_opp_obs)

        obs = obs_dict["agent0"]
        reward = float(r_dict["agent0"])
        done = bool(d_dict.get("agent0", False))
        trunc = bool(t_dict.get("agent0", False))
        info = info_dict.get("agent0", {})
        return obs, reward, done, trunc, info

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass
