"""
Evaluation maps for the World Model Arena.

These are the same 5 maps used by the leaderboard. Your policy is evaluated
on each map for EPISODES_PER_MAP episodes and scored on route completion,
return, and episode length.
"""

EVAL_MAPS = {
    "curve_a": {"map_config": {"type": "block_sequence", "config": "CrCSC"}},
    "chicane": {"map_config": {"type": "block_sequence", "config": "SCSCS"}},
    "long_straight": {"map_config": {"type": "block_sequence", "config": "SSSSS"}},
    "tight_s": {"map_config": {"type": "block_sequence", "config": "CCSCC"}},
    "oval": {"map_config": {"type": "block_sequence", "config": "SCCS"}},
}

EPISODES_PER_MAP = 4
