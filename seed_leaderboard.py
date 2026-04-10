"""
Seed the World Model Arena leaderboard with baseline + fake student data.

Usage:
    python seed_leaderboard.py            # Insert 7 submissions + 140 episode_results
    python seed_leaderboard.py --clear    # Wipe everything first, then re-seed

Uploads videos from student_starter/videos/ to Supabase storage.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import uuid
from pathlib import Path

# ── Load .env ──────────────────────────────────────────────────────────────

def _load_env():
    env_path = Path(__file__).parent / "student_starter" / ".env"
    if not env_path.exists():
        print(f"ERROR: {env_path} not found. Copy .env.example and fill in credentials.")
        sys.exit(1)
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


_load_env()

from supabase import create_client

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

MAPS = ["curve_a", "chicane", "long_straight", "tight_s", "oval"]
EPISODES_PER_MAP = 4

# ── Video upload ───────────────────────────────────────────────────────────

def upload_video(local_path: str, storage_key: str) -> str | None:
    """Upload a video to Supabase storage and return its public URL."""
    p = Path(local_path)
    if not p.exists():
        print(f"  WARNING: video not found: {local_path}")
        return None

    try:
        with open(p, "rb") as f:
            sb.storage.from_("videos").upload(
                storage_key, f, {"content-type": "video/mp4"}
            )
        url = f"{SUPABASE_URL}/storage/v1/object/public/videos/{storage_key}"
        print(f"  uploaded {p.name} -> {storage_key}")
        return url
    except Exception as e:
        # If already exists, just return the URL
        if "Duplicate" in str(e) or "already exists" in str(e):
            url = f"{SUPABASE_URL}/storage/v1/object/public/videos/{storage_key}"
            print(f"  {p.name} already exists in storage")
            return url
        print(f"  WARNING: upload failed: {e}")
        return None


# ── Seed data ──────────────────────────────────────────────────────────────

def generate_map_scores(mean_completion: float) -> dict:
    """Generate per-map completion scores that average to mean_completion."""
    scores = {}
    # Give each map a score with some variance around the mean
    remaining = mean_completion * len(MAPS)
    for i, m in enumerate(MAPS):
        if i == len(MAPS) - 1:
            scores[m] = max(0.0, min(1.0, remaining))
        else:
            jitter = random.uniform(-0.12, 0.12)
            val = max(0.0, min(1.0, mean_completion + jitter))
            scores[m] = val
            remaining -= val
    return scores


def generate_episodes(submission_id: str, map_scores: dict, mean_return: float) -> list[dict]:
    """Generate 4 episodes per map with realistic variance."""
    episodes = []
    for m in MAPS:
        base_completion = map_scores[m]
        for ep_idx in range(EPISODES_PER_MAP):
            comp = max(0.0, min(1.0, base_completion + random.uniform(-0.08, 0.08)))
            ret = mean_return * (comp / max(base_completion, 0.01)) + random.uniform(-5, 5)
            length = int(150 + comp * 850 + random.randint(-30, 30))
            episodes.append({
                "submission_id": submission_id,
                "map_name": m,
                "episode_idx": ep_idx,
                "return_val": round(ret, 2),
                "length": max(50, length),
                "route_completion": round(comp, 4),
            })
    return episodes


SEED_SUBMISSIONS = [
    {
        "creator_name": "Ravi Krishnan",
        "creator_uid": "ravi-k-2026",
        "tag": "diamond-tuned-v3",
        "wm_type": "diamond",
        "mean_return": 142.5,
        "mean_length": 620,
        "route_completion": 0.612,
        "wm_mse_1step": 0.003421,
        "wm_mse_rollout": 0.041200,
        "wm_reward_r2": 0.912,
        "wm_done_f1": 0.951,
        "dream_score": 155.3,
        "dream_real_gap": 0.09,
        "video_key": "seed/ravi_diamond_v3.mp4",
        "video_source": "02_trained.mp4",
    },
    {
        "creator_name": "Amol Deshmukh",
        "creator_uid": "amol-d-2026",
        "tag": "iris-big-ctx-v2",
        "wm_type": "iris",
        "mean_return": 118.7,
        "mean_length": 560,
        "route_completion": 0.558,
        "wm_mse_1step": 0.005102,
        "wm_mse_rollout": 0.068400,
        "wm_reward_r2": 0.874,
        "wm_done_f1": 0.928,
        "dream_score": 135.1,
        "dream_real_gap": 0.14,
        "video_key": "seed/amol_iris_v2.mp4",
        "video_source": "02_trained.mp4",
    },
    {
        "creator_name": "Instructor (DIAMOND)",
        "creator_uid": "instructor-diamond",
        "tag": "baseline-diamond",
        "wm_type": "diamond",
        "mean_return": 108.2,
        "mean_length": 530,
        "route_completion": 0.534,
        "wm_mse_1step": 0.004850,
        "wm_mse_rollout": 0.058100,
        "wm_reward_r2": 0.889,
        "wm_done_f1": 0.935,
        "dream_score": 121.0,
        "dream_real_gap": 0.12,
        "video_key": "seed/baseline_diamond.mp4",
        "video_source": "02_trained.mp4",
    },
    {
        "creator_name": "Priya Sharma",
        "creator_uid": "priya-s-2026",
        "tag": "diamond-500ep-v1",
        "wm_type": "diamond",
        "mean_return": 88.4,
        "mean_length": 480,
        "route_completion": 0.489,
        "wm_mse_1step": 0.006200,
        "wm_mse_rollout": 0.082300,
        "wm_reward_r2": 0.842,
        "wm_done_f1": 0.908,
        "dream_score": 108.6,
        "dream_real_gap": 0.22,
        "video_key": "seed/priya_diamond_v1.mp4",
        "video_source": "02_trained.mp4",
    },
    {
        "creator_name": "Instructor (IRIS)",
        "creator_uid": "instructor-iris",
        "tag": "baseline-iris",
        "wm_type": "iris",
        "mean_return": 72.1,
        "mean_length": 430,
        "route_completion": 0.421,
        "wm_mse_1step": 0.007400,
        "wm_mse_rollout": 0.095200,
        "wm_reward_r2": 0.810,
        "wm_done_f1": 0.892,
        "dream_score": 86.0,
        "dream_real_gap": 0.19,
        "video_key": "seed/baseline_iris.mp4",
        "video_source": "02_trained.mp4",
    },
    {
        "creator_name": "Vikram Patel",
        "creator_uid": "vikram-p-2026",
        "tag": "hybrid-v1",
        "wm_type": "diamond",
        "mean_return": 51.3,
        "mean_length": 370,
        "route_completion": 0.347,
        "wm_mse_1step": 0.009800,
        "wm_mse_rollout": 0.132000,
        "wm_reward_r2": 0.763,
        "wm_done_f1": 0.854,
        "dream_score": 67.2,
        "dream_real_gap": 0.31,
        "video_key": "seed/vikram_hybrid_v1.mp4",
        "video_source": "02_trained.mp4",
    },
    {
        "creator_name": "Instructor (Scripted)",
        "creator_uid": "instructor-scripted",
        "tag": "baseline-scripted",
        "wm_type": "none",
        "mean_return": 22.8,
        "mean_length": 250,
        "route_completion": 0.182,
        "wm_mse_1step": None,
        "wm_mse_rollout": None,
        "wm_reward_r2": None,
        "wm_done_f1": None,
        "dream_score": None,
        "dream_real_gap": None,
        "video_key": "seed/baseline_scripted.mp4",
        "video_source": "01_scripted.mp4",
    },
]


# ── Main ───────────────────────────────────────────────────────────────────

def clear_all():
    """Delete all submissions and episode_results."""
    print("Clearing all existing data...")

    # Delete episode_results first (FK constraint)
    try:
        sb.table("episode_results").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
        print("  episode_results cleared")
    except Exception as e:
        print(f"  episode_results clear failed: {e}")

    # Delete submissions
    try:
        sb.table("submissions").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
        print("  submissions cleared")
    except Exception as e:
        print(f"  submissions clear failed: {e}")

    # Clear seed videos from storage
    try:
        files = sb.storage.from_("videos").list("seed")
        if files:
            paths = [f"seed/{f['name']}" for f in files]
            sb.storage.from_("videos").remove(paths)
            print(f"  removed {len(paths)} seed videos from storage")
    except Exception as e:
        print(f"  storage clear: {e}")

    print()


def seed():
    """Insert 7 submissions + 140 episode_results + upload videos."""
    videos_dir = Path(__file__).parent / "student_starter" / "videos"

    random.seed(42)  # Reproducible episode generation

    total_episodes = 0

    for i, entry in enumerate(SEED_SUBMISSIONS):
        print(f"\n[{i+1}/{len(SEED_SUBMISSIONS)}] {entry['creator_name']} ({entry['tag']})")

        # Upload video
        video_url = None
        video_path = videos_dir / entry["video_source"]
        if video_path.exists():
            video_url = upload_video(str(video_path), entry["video_key"])

        # Generate map_scores
        map_scores = generate_map_scores(entry["route_completion"])

        # Build submission row
        submission = {
            "creator_name": entry["creator_name"],
            "creator_uid": entry["creator_uid"],
            "tag": entry["tag"],
            "wm_type": entry["wm_type"],
            "mean_return": entry["mean_return"],
            "mean_length": entry["mean_length"],
            "route_completion": entry["route_completion"],
            "map_scores": map_scores,
            "video_url": video_url,
        }

        # Add WM metrics for non-scripted
        if entry["wm_mse_1step"] is not None:
            submission["wm_mse_1step"] = entry["wm_mse_1step"]
            submission["wm_mse_rollout"] = entry["wm_mse_rollout"]
            submission["wm_reward_r2"] = entry["wm_reward_r2"]
            submission["wm_done_f1"] = entry["wm_done_f1"]
            submission["dream_score"] = entry["dream_score"]
            submission["dream_real_gap"] = entry["dream_real_gap"]

        # Insert submission
        result = sb.table("submissions").insert(submission).execute()
        submission_id = result.data[0]["id"]
        print(f"  submission -> {submission_id}")

        # Generate + insert episode results
        episodes = generate_episodes(submission_id, map_scores, entry["mean_return"])
        sb.table("episode_results").insert(episodes).execute()
        total_episodes += len(episodes)
        print(f"  {len(episodes)} episode results inserted")

    print(f"\n{'='*60}")
    print(f"  SEED COMPLETE")
    print(f"  {len(SEED_SUBMISSIONS)} submissions, {total_episodes} episode results")
    print(f"{'='*60}")


def main():
    ap = argparse.ArgumentParser(description="Seed the World Model Arena leaderboard")
    ap.add_argument("--clear", action="store_true", help="Wipe all data before seeding")
    args = ap.parse_args()

    if args.clear:
        clear_all()

    seed()


if __name__ == "__main__":
    main()
