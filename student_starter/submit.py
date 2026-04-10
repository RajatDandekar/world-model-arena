"""
Submit to the World Model Arena — evaluate locally and upload to the leaderboard.

One command does everything:
  1. Evaluates your policy on 5 maps × 4 episodes (real MetaDrive)
  2. Records a top-down video of the best episode
  3. Computes world model quality metrics (if --wm provided)
  4. Uploads results + video to the Supabase leaderboard

Usage:
    python submit.py --tag my-agent-v3 --name "Alice Zhang" --uid alice123
    python submit.py --tag diamond-v5 --wm checkpoints/wm.pt --wm-type diamond
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def _load_env():
    """Load .env file if present."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def _get_supabase():
    """Initialize Supabase client from environment."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        print("ERROR: SUPABASE_URL and SUPABASE_KEY must be set.")
        print("  Copy .env.example to .env and fill in your credentials.")
        sys.exit(1)
    try:
        from supabase import create_client
    except ImportError:
        print("ERROR: 'supabase' package not installed. Run: pip install supabase python-dotenv")
        sys.exit(1)
    return create_client(url, key)


# ── Step 1: Evaluate on real MetaDrive ────────────────────────────────────

def evaluate_policy(policy_path: str) -> dict:
    """Run policy on 5 maps × 4 episodes. Returns metrics + per-episode results."""
    from dream_train_policy import ActorCritic
    from env import RacingEnv
    from eval_maps import EVAL_MAPS, EPISODES_PER_MAP

    policy = ActorCritic.load(policy_path, device="cpu")
    policy.eval()

    all_episodes = []
    map_scores = {}
    best_episode = {"return": -float("inf"), "map": None, "idx": 0}

    for map_name, map_cfg in EVAL_MAPS.items():
        env = RacingEnv(map_name=map_name, opponent_policy="still", render=False)
        map_returns, map_lengths, map_completions = [], [], []

        for ep_idx in range(EPISODES_PER_MAP):
            obs, _ = env.reset()
            total_r = 0.0
            steps = 0
            while True:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, dtype=torch.float32)[None]
                    dist, _ = policy.forward(obs_t)
                    action = dist.mean.clamp(-1, 1).squeeze(0).numpy().astype(np.float32)
                obs, r, done, trunc, info = env.step(action)
                total_r += r
                steps += 1
                if done or trunc:
                    break

            completion = info.get("route_completion", info.get("progress", 0.0))
            map_returns.append(total_r)
            map_lengths.append(steps)
            map_completions.append(completion)

            all_episodes.append({
                "map_name": map_name,
                "episode_idx": ep_idx,
                "return_val": float(total_r),
                "length": steps,
                "route_completion": float(completion),
            })

            if total_r > best_episode["return"]:
                best_episode = {"return": total_r, "map": map_name, "idx": ep_idx}

            print(f"  {map_name} ep {ep_idx+1}/{EPISODES_PER_MAP}: "
                  f"return={total_r:7.1f}  length={steps:4d}  completion={completion:.3f}")

        env.close()
        map_scores[map_name] = float(np.mean(map_completions))

    returns = [e["return_val"] for e in all_episodes]
    lengths = [e["length"] for e in all_episodes]
    completions = [e["route_completion"] for e in all_episodes]

    return {
        "mean_return": float(np.mean(returns)),
        "mean_length": float(np.mean(lengths)),
        "route_completion": float(np.mean(completions)),
        "map_scores": map_scores,
        "episodes": all_episodes,
        "best_episode": best_episode,
    }


# ── Step 2: Record video of best episode ─────────────────────────────────

def record_best_episode(policy_path: str, map_name: str, out_path: str, max_steps: int = 1000) -> str | None:
    """Record a top-down video of the policy on the given map."""
    try:
        import mediapy
    except ImportError:
        print("  (skipping video — install mediapy: pip install mediapy)")
        return None

    from dream_train_policy import ActorCritic
    from env import RacingEnv

    policy = ActorCritic.load(policy_path, device="cpu")
    policy.eval()

    env = RacingEnv(map_name=map_name, opponent_policy="still", render=False)
    obs, _ = env.reset()

    # Set up top-down camera
    class _FakeMainCamera:
        def __init__(self, agent):
            self.current_track_agent = agent
    agent0 = env._env.agents["agent0"]
    env._env.engine.main_camera = _FakeMainCamera(agent0)

    frames = []
    for t in range(max_steps):
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32)[None]
            dist, _ = policy.forward(obs_t)
            action = dist.mean.clamp(-1, 1).squeeze(0).numpy().astype(np.float32)
        obs, r, done, trunc, info = env.step(action)

        frame = env._env.render(
            mode="topdown",
            film_size=(3000, 3000),
            screen_size=(900, 900),
            semantic_map=True,
            draw_target_vehicle_trajectory=True,
            draw_contour=True,
        )
        if frame is not None:
            frames.append(np.asarray(frame).swapaxes(0, 1))

        if done or trunc:
            break

    env.close()

    if not frames:
        print("  (no frames captured — skipping video)")
        return None

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(out_path, frames, fps=30)
    print(f"  video → {out_path} ({len(frames)} frames)")
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Evaluate locally and submit to the World Model Arena")
    ap.add_argument("--policy", type=str, default="checkpoints/policy.pt",
                    help="Path to trained policy checkpoint")
    ap.add_argument("--wm", type=str, default=None,
                    help="Path to world model checkpoint (optional, for WM leaderboard)")
    ap.add_argument("--wm-type", choices=["iris", "diamond", "custom"], default="diamond")
    ap.add_argument("--tag", type=str, required=True, help="Submission tag (e.g. my-agent-v3)")
    ap.add_argument("--name", type=str, default="Student", help="Your display name")
    ap.add_argument("--uid", type=str, default="000000000", help="Your unique student ID")
    ap.add_argument("--no-video", action="store_true", help="Skip video recording")
    ap.add_argument("--no-upload", action="store_true", help="Evaluate locally only, don't upload")
    args = ap.parse_args()

    _load_env()

    policy_path = Path(args.policy)
    if not policy_path.exists():
        print(f"ERROR: Policy checkpoint not found: {policy_path}")
        sys.exit(1)

    # ── 1. Evaluate on real MetaDrive ─────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  World Model Arena — Local Evaluation")
    print(f"  Tag: {args.tag}  |  Name: {args.name}  |  UID: {args.uid}")
    print(f"{'='*60}\n")

    print("Step 1/4: Evaluating policy on 5 maps × 4 episodes...")
    eval_results = evaluate_policy(str(policy_path))

    print(f"\n  Mean return:     {eval_results['mean_return']:7.1f}")
    print(f"  Mean length:     {eval_results['mean_length']:7.1f}")
    print(f"  Route completion: {eval_results['route_completion']:.3f}")
    print(f"  Per-map: {json.dumps(eval_results['map_scores'], indent=None)}")

    # ── 2. Record video ───────────────────────────────────────────────
    video_path = None
    if not args.no_video:
        best = eval_results["best_episode"]
        print(f"\nStep 2/4: Recording video on best map ({best['map']})...")
        video_path = record_best_episode(
            str(policy_path),
            best["map"],
            f"videos/{args.tag}.mp4",
        )
    else:
        print("\nStep 2/4: Skipping video (--no-video)")

    # ── 3. WM quality metrics ─────────────────────────────────────────
    wm_metrics = {}
    if args.wm and Path(args.wm).exists():
        print(f"\nStep 3/4: Computing world model metrics...")
        from eval_wm import evaluate_wm
        wm_metrics = evaluate_wm(args.wm, args.wm_type, "data/held_out", str(policy_path))
        if wm_metrics.get("error"):
            print(f"  WARNING: WM eval error: {wm_metrics['error']}")
        else:
            print(f"  1-step MSE:  {wm_metrics['wm_mse_1step']:.6f}")
            print(f"  15-step MSE: {wm_metrics['wm_mse_rollout']:.6f}")
            print(f"  Reward R²:   {wm_metrics['wm_reward_r2']:.4f}")
            print(f"  Done F1:     {wm_metrics['wm_done_f1']:.4f}")
            if wm_metrics.get("dream_score") is not None:
                print(f"  Dream score: {wm_metrics['dream_score']:.2f}")
    else:
        print(f"\nStep 3/4: Skipping WM metrics (no --wm provided)")

    # Compute dream-real gap
    dream_score = wm_metrics.get("dream_score")
    dream_real_gap = None
    if dream_score is not None:
        real = eval_results["mean_return"]
        dream_real_gap = abs(dream_score - real) / max(abs(real), 1.0)

    # ── 4. Upload to Supabase ─────────────────────────────────────────
    if args.no_upload:
        print(f"\nStep 4/4: Skipping upload (--no-upload)")
        _print_summary(eval_results, wm_metrics, dream_real_gap)
        return

    print(f"\nStep 4/4: Uploading to leaderboard...")
    sb = _get_supabase()

    # Upload video
    video_url = None
    if video_path and Path(video_path).exists():
        video_key = f"{args.uid}/{args.tag}/{int(time.time())}.mp4"
        with open(video_path, "rb") as f:
            sb.storage.from_("videos").upload(video_key, f, {"content-type": "video/mp4"})
        # Build public URL
        supabase_url = os.environ["SUPABASE_URL"]
        video_url = f"{supabase_url}/storage/v1/object/public/videos/{video_key}"
        print(f"  video uploaded → {video_url}")

    # Insert submission
    submission = {
        "creator_name": args.name,
        "creator_uid": args.uid,
        "tag": args.tag,
        "wm_type": args.wm_type,
        "mean_return": eval_results["mean_return"],
        "mean_length": eval_results["mean_length"],
        "route_completion": eval_results["route_completion"],
        "map_scores": eval_results["map_scores"],
        "video_url": video_url,
    }

    # Add WM metrics if available
    if wm_metrics and not wm_metrics.get("error"):
        submission["wm_mse_1step"] = wm_metrics.get("wm_mse_1step")
        submission["wm_mse_rollout"] = wm_metrics.get("wm_mse_rollout")
        submission["wm_reward_r2"] = wm_metrics.get("wm_reward_r2")
        submission["wm_done_f1"] = wm_metrics.get("wm_done_f1")
        submission["dream_score"] = wm_metrics.get("dream_score")
        submission["dream_real_gap"] = dream_real_gap

    result = sb.table("submissions").insert(submission).execute()
    submission_id = result.data[0]["id"]
    print(f"  submission → {submission_id}")

    # Insert per-episode results
    episode_rows = [
        {**ep, "submission_id": submission_id}
        for ep in eval_results["episodes"]
    ]
    sb.table("episode_results").insert(episode_rows).execute()
    print(f"  {len(episode_rows)} episode results uploaded")

    _print_summary(eval_results, wm_metrics, dream_real_gap)
    print(f"\n  View the leaderboard at your instructor's URL")
    print(f"  Submission ID: {submission_id}")


def _print_summary(eval_results: dict, wm_metrics: dict, dream_real_gap: float | None):
    """Print a final results summary."""
    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Route completion: {eval_results['route_completion']:.1%}")
    print(f"  Mean return:      {eval_results['mean_return']:.1f}")
    print(f"  Mean length:      {eval_results['mean_length']:.0f}")

    if wm_metrics and not wm_metrics.get("error"):
        print(f"\n  WM 1-step MSE:    {wm_metrics.get('wm_mse_1step', 0):.6f}")
        print(f"  WM 15-step MSE:   {wm_metrics.get('wm_mse_rollout', 0):.6f}")
        print(f"  Reward R²:        {wm_metrics.get('wm_reward_r2', 0):.4f}")
        print(f"  Done F1:          {wm_metrics.get('wm_done_f1', 0):.4f}")

    if dream_real_gap is not None:
        tier = "Advanced (<15%)" if dream_real_gap < 0.15 else \
               "Intermediate (<30%)" if dream_real_gap < 0.30 else \
               "Keep pushing!"
        print(f"\n  Dream-Real Gap:   {dream_real_gap:.1%}  ({tier})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
