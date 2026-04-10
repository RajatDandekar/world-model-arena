# World Model Arena — Student Starter Kit

Welcome. Your assignment: **train a world model, dream a policy inside it, and race that policy on a real simulator against your classmates.**

This is the hands-on counterpart to the IRIS and DIAMOND lectures. You will **build** both paradigms — not just read about them — and then enter a policy trained entirely inside your own world model's dreams into a head-to-head leaderboard.

---

## The 5-Step Pipeline

```
data_collect.py  →  train_wm.py  →  dream_train_policy.py  →  eval_policy_real.py  →  submit.py
     Step 1            Step 2              Step 3                    Step 4              Step 5
```

Run them in order the first time. Then iterate on the middle two.

---

## Watch it drive (3D simulator window)

After Step 3, you can open a real MetaDrive 3D window and watch your policy:

```bash
python demo_live.py --policy trained --ckpt checkpoints/policy.pt
```

**Run this in your own Terminal.app**, not inside an editor / IDE / headless
shell — MetaDrive uses Panda3D which needs an active macOS GUI session to
open a window.

Other options:

```bash
python demo_live.py --policy scripted   # no checkpoint needed
python demo_live.py --policy random     # pure exploration
```

For headless CI / grading, use `demo_record.py` instead (writes MP4).

---

## Quick Start (on a T4 Colab or an M-series Mac)

```bash
# 0. Install
conda create -n wma python=3.11 -y && conda activate wma
pip install -e ../metadrive            # or: git+https://github.com/metadriverse/metadrive.git
pip install torch numpy tensorboard gymnasium==0.29 pytest
pip install supabase python-dotenv mediapy   # for submission

# 1. Collect 200 real driving episodes (~5-10 min)
python data_collect.py --n-episodes 200

# 2. Train a world model  (--model iris  or  --model diamond)
python train_wm.py --model diamond --steps 5000

# 3. Train a policy inside its dreams  (no real env during this step!)
python dream_train_policy.py --wm checkpoints/wm.pt --wm-type diamond --iterations 500

# 4. Deploy the policy on the real simulator
python eval_policy_real.py --policy checkpoints/policy.pt

# 5. Evaluate + upload to the leaderboard (one command!)
python submit.py --tag my-first-try --wm-type diamond \
    --wm checkpoints/wm.pt --name "Your Name" --uid 12345
```

If steps 1-5 finish cleanly, you can already submit. Now the *actual* assignment begins: make it better.

---

## Submitting to the Arena

One command evaluates locally and uploads to the leaderboard:

```bash
python submit.py \
  --policy checkpoints/policy.pt \
  --wm checkpoints/wm.pt \
  --wm-type diamond \
  --tag my-agent-v3 \
  --name "Alice Zhang" \
  --uid alice123
```

**What happens when you run `submit.py`:**

1. **Real-env evaluation** — your policy runs on **5 maps × 4 episodes** on your local MetaDrive
2. **Video recording** — a top-down video of the best episode is recorded
3. **WM quality metrics** — if `--wm` is provided, evaluates on held-out data (1-step MSE, 15-step MSE, reward R², done F1)
4. **Dream-real gap** — `|dream_score - real_return| / max(|real_return|, 1)` — how much your WM hallucinated
5. **Upload** — results, metrics, and video go to the leaderboard automatically

**Setup:** Copy `.env.example` to `.env` and fill in the Supabase credentials your instructor provides:

```bash
cp .env.example .env
# Edit .env with your instructor's SUPABASE_URL and SUPABASE_KEY
```

**Options:**
- `--no-video` — skip video recording (faster)
- `--no-upload` — evaluate locally only, don't upload to leaderboard

---

## The WorldModel Protocol

Everything revolves around a single interface in [`world_model.py`](world_model.py):

```python
class WorldModel(Protocol):
    obs_dim: int
    action_dim: int
    context_len: int

    def fit(self, dataset, steps, ...) -> dict: ...
    def save(self, path) -> None: ...
    @classmethod
    def load(cls, path) -> "WorldModel": ...

    def reset_dream(self, init_obs, init_actions) -> torch.Tensor: ...
    def step_dream(self, latent, action) -> (next_latent, r, d, obs_hat): ...

    def eval_next_obs_mse(self, dataset) -> float: ...
    def eval_rollout_mse(self, dataset, horizon) -> float: ...
```

**The dream RL loop in `dream_train_policy.py` only ever calls `reset_dream()` and `step_dream()`.** That means you can swap in *any* world model — IRIS, DIAMOND, or something you invent — as long as it satisfies the Protocol. No other code needs to change.

Run the Protocol contract tests to confirm your implementation is valid:

```bash
pytest tests/test_world_model.py -q
```

---

## The Two Reference Implementations

| | `wm_iris/` | `wm_diamond/` |
|---|---|---|
| **Idea** | Discrete tokens + GPT | Continuous obs + diffusion denoiser |
| **Encoder** | VQ-VAE (512 codes, dim 32) | None — obs is the latent |
| **Dynamics** | 6-layer causal Transformer | 6-block MLP ResNet with AdaGN |
| **Training objective** | next-token CE + reward MSE + VQ loss | EDM-weighted denoising MSE |
| **Inference** | autoregressive sampling | **3-step** Euler sampler |
| **Key hyperparameters** | `n_codes`, `d_model`, `context_len` | `hidden`, `n_blocks`, `sigma_offset` |
| **Paper** | [IRIS, ICLR 2023](https://arxiv.org/abs/2209.00588) | [DIAMOND, NeurIPS 2024](https://arxiv.org/abs/2405.12399) |

Both reference models are deliberately small (~2-3M params) so they train in under 10 minutes on a T4. Both are fully functional — you can submit a pipeline with either one without changing a line.

The **real** question is: which one wins when you deploy it on the real track? Go find out.

---

## The Two Leaderboards

### Race Leaderboard (competitive, ELO)
Your policy is evaluated on **5 maps × 4 episodes** locally, then results upload to the leaderboard. ELO is computed from head-to-head route completion across all maps. This is the one you compete on.

### World Model Quality Leaderboard (scientific)
If you include `--wm`, your world model is evaluated on held-out reference data:
- **1-step MSE** — single-step prediction error (target: < 0.01)
- **15-step rollout MSE** — cumulative drift over 15 steps (target: < 0.1)
- **Reward R²** — how well your WM predicts reward (target: > 0.8)
- **Done F1** — termination classification accuracy (target: > 0.9)
- **Dream-real gap** — the gap between dream reward and real reward

The **dream-real gap** is the number to watch. A policy that scores huge in its own dream and crashes in the real env is telling you your WM is hallucinating. A small gap means your WM actually learned physics.

---

## What Metrics Are Computed

| Metric | What It Measures | Good Value |
|---|---|---|
| **Route Completion** | % of track completed (avg across 5 maps × 4 eps) | > 50% |
| **Mean Return** | Cumulative reward per episode | Higher is better |
| **ELO** | Head-to-head ranking vs other students | Starts at 1200 |
| **1-Step MSE** | WM next-obs prediction error | < 0.01 |
| **15-Step MSE** | WM open-loop drift | < 0.1 |
| **Dream-Real Gap** | `\|dream - real\| / max(\|real\|, 1)` | < 15% (advanced) |

---

## Tips for Beating the Baselines

1. **Collect more diverse data.** The reference uses 200 episodes of scripted + random driving. Try 1000. Try including crash trajectories on purpose so the WM learns termination dynamics.
2. **Scale the WM.** Bump `hidden` from 384 → 512 in DIAMOND. Bump `d_model` from 256 → 384 in IRIS. Watch GPU memory.
3. **Tune the DIAMOND noise schedule.** The paper uses `sigma_offset = 0.3`. Try 0.2, 0.4. Watch the 15-step rollout MSE.
4. **Use more sampling steps at eval time.** DIAMOND's 3 steps is the training-time sweet spot, but evaluating with 5-10 steps sometimes improves the dream-real gap.
5. **Dream for longer.** The reference rolls out 15 steps. Try 25-30. Warning: rollout errors compound fast.
6. **Shape the reward.** The WM's reward head is trained on MetaDrive's raw reward. Smooth it or add a progress bonus inside the dream RL loop.
7. **Train the policy longer.** 500 dream iterations is a starting point — 2000 is often better.
8. **Try a hybrid.** Use IRIS's VQ-VAE to get discrete tokens, then run diffusion in the *token* space. Nothing in the Protocol stops you.

---

## Grading Rubric

| Tier | Requirements | Points |
|---|---|---|
| **Baseline** | Pipeline runs; submission accepted; real-env return > 0; 1-page report identifying a failure mode you saw | 60 |
| **Intermediate** | Beat `baseline_iris` or `baseline_diamond` on ≥3 of 5 maps; dream-real gap < 30%; 3-page comparison report | 85 |
| **Advanced** | Beat both baselines on ≥4 maps; dream-real gap < 15%; novel contribution (data / architecture / training trick) with ablation study | 100 |

---

## Files in This Starter Kit

```
student_starter/
├── README.md                 ← you are here
├── world_model.py            ← the Protocol — read this first
├── env.py                    ← MetaDrive wrapper (don't touch)
├── data.py                   ← TrajectoryDataset
├── data_collect.py           ← Step 1
├── train_wm.py               ← Step 2
├── dream_train_policy.py     ← Step 3 (REINFORCE with λ-returns)
├── eval_policy_real.py       ← Step 4
├── submit.py                 ← Step 5 (evaluate + upload)
├── eval_maps.py              ← The 5 evaluation maps
├── eval_wm.py                ← Standalone WM quality checker
├── .env.example              ← Supabase credentials template
├── data/held_out/            ← 50 reference episodes for WM eval
├── wm_iris/
│   └── iris.py               ← Reference mini-IRIS (~3M params)
├── wm_diamond/
│   └── diamond.py            ← Reference mini-DIAMOND (~2.5M params)
└── tests/
    └── test_world_model.py   ← Protocol contract tests
```

---

## Help / Debugging

If your pipeline runs but your policy scores zero on the real env, check in order:

1. **Is the WM reconstruction accurate?** Print `wm.eval_next_obs_mse(dataset)`. Should be < 0.1 for the reference models.
2. **Is the rollout MSE exploding?** Print `wm.eval_rollout_mse(dataset, horizon=15)`. If this is >> next-obs MSE, your WM is drifting — collect more data or use a stronger architecture.
3. **Is the dream reward wildly optimistic?** Your policy is probably exploiting the WM. Add L2 regularization to the reward head or clip its output.
4. **Does the policy look reasonable in real env with rendering on?** `python eval_policy_real.py --render` and watch it drive. The failure mode is usually obvious.

You can also run WM metrics standalone:
```bash
python eval_wm.py --wm checkpoints/wm.pt --wm-type diamond
```

---

## Supabase Setup (Instructor Reference)

To set up the leaderboard backend for your class:

1. **Create project** at [supabase.com](https://supabase.com) → "New Project" → name: `world-model-arena`
2. **Create tables** — go to SQL Editor and run:

```sql
CREATE TABLE submissions (
  id              UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  creator_name    TEXT NOT NULL,
  creator_uid     TEXT NOT NULL,
  tag             TEXT NOT NULL,
  wm_type         TEXT NOT NULL DEFAULT 'diamond',
  mean_return     FLOAT,
  mean_length     FLOAT,
  route_completion FLOAT,
  map_scores      JSONB DEFAULT '{}'::jsonb,
  wm_mse_1step    FLOAT,
  wm_mse_rollout  FLOAT,
  wm_reward_r2    FLOAT,
  wm_done_f1      FLOAT,
  dream_score     FLOAT,
  dream_real_gap  FLOAT,
  video_url       TEXT,
  created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE episode_results (
  id              UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  submission_id   UUID REFERENCES submissions(id) ON DELETE CASCADE,
  map_name        TEXT NOT NULL,
  episode_idx     INT NOT NULL,
  return_val      FLOAT NOT NULL,
  length          INT NOT NULL,
  route_completion FLOAT NOT NULL
);

ALTER TABLE submissions ENABLE ROW LEVEL SECURITY;
ALTER TABLE episode_results ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Anyone can read submissions" ON submissions FOR SELECT USING (true);
CREATE POLICY "Anyone can insert submissions" ON submissions FOR INSERT WITH CHECK (true);
CREATE POLICY "Anyone can read episodes" ON episode_results FOR SELECT USING (true);
CREATE POLICY "Anyone can insert episodes" ON episode_results FOR INSERT WITH CHECK (true);
```

3. **Create storage bucket** — Storage → New Bucket → name: `videos`, check "Public bucket", add public read + insert policies
4. **Get credentials** — Settings → API → copy Project URL and anon key
5. **Distribute** — share the URL + key with students (they go in `.env`)
6. **Deploy leaderboard** — push `leaderboard/index.html` to Vercel (update `SUPABASE_URL` and `SUPABASE_KEY` constants in the HTML)

---

Good luck. Go dream a policy.
