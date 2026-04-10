# World Model Arena — Design Document

**Version:** 0.1 (draft)
**Author:** Raj + Claude
**Status:** Spec for implementation
**Parent project:** Full fork of [`VAIL-UCLA/MetaDrive-Arena`](https://github.com/VAIL-UCLA/MetaDrive-Arena)
**Related reading:** `MetaDrive_Arena_Analysis.md`

---

## 1. One-Line Pitch

> **World Model Arena** is MetaDrive-Arena, but instead of training a policy directly on the simulator, students train a **world model** from logged data and then train their driving policy **entirely inside that world model's dreams**. The policy is then deployed on the real MetaDrive simulator and ranked on a public leaderboard.

This gives students a concrete, head-to-head competitive platform that forces them to understand **exactly** what IRIS and DIAMOND are doing — not just read about them.

---

## 2. Why This Is the Right Project

The DIAMOND Part 5 lecture ends with a question: *"Can you actually train a policy from dreams?"* Every student will say yes. The arena makes them prove it.

Unlike standard RL benchmarks, the world model pipeline has three failure modes that are **invisible** unless you compete against other students:

| Failure mode | What it looks like | Why the arena surfaces it |
|---|---|---|
| **Dream collapse** | WM reconstructs well but policy trained in dreams crashes in real env | Leaderboard uses real env — dream-env training is blind to it |
| **Exploitation of model errors** | Policy learns to "glitch" the WM into giving free reward | Real env deployment instantly kills the score |
| **Short-horizon overfitting** | Training on 15-step rollouts → policy can't handle 3000-step real episodes | Real eval runs full episodes |

A student only understands these once they watch their policy work beautifully in their own dream and then drive straight into a wall in a head-to-head match.

---

## 3. Learning Objectives

By the end of the arena module, students should be able to:

1. **Collect** a balanced offline driving dataset from MetaDrive (random + scripted + teacher rollouts).
2. **Train an encoder** (VQ-VAE for IRIS path, or Cosmos-style continuous VAE for DIAMOND path).
3. **Train a dynamics model** — either
   - a Transformer on discrete tokens (mini-IRIS), or
   - an EDM-preconditioned U-Net denoiser (mini-DIAMOND).
4. **Train a policy inside the learned world model** using dream rollouts + REINFORCE with λ-returns.
5. **Deploy** that policy on the real MetaDrive simulator via the arena's submission API.
6. **Diagnose** a mismatch between "dream score" and "real score" and iterate.

These map directly to the IRIS and DIAMOND paper contributions — **by building them, students internalize them**.

---

## 4. End-to-End Student Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  1. COLLECT      python data_collect.py --n-episodes 500            │
│                  → data/raw/*.npz (obs, actions, rewards, dones)    │
│                                                                     │
│  2. TRAIN WM     python train_wm.py --model diamond                 │
│                  → checkpoints/wm_diamond.pt  (or wm_iris.pt)       │
│                                                                     │
│  3. DREAM RL     python dream_train_policy.py --wm diamond          │
│                  → checkpoints/policy.pt                            │
│                                                                     │
│  4. LOCAL EVAL   python eval_policy_real.py                         │
│                  → "real env reward: 1243 ± 84"                     │
│                                                                     │
│  5. SUBMIT       python submit.py --tag "diamond-v3"                │
│                  → arena.vizuara.com/leaderboard                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

Each step is a single script with sensible defaults. Students can run the whole pipeline end-to-end in **under 90 minutes on a T4**, or under 30 minutes on an M3 Pro.

---

## 5. Repository Layout

Mirrors MetaDrive-Arena one-for-one, with world-model-specific code added.

```
world-model-arena/
├── README.md                         # 5-minute quick start
├── WorldModel-Arena-Design.md        # this document
├── environment.yml                   # conda spec (py3.11, torch, metadrive, ...)
│
├── student_starter/                  # WHAT STUDENTS CLONE
│   ├── README.md                     # assignment spec + grading rubric
│   ├── env.py                        # MetaDrive wrapper (identical to MD-Arena)
│   ├── racing_maps.py                # 2 public maps (circuit, hairpin)
│   │
│   ├── data_collect.py               # Step 1: collect offline data
│   ├── data/                         # raw trajectories live here
│   │
│   ├── world_model.py                # shared Protocol interface
│   ├── wm_iris/                      # mini-IRIS reference implementation
│   │   ├── vqvae.py                  # VQ-VAE encoder/decoder
│   │   ├── transformer.py            # causal GPT over tokens
│   │   └── iris.py                   # full IRIS wrapper
│   ├── wm_diamond/                   # mini-DIAMOND reference implementation
│   │   ├── unet.py                   # conv U-Net denoiser
│   │   ├── edm.py                    # EDM preconditioning + sampler
│   │   └── diamond.py                # full DIAMOND wrapper
│   │
│   ├── train_wm.py                   # Step 2: world model training
│   ├── dream_train_policy.py         # Step 3: RL inside dreams
│   ├── eval_policy_real.py           # Step 4: real env evaluation
│   ├── submit.py                     # Step 5: upload to arena
│   │
│   ├── agents/
│   │   ├── baseline_iris/            # reference agent — student target
│   │   │   ├── agent.py              # Policy class
│   │   │   ├── model.pt              # trained in mini-IRIS dreams
│   │   │   └── README.md
│   │   └── baseline_diamond/         # reference agent — student target
│   │       ├── agent.py
│   │       ├── model.pt              # trained in mini-DIAMOND dreams
│   │       └── README.md
│   │
│   └── tests/
│       └── test_world_model.py       # WM contract tests (see Section 8)
│
├── server/                           # WHAT THE ORGANIZER DEPLOYS
│   ├── app.py                        # FastAPI routes
│   ├── worker.py                     # job queue daemon
│   ├── evaluator.py                  # loads student agent, runs real env
│   ├── wm_verifier.py                # NEW: validates the uploaded WM
│   ├── database.py                   # SQLite schema
│   ├── config.py                     # MAX_WORKERS, NUM_GPUS, timeouts
│   ├── server_maps.py                # 5 private maps
│   └── frontend/                     # React dashboard (can start as templated HTML)
│
└── infrastructure/
    ├── Dockerfile.evaluator          # sandboxed eval container
    ├── docker-compose.yml
    └── systemd/                      # server service files
```

---

## 6. The World Model Protocol — The Single Most Important Interface

Every student must implement a class that satisfies this Protocol. The entire starter kit (training scripts, dream RL loop, tests) is written against it.

```python
# student_starter/world_model.py
from typing import Protocol, Tuple
import torch
import numpy as np

class WorldModel(Protocol):
    """
    The contract every student's world model must satisfy.

    Design notes:
    - Everything is torch tensors on whatever device the student chose.
    - Batch-first. Shapes are documented per method.
    - The dream loop only ever calls reset_dream / step_dream.
    - fit() is called by train_wm.py; it is the student's creative freedom.
    """

    device: torch.device
    obs_dim: int           # real env obs dimension (91 for MetaDrive)
    action_dim: int        # real env action dimension (2 for MetaDrive)

    # ---------- training ----------
    def fit(self, dataset: "TrajectoryDataset", steps: int) -> dict:
        """Train WM on offline data. Returns metrics dict."""
        ...

    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> "WorldModel": ...

    # ---------- imagination / dreaming ----------
    def reset_dream(self, batch_size: int, init_obs: torch.Tensor) -> torch.Tensor:
        """
        Seed a dream from real observations.
        init_obs: (B, context_len, obs_dim) — last few real steps
        Returns: (B, latent_dim) — initial hidden/latent state used by step_dream
        """
        ...

    def step_dream(
        self,
        latent: torch.Tensor,          # (B, latent_dim)
        action: torch.Tensor,          # (B, action_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One step of imagined rollout.
        Returns:
            next_latent: (B, latent_dim)
            reward_hat:  (B,)     — predicted scalar reward
            done_hat:    (B,)     — predicted done probability in [0, 1]
            obs_hat:     (B, obs_dim) — decoded observation (for the policy)
        """
        ...
```

**Why obs_dim = 91?** MetaDrive's racing env returns a ~91-D lidar + ego + nav vector. We deliberately **do not** use pixel observations on the real track — they're expensive and not the point of this class. The WM still learns a rich dynamics model; it's just the lidar-space version of IRIS/DIAMOND. Section 11 discusses a pixel-based extension.

---

## 7. Reference Implementation 1 — mini-IRIS (Discrete Token Path)

```
obs_t        ─► VQ-VAE encoder ──► discrete token z_t ∈ [0, K)
                                         │
                                         ▼
                         ┌──────────────────────────────┐
                         │ Causal Transformer (GPT-lite) │
                         │  context: [z_{t-H}..z_t, a_t] │
                         │  predict: z_{t+1}, r_t, d_t   │
                         └──────────────────────────────┘
                                         │
                                         ▼
                              z_{t+1} ──► VQ-VAE decoder ──► ô_{t+1}
```

| Component | Size | Notes |
|---|---|---|
| **VQ-VAE encoder** | obs (91) → 2-layer MLP → quantize | Codebook K=512, code dim=32 |
| **VQ-VAE decoder** | code → 2-layer MLP → obs (91) | MSE reconstruction loss |
| **Transformer** | 6 layers, d=256, heads=8, ctx=32 | Predicts next-token + reward + done |
| **Params** | ~3M total | Trains in ~10 min on T4 |

**Training objective (multi-task):**

```
L = L_vq (reconstruction + codebook + commitment)
  + L_token (cross-entropy on z_{t+1})
  + L_reward (MSE on r_t)
  + L_done (BCE on d_t)
```

Dream rollout is pure autoregressive sampling — exactly what IRIS does.

---

## 8. Reference Implementation 2 — mini-DIAMOND (Continuous Diffusion Path)

```
obs_{t-3..t-1}, a_{t-1..t} ─► cond ─┐
                                      ▼
                      ┌─────────────────────────────┐
                      │  EDM-preconditioned U-Net   │
                      │  denoises ε → ô_t           │
                      │  3-step Euler sampler       │
                      └─────────────────────────────┘
                                      │
                                      ▼
                                    ô_t  ──► reward head ──► r_t
                                        └──► done head    ──► d_t
```

Because observations are 91-D vectors (not images), the "U-Net" is actually a **1D MLP ResNet** with AdaGN conditioning on past frames and actions. Same EDM preconditioning — same 3-step sweet spot — same lesson as the paper.

| Component | Size | Notes |
|---|---|---|
| **Denoiser** | 6 ResBlocks, d=384, AdaGN on (past obs, actions, σ) | ~2.5M params |
| **Preconditioner** | c_skip, c_out, c_in, c_noise per Karras 2022 | σ_data=0.5 |
| **Training noise** | ln σ ~ N(-0.4, 1.2²), loss weight per EDM | Offset noise σ_offset=0.3 |
| **Sampler** | Euler, 3 steps, σ_max=5.0, σ_min=0.02, ρ=7 | Deterministic, paper sweet spot |
| **Reward head** | Small 2-layer MLP on denoised obs | MSE |
| **Done head** | Small 2-layer MLP on denoised obs | BCE |

Dream rollout: 3-step denoise → feed as "next obs" → repeat for H=15 steps.

Identical in spirit to the paper — just lidar-vectors instead of Atari frames.

---

## 9. Dream RL Loop (identical for both WMs)

```python
# dream_train_policy.py (simplified)
policy   = ActorCritic(obs_dim, action_dim)
wm       = WorldModel.load(args.wm_checkpoint)
dataset  = TrajectoryDataset("data/raw")

for iteration in range(args.iterations):
    # 1. sample real-seed states
    init_obs = dataset.sample_context(batch=64, ctx_len=4)  # (64, 4, 91)
    latent   = wm.reset_dream(batch_size=64, init_obs=init_obs)

    obs_traj, act_traj, rew_traj, val_traj, logp_traj = [], [], [], [], []

    # 2. imagine 15 steps
    for t in range(15):
        obs_hat                       = wm.decode(latent)       # or cached
        action, logp, value           = policy(obs_hat)
        latent, r_hat, d_hat, obs_hat = wm.step_dream(latent, action)
        obs_traj.append(obs_hat);  act_traj.append(action)
        rew_traj.append(r_hat);    val_traj.append(value)
        logp_traj.append(logp)

    # 3. λ-return targets
    returns = lambda_returns(rew_traj, val_traj, gamma=0.985, lam=0.95)

    # 4. REINFORCE + value loss + entropy bonus
    loss = -(logp_traj * (returns - val_traj).detach()).mean() \
         + 0.5 * (val_traj - returns).pow(2).mean()             \
         - 0.01 * entropy(policy, obs_traj)

    loss.backward();  optim.step();  optim.zero_grad()
```

Students don't touch the loop — they just swap `WorldModel` implementations. **The whole point is that the dream loop is model-agnostic.**

---

## 10. Ranking, Leaderboard, and Metrics

The arena serves **two** leaderboards — one scientific, one competitive.

### 10a. Real-Env Leaderboard (competitive)

Identical to MetaDrive-Arena. Submitted policy is run on 5 private maps × 4 episodes, ranked by ELO on head-to-head matchups.

**Primary metric:** `mean_route_completion` on private maps.
**ELO update:** K=32, per-match, against the existing pool.

### 10b. World Model Leaderboard (scientific)

A **separate** leaderboard that evaluates the world model itself — not the policy. When a student submits, the evaluator runs:

| Metric | What it measures | Good value |
|---|---|---|
| **Next-obs MSE** | 1-step prediction error on held-out data | < 0.01 |
| **15-step rollout MSE** | cumulative open-loop drift | < 0.1 |
| **Reward R²** | how well reward head predicts true reward | > 0.8 |
| **Done F1** | termination classification quality | > 0.9 |
| **Dream-real score gap** | `dream_eval_reward − real_eval_reward` | close to 0 |

The dream-real gap is the **most instructive single number** — it quantifies how much your WM is lying to your policy. Students who push this gap below 10% are the ones who truly understood the lecture.

### 10c. Scoring Rubric for Grading

```
Grade = 0.4 · (real env ELO percentile)
      + 0.3 · (world model MSE percentile, inverted)
      + 0.2 · (dream-real gap, inverted)
      + 0.1 · (code quality + README writeup)
```

---

## 11. Server Additions Beyond MetaDrive-Arena

Most of the server code is reusable. The **new** parts are:

### 11a. New Table — `world_models`

```sql
CREATE TABLE world_models (
    id INTEGER PRIMARY KEY,
    submission_id INTEGER REFERENCES submissions(id),
    wm_type TEXT CHECK (wm_type IN ('iris', 'diamond', 'custom')),
    next_obs_mse REAL,
    rollout_mse_15 REAL,
    reward_r2 REAL,
    done_f1 REAL,
    dream_score REAL,
    real_score REAL,
    dream_real_gap REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 11b. New Verifier — `wm_verifier.py`

On upload, the server runs the student's WM against a **held-out reference dataset** that never leaves the server, measures all the metrics above, and stores them in `world_models`. This happens **before** any policy race is scheduled.

### 11c. Optional — `dream_score` endpoint

```
POST /api/wm/rollout  { submission_id, init_obs }
      → returns the 15-step rollout (obs_hat, rew_hat) JSON
      → frontend renders a side-by-side real vs dream animation
```

This is the **showcase feature** that makes the competition memorable. Students watch a replay of "here's what my WM thinks will happen" vs "here's what actually happens." Nothing teaches model-based RL like watching your own dream go wrong.

### 11d. Sandboxing requirements — same as MetaDrive-Arena + extras

- Eval container must have `torch` **on GPU** (WM inference is nontrivial).
- Per-submission disk quota: **200MB** (WM checkpoints are bigger than policy checkpoints).
- Wall-clock timeout: **180s** for WM verification + **1800s** for the policy race (same as MD-Arena).

---

## 12. Starter Kit Contents (Student-Facing)

What students get when they clone the repo:

| File | Purpose | Lines |
|---|---|---|
| `data_collect.py` | Collect N episodes with random + scripted opponents | ~150 |
| `world_model.py` | The Protocol interface (skim and understand) | ~80 |
| `wm_iris/iris.py` | Full mini-IRIS (trains to baseline level) | ~400 |
| `wm_diamond/diamond.py` | Full mini-DIAMOND (trains to baseline level) | ~400 |
| `train_wm.py` | `--model {iris,diamond,custom}` dispatcher | ~120 |
| `dream_train_policy.py` | REINFORCE-λ inside any WM | ~200 |
| `eval_policy_real.py` | Deploy policy on real MetaDrive | ~100 |
| `submit.py` | Package + upload to arena | ~80 |
| `tests/test_world_model.py` | Protocol contract tests | ~60 |

**Two reference agents** ship pre-trained:
- `agents/baseline_iris/` — achievable by running the scripts verbatim
- `agents/baseline_diamond/` — same, for the diffusion path

Beating either is the baseline assignment. Beating **both while dream-real gap < 15%** is the stretch goal.

---

## 13. What Students Must Do vs What They Get for Free

| Task | Given | Student implements |
|---|---|---|
| Environment wrapper | ✔ `env.py` | — |
| Data collection | ✔ `data_collect.py` | Optional: better opponent policies |
| VQ-VAE | ✔ reference | Optional: tune K, code dim |
| IRIS transformer | ✔ reference | Optional: bigger context, better tokenization |
| DIAMOND U-Net | ✔ reference | Optional: more layers, offset noise tuning |
| EDM preconditioning | ✔ `wm_diamond/edm.py` | — (read and understand) |
| Dream RL loop | ✔ `dream_train_policy.py` | — |
| Policy architecture | ✔ small actor-critic | Optional: bigger, frame-stacking |
| Real eval | ✔ `eval_policy_real.py` | — |
| Submission | ✔ `submit.py` | — |

The starter kit is deliberately **fully functional end-to-end**. A student who changes nothing can already submit and appear on the leaderboard. The creative freedom is in the middle: which WM architecture, which data, which tricks.

---

## 14. Assignment Rubric (for course use)

### Baseline (60 points)
- Successfully run the pipeline end-to-end
- Submit an agent that scores > 0 on the real-env leaderboard
- Write a 1-page report identifying one failure mode you saw

### Intermediate (85 points)
- Beat the baseline IRIS agent OR baseline DIAMOND agent on ≥3 of 5 private maps
- Dream-real gap < 30%
- 3-page report comparing the two paradigms

### Advanced (100 points)
- Beat both baselines on ≥4 maps
- Dream-real gap < 15%
- Novel contribution: data augmentation, reward shaping, self-play data collection, or architectural tweak documented in the report
- 5-page report with ablation studies

---

## 15. Build Plan

### Milestone 1 — Local Prototype (no server)
- [x] Design doc (this file)
- [ ] `student_starter/env.py` (fork from MD-Arena)
- [ ] `student_starter/world_model.py` (Protocol)
- [ ] `student_starter/data_collect.py`
- [ ] `student_starter/wm_iris/` reference implementation
- [ ] `student_starter/wm_diamond/` reference implementation
- [ ] `student_starter/train_wm.py`
- [ ] `student_starter/dream_train_policy.py`
- [ ] `student_starter/eval_policy_real.py`
- [ ] Verify: WM reference achieves MSE < 0.05, dream policy achieves real reward > 500

### Milestone 2 — Contract tests + CI
- [ ] `tests/test_world_model.py` — Protocol conformance tests
- [ ] GitHub Actions running tests + a 100-step dream rollout sanity check

### Milestone 3 — Server fork
- [ ] Fork MetaDrive-Arena server code
- [ ] Add `world_models` table + migration
- [ ] Add `wm_verifier.py` with held-out reference dataset
- [ ] Add `/api/wm/rollout` showcase endpoint

### Milestone 4 — Frontend
- [ ] Dual leaderboard (real + WM)
- [ ] Dream-vs-real replay viewer
- [ ] Per-student detail page

### Milestone 5 — Deployment
- [ ] Docker container for eval workers
- [ ] Deploy to whatever cluster we use for the RL arena
- [ ] Dry run with 5 internal agents
- [ ] Student-facing docs + release

---

## 16. Open Questions

1. **Observation modality.** Lidar-vector observations keep the WM small and fast, but "DIAMOND on pixels" is what the paper is famous for. Do we add a pixel-mode track once the vector track is working? (Proposal: yes, as a stretch assignment, using MetaDrive's `image_observation=True` at 64×64.)

2. **Off-policy dream data.** Do we let students re-collect data with their trained policy (DreamerV3-style) or keep it purely offline (single data collection pass)? (Proposal: start offline for simplicity, allow iterative collection as an advanced option.)

3. **Compute budget.** Should the server cap total GPU time per submission to make this fair? (Proposal: yes, 10 minutes of H100 wall-clock per submission for the WM verification step.)

4. **Collaboration policy.** Students will share tricks. Do we want that? (Proposal: yes, publicly celebrate shared tricks — this is a class, not Kaggle.)

---

## 17. Why This Beats a Standard Colab Notebook

We already have the DIAMOND-from-scratch Colab notebook. It's a great teaching tool for a solo learner. The arena adds three things the notebook can't:

| Notebook | Arena |
|---|---|
| Run once, get a plot | Submit dozens of times, watch your ELO climb |
| Your model vs itself | Your model vs 40 classmates' models |
| No stakes | Real leaderboard, public ranking, bragging rights |
| Read about "dream-real gap" | Feel the dream-real gap when your policy crashes in a matchup |

The arena turns an abstract concept (model-based RL) into a **felt** experience. That's what the Vizuara bootcamp promises.

---

## 18. Summary

World Model Arena is a natural extension of MetaDrive-Arena that:

- **Forces** students to implement both the IRIS and DIAMOND ideas (not just read them)
- **Grounds** the lecture in a concrete, competitive, visual deployment
- **Reuses** ~80% of MetaDrive-Arena's code (server, ELO, sandboxing, frontend shell)
- **Adds** a principled WorldModel Protocol, two reference implementations, a dream RL loop, and a second leaderboard for WM quality
- **Creates** the showcase "dream-vs-real" replay feature that makes the competition viral within the class

Next step: begin Milestone 1 (`student_starter/` scaffolding + reference implementations).
