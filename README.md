# World Model Arena

**Train a world model. Dream a policy. Race it for real.**

Build an IRIS or DIAMOND world model from driving data, train a policy entirely inside your model's imagination, then deploy it on MetaDrive and compete on a live leaderboard against your classmates.

[View Leaderboard](https://world-model-arena.vercel.app) · [Vizuara](https://vizuara.ai)

---

## Quick Start

```bash
# 1. Fork this repo and clone it
git clone https://github.com/RajatDandekar/world-model-arena.git
cd world-model-arena/student_starter

# 2. Install dependencies
conda create -n wma python=3.11 -y && conda activate wma
pip install -r ../requirements.txt

# 3. Collect driving data (~5 min)
python data_collect.py --n-episodes 200

# 4. Train a world model (IRIS or DIAMOND)
python train_wm.py --model diamond --steps 5000

# 5. Train a policy inside the world model's dreams
python dream_train_policy.py --wm checkpoints/wm.pt --wm-type diamond --iterations 500

# 6. Evaluate on real MetaDrive
python eval_policy_real.py --policy checkpoints/policy.pt

# 7. Submit to the leaderboard!
python submit.py --tag my-first-try --wm-type diamond \
    --wm checkpoints/wm.pt --name "Your Name" --uid your-id
```

If steps 3-7 finish cleanly, you're on the leaderboard. Now the real work begins — make it better.

---

## Project Structure

```
world-model-arena/
├── README.md                     ← you are here
├── requirements.txt              ← pip install -r requirements.txt
├── leaderboard/
│   └── index.html                ← live leaderboard (hosted on Vercel)
└── student_starter/
    ├── README.md                 ← detailed assignment spec + grading rubric
    ├── world_model.py            ← the Protocol interface — read this first
    ├── data_collect.py           ← Step 1: collect 200 driving episodes
    ├── train_wm.py               ← Step 2: train IRIS or DIAMOND
    ├── dream_train_policy.py     ← Step 3: RL inside the world model
    ├── eval_policy_real.py       ← Step 4: deploy on real MetaDrive
    ├── submit.py                 ← Step 5: evaluate + upload to leaderboard
    ├── wm_iris/iris.py           ← Reference mini-IRIS (~3M params)
    ├── wm_diamond/diamond.py     ← Reference mini-DIAMOND (~2.5M params)
    ├── agents/                   ← Pre-trained instructor baselines
    ├── checkpoints/              ← Saved model weights
    ├── data/held_out/            ← 50 reference episodes for WM evaluation
    └── videos/                   ← Recorded driving videos
```

---

## How Scoring Works

### Race Leaderboard (competitive)
Your policy runs on **5 maps × 4 episodes** on your local MetaDrive. Route completion is the primary metric. ELO is computed from head-to-head comparisons across all maps.

### World Model Quality (scientific)
If you include `--wm` in your submission, your world model is evaluated on held-out data:
- **1-step MSE** — single-step prediction error
- **15-step MSE** — cumulative drift over 15 steps
- **Dream-Real Gap** — `|dream_score - real_return| / max(|real_return|, 1)` — did your WM learn real physics?

A small dream-real gap (<15%) means your world model actually learned the simulator dynamics. A large gap means your policy is exploiting hallucinations in its own dream.

---

## The Two Reference World Models

| | IRIS | DIAMOND |
|---|---|---|
| **Idea** | Discrete tokens + GPT | Continuous obs + diffusion denoiser |
| **Encoder** | VQ-VAE (512 codes) | None — obs is the latent |
| **Dynamics** | Causal Transformer | MLP ResNet with AdaGN |
| **Training** | Next-token CE + VQ loss | EDM-weighted denoising MSE |
| **Paper** | [IRIS, ICLR 2023](https://arxiv.org/abs/2209.00588) | [DIAMOND, NeurIPS 2024](https://arxiv.org/abs/2405.12399) |

Both are deliberately small (~2-3M params) so they train in under 10 minutes on a T4.

---

## Tips for Beating the Baselines

1. **Collect more diverse data** — 1000 episodes, including crashes
2. **Scale the world model** — increase hidden dims or add layers
3. **Tune the noise schedule** — DIAMOND's `sigma_offset` matters
4. **Dream for longer** — 25-30 step rollouts instead of 15
5. **Train the policy longer** — 2000 dream iterations instead of 500
6. **Try a hybrid** — VQ-VAE encoder + diffusion dynamics

See [`student_starter/README.md`](student_starter/README.md) for the full assignment spec and grading rubric.

---

Built by [Vizuara](https://vizuara.ai)
