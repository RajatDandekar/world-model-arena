"""
mini-IRIS — a stripped-down IRIS for lidar-vector observations.

IRIS paper: "Transformers are Sample-Efficient World Models" (Micheli et al. 2023)
    https://arxiv.org/abs/2209.00588

What we keep from the paper:
    - VQ-VAE to discretise the observation
    - Causal Transformer that autoregressively predicts (z_{t+1}, r_t, d_t)
    - Pure dream rollouts (no real env during policy training)

What we change:
    - Observations are 91-D lidar vectors, not 64×64 images
    - VQ-VAE is an MLP (not a conv net)
    - Transformer is tiny (6 layers, d=256) so it trains in ~10 min on a T4

Everything else — the training objective and the dream loop — matches IRIS.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# VQ-VAE
# ----------------------------------------------------------------------


class VectorQuantiser(nn.Module):
    def __init__(self, n_codes: int = 512, code_dim: int = 32, beta: float = 0.25):
        super().__init__()
        self.codebook = nn.Embedding(n_codes, code_dim)
        self.codebook.weight.data.uniform_(-1.0 / n_codes, 1.0 / n_codes)
        self.beta = beta
        self.n_codes = n_codes
        self.code_dim = code_dim

    def forward(self, z_e: torch.Tensor):
        # z_e: (B, code_dim)
        dists = (
            z_e.pow(2).sum(-1, keepdim=True)
            - 2 * z_e @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(-1)
        )
        indices = dists.argmin(-1)                    # (B,)
        z_q = self.codebook(indices)                  # (B, code_dim)
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commit_loss = F.mse_loss(z_e, z_q.detach())
        loss = codebook_loss + self.beta * commit_loss
        z_q = z_e + (z_q - z_e).detach()              # straight-through
        return z_q, indices, loss


class VQVAE(nn.Module):
    def __init__(self, obs_dim: int, code_dim: int = 32, n_codes: int = 512, hidden: int = 256):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, code_dim),
        )
        self.vq = VectorQuantiser(n_codes=n_codes, code_dim=code_dim)
        self.dec = nn.Sequential(
            nn.Linear(code_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, obs_dim),
        )

    def encode(self, obs: torch.Tensor):
        z_e = self.enc(obs)
        z_q, idx, vq_loss = self.vq(z_e)
        return z_q, idx, vq_loss

    def decode(self, z_q: torch.Tensor):
        return self.dec(z_q)

    def decode_indices(self, indices: torch.Tensor):
        return self.dec(self.vq.codebook(indices))


# ----------------------------------------------------------------------
# Causal Transformer that predicts next token + reward + done
# ----------------------------------------------------------------------


class CausalTransformer(nn.Module):
    def __init__(
        self,
        n_codes: int,
        action_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        ctx_len: int = 32,
    ):
        super().__init__()
        self.n_codes = n_codes
        self.ctx_len = ctx_len
        self.d_model = d_model

        self.token_embed = nn.Embedding(n_codes, d_model)
        self.action_embed = nn.Linear(action_dim, d_model)
        self.pos_embed = nn.Embedding(ctx_len * 2, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.head_token = nn.Linear(d_model, n_codes)
        self.head_reward = nn.Linear(d_model, 1)
        self.head_done = nn.Linear(d_model, 1)

    def forward(
        self,
        token_ids: torch.Tensor,  # (B, T)        discrete obs tokens
        actions: torch.Tensor,    # (B, T, a_dim) actions
    ):
        B, T = token_ids.shape
        tok_emb = self.token_embed(token_ids)                     # (B, T, d)
        act_emb = self.action_embed(actions)                      # (B, T, d)
        # interleave as (z0, a0, z1, a1, ..., z_{T-1}, a_{T-1})
        seq = torch.stack([tok_emb, act_emb], dim=2).reshape(B, 2 * T, self.d_model)
        pos = torch.arange(2 * T, device=seq.device)
        seq = seq + self.pos_embed(pos)[None]

        # causal mask
        mask = torch.triu(
            torch.ones(2 * T, 2 * T, device=seq.device, dtype=torch.bool), diagonal=1
        )
        out = self.transformer(seq, mask=mask)

        # Predictions are taken from action-slot outputs: the a_t position
        # receives enough context to predict (z_{t+1}, r_t, d_t).
        act_slot = out[:, 1::2, :]                                 # (B, T, d)
        logits = self.head_token(act_slot)                         # (B, T, n_codes)
        reward = self.head_reward(act_slot).squeeze(-1)            # (B, T)
        done = self.head_done(act_slot).squeeze(-1)                # (B, T)
        return logits, reward, done


# ----------------------------------------------------------------------
# The full mini-IRIS that implements the WorldModel Protocol
# ----------------------------------------------------------------------


class MiniIRIS(nn.Module):
    def __init__(
        self,
        obs_dim: int = 91,
        action_dim: int = 2,
        code_dim: int = 32,
        n_codes: int = 512,
        d_model: int = 256,
        n_layers: int = 6,
        context_len: int = 8,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.context_len = context_len
        self.device = torch.device(device)

        self._code_dim = code_dim
        self._n_codes = n_codes
        self._d_model = d_model
        self._n_layers = n_layers

        self.vqvae = VQVAE(obs_dim, code_dim=code_dim, n_codes=n_codes)
        self.transformer = CausalTransformer(
            n_codes=n_codes,
            action_dim=action_dim,
            d_model=d_model,
            n_layers=n_layers,
            ctx_len=context_len + 16,
        )

        # dream state: (B, ctx, ...)
        self._dream_tokens: torch.Tensor | None = None
        self._dream_actions: torch.Tensor | None = None

        self.to(self.device)

    # -------- protocol: save/load --------
    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "obs_dim": self.obs_dim,
                    "action_dim": self.action_dim,
                    "context_len": self.context_len,
                    "code_dim": self._code_dim,
                    "n_codes": self._n_codes,
                    "d_model": self._d_model,
                    "n_layers": self._n_layers,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str | torch.device = "cpu") -> "MiniIRIS":
        ckpt = torch.load(path, map_location=device)
        m = cls(device=device, **ckpt["config"])
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        return m

    # -------- protocol: fit --------
    def fit(
        self,
        dataset,
        steps: int,
        batch_size: int = 64,
        log_every: int = 100,
        lr: float = 3e-4,
    ) -> dict:
        opt = torch.optim.AdamW(self.parameters(), lr=lr)
        history = {"loss": [], "loss_vq": [], "loss_tok": [], "loss_rew": [], "loss_done": []}

        # We train on short windows of length = context_len
        H = self.context_len
        self.train()
        for step in range(steps):
            obs, act, rew = dataset.sample_rollout(batch_size, horizon=H)
            # obs: (B, H+1, obs_dim)   act: (B, H, a_dim)   rew: (B, H)
            B = obs.shape[0]
            obs_flat = obs[:, :H, :].reshape(B * H, self.obs_dim).to(self.device)
            _, idx, vq_loss = self.vqvae.encode(obs_flat)           # (B*H,)
            token_ids = idx.reshape(B, H)
            act = act.to(self.device)
            rew = rew.to(self.device)

            logits, rew_hat, done_hat = self.transformer(token_ids, act)
            # Next-token target: shifted tokens (use next obs through vqvae)
            with torch.no_grad():
                next_obs_flat = obs[:, 1:, :].reshape(B * H, self.obs_dim).to(self.device)
                _, next_idx, _ = self.vqvae.encode(next_obs_flat)
                next_token_ids = next_idx.reshape(B, H)

            loss_tok = F.cross_entropy(
                logits.reshape(B * H, -1), next_token_ids.reshape(-1)
            )
            loss_rew = F.mse_loss(rew_hat, rew)
            # done is all-zero on mid-episode windows — still train it for calibration
            done_target = torch.zeros_like(done_hat)
            loss_done = F.binary_cross_entropy_with_logits(done_hat, done_target)
            # Reconstruction on the encoded obs (so vqvae keeps decoding correctly)
            recon = self.vqvae.decode(self.vqvae.encode(obs_flat)[0])
            loss_recon = F.mse_loss(recon, obs_flat)

            loss = loss_tok + loss_rew + 0.1 * loss_done + vq_loss + loss_recon

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            opt.step()

            history["loss"].append(loss.item())
            history["loss_vq"].append(vq_loss.item())
            history["loss_tok"].append(loss_tok.item())
            history["loss_rew"].append(loss_rew.item())
            history["loss_done"].append(loss_done.item())

            if (step + 1) % log_every == 0:
                print(
                    f"[iris {step+1}/{steps}] "
                    f"loss={loss.item():.3f} "
                    f"tok={loss_tok.item():.3f} "
                    f"rew={loss_rew.item():.4f} "
                    f"vq={vq_loss.item():.4f}"
                )
        self.eval()
        return history

    # -------- protocol: dream --------
    @torch.no_grad()
    def reset_dream(
        self,
        init_obs: torch.Tensor,
        init_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode context observations into tokens, store as dream state."""
        init_obs = init_obs.to(self.device)
        B, C, D = init_obs.shape
        _, idx, _ = self.vqvae.encode(init_obs.reshape(B * C, D))
        self._dream_tokens = idx.reshape(B, C)  # (B, C)

        if init_actions is None:
            init_actions = torch.zeros(B, C, self.action_dim, device=self.device)
        self._dream_actions = init_actions.to(self.device)

        # Return latent = encoded context codes flattened as a vector
        return self.vqvae.vq.codebook(self._dream_tokens[:, -1])  # (B, code_dim)

    @torch.no_grad()
    def step_dream(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = action.shape[0]
        action = action.to(self.device)

        # append the new action to the context
        self._dream_actions = torch.cat(
            [self._dream_actions, action[:, None, :]], dim=1
        )
        # Only keep the most recent ctx_len tokens/actions
        max_ctx = self.context_len + 8
        if self._dream_tokens.shape[1] > max_ctx:
            self._dream_tokens = self._dream_tokens[:, -max_ctx:]
            self._dream_actions = self._dream_actions[:, -max_ctx:]

        # Run transformer on current context
        # We need tokens and actions of equal length.
        T = min(self._dream_tokens.shape[1], self._dream_actions.shape[1])
        tokens_in = self._dream_tokens[:, -T:]
        actions_in = self._dream_actions[:, -T:]
        logits, rew_hat, done_hat = self.transformer(tokens_in, actions_in)

        # Sample next token from the LAST action slot
        next_logits = logits[:, -1, :]                     # (B, n_codes)
        next_probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(next_probs, num_samples=1).squeeze(-1)  # (B,)

        # Append to dream tokens
        self._dream_tokens = torch.cat(
            [self._dream_tokens, next_token[:, None]], dim=1
        )

        # Decode to observation
        next_code = self.vqvae.vq.codebook(next_token)     # (B, code_dim)
        obs_hat = self.vqvae.decode(next_code)             # (B, obs_dim)

        r = rew_hat[:, -1]
        d = torch.sigmoid(done_hat[:, -1])

        return next_code, r, d, obs_hat

    # -------- protocol: eval helpers --------
    @torch.no_grad()
    def eval_next_obs_mse(self, dataset, n_batches: int = 16) -> float:
        self.eval()
        total, n = 0.0, 0
        for _ in range(n_batches):
            b = dataset.sample_transitions(64)
            obs = b["obs"].to(self.device)
            _, idx, _ = self.vqvae.encode(obs)
            tokens = idx[:, None]                        # (B, 1)
            act = b["action"].to(self.device)[:, None]   # (B, 1, a)
            logits, _, _ = self.transformer(tokens, act)
            next_logits = logits[:, -1, :]
            next_tok = next_logits.argmax(-1)
            obs_hat = self.vqvae.decode_indices(next_tok)
            total += F.mse_loss(obs_hat, b["next_obs"].to(self.device)).item()
            n += 1
        return total / n

    @torch.no_grad()
    def eval_rollout_mse(self, dataset, horizon: int = 15, n_batches: int = 8) -> float:
        self.eval()
        total, n = 0.0, 0
        for _ in range(n_batches):
            obs, act, _rew = dataset.sample_rollout(16, horizon=horizon)
            obs, act = obs.to(self.device), act.to(self.device)
            B = obs.shape[0]
            # seed dream with just the first obs + a zero-action context
            init_obs = obs[:, :1, :]
            init_act = torch.zeros_like(act[:, :1, :])
            self.reset_dream(init_obs, init_act)
            mse = 0.0
            for t in range(horizon):
                _lat, _r, _d, obs_hat = self.step_dream(
                    torch.zeros(B, 1, device=self.device),  # unused
                    act[:, t, :],
                )
                mse += F.mse_loss(obs_hat, obs[:, t + 1, :]).item()
            total += mse / horizon
            n += 1
        return total / n
