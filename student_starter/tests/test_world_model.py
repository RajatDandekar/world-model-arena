"""
WorldModel Protocol contract tests.

These are fast, CPU-only, synthetic-data tests that verify any world model
implementation satisfies the Protocol and can be plugged into the dream RL
loop. Students can (and should) run these before submitting.

    pytest tests/test_world_model.py -q
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch

from world_model import WorldModel
from data import TrajectoryDataset


# ------------------------------------------------------------------
# Helper: fabricate a tiny dataset so we don't need MetaDrive installed
# ------------------------------------------------------------------


def _make_fake_dataset(tmpdir: str, n_eps: int = 4, ep_len: int = 50, obs_dim: int = 91):
    for i in range(n_eps):
        obs = np.random.randn(ep_len + 1, obs_dim).astype(np.float32)
        act = np.random.uniform(-1, 1, size=(ep_len, 2)).astype(np.float32)
        rew = np.random.randn(ep_len).astype(np.float32)
        done = np.zeros(ep_len, dtype=bool)
        done[-1] = True
        np.savez_compressed(
            os.path.join(tmpdir, f"ep_{i:03d}.npz"),
            obs=obs, action=act, reward=rew, done=done,
        )
    return TrajectoryDataset(tmpdir, device="cpu")


# ------------------------------------------------------------------
# Parameterise over both reference implementations
# ------------------------------------------------------------------


def _build_iris():
    from wm_iris import MiniIRIS
    return MiniIRIS(obs_dim=91, action_dim=2, d_model=128, n_layers=2, context_len=4)


def _build_diamond():
    from wm_diamond import MiniDIAMOND
    return MiniDIAMOND(obs_dim=91, action_dim=2, hidden=128, n_blocks=2, context_len=4)


@pytest.fixture(scope="module")
def dataset():
    with tempfile.TemporaryDirectory() as tmp:
        yield _make_fake_dataset(tmp)


@pytest.mark.parametrize("builder", [_build_iris, _build_diamond], ids=["iris", "diamond"])
def test_protocol_conformance(builder):
    wm = builder()
    assert isinstance(wm, WorldModel), "must satisfy the WorldModel Protocol"
    assert wm.obs_dim == 91
    assert wm.action_dim == 2
    assert wm.context_len >= 1


@pytest.mark.parametrize("builder", [_build_iris, _build_diamond], ids=["iris", "diamond"])
def test_tiny_training_loop(builder, dataset):
    wm = builder()
    history = wm.fit(dataset, steps=5, batch_size=8, log_every=10)
    assert "loss" in history and len(history["loss"]) == 5
    assert all(np.isfinite(h) for h in history["loss"]), "loss must stay finite"


@pytest.mark.parametrize("builder", [_build_iris, _build_diamond], ids=["iris", "diamond"])
def test_dream_rollout(builder, dataset):
    wm = builder()
    wm.fit(dataset, steps=5, batch_size=8, log_every=10)

    init_obs, init_act = dataset.sample_context(batch_size=4, ctx_len=wm.context_len)
    latent = wm.reset_dream(init_obs, init_act)
    assert latent.shape[0] == 4

    for _ in range(5):
        action = torch.zeros(4, 2)
        latent, r, d, obs_hat = wm.step_dream(latent, action)
        assert obs_hat.shape == (4, 91)
        assert r.shape == (4,)
        assert d.shape == (4,)
        assert torch.isfinite(obs_hat).all()


@pytest.mark.parametrize("builder", [_build_iris, _build_diamond], ids=["iris", "diamond"])
def test_save_load_roundtrip(builder, dataset, tmp_path):
    wm = builder()
    wm.fit(dataset, steps=3, batch_size=8, log_every=10)
    path = tmp_path / "wm.pt"
    wm.save(str(path))

    # Reload from same class via .load classmethod
    wm2 = builder().__class__.load(str(path), device="cpu")
    init_obs, init_act = dataset.sample_context(batch_size=2, ctx_len=wm2.context_len)
    _ = wm2.reset_dream(init_obs, init_act)
    _lat, _r, _d, obs_hat = wm2.step_dream(
        torch.zeros(2, 1), torch.zeros(2, 2)
    )
    assert obs_hat.shape == (2, 91)
