"""Stable-Baselines3 PPO `.zip` loader with cross-version compatibility.

Some packaged SB3 checkpoints were trained/saved under a different NumPy version.
SB3 stores various Python objects (including Gym spaces and RNG state) as
cloudpickled blobs inside the `.zip`. Those blobs may reference internal NumPy
modules (`numpy._core.*`) or BitGenerator constructors that changed between
NumPy 1.x and 2.x.

This module provides:
- `infer_ppo_model_spaces`: infer observation/action spaces directly from the
  archive (without executing the cloudpickled Gym spaces).
- `load_ppo_model`: load the PPO model with small NumPy shims + inferred spaces.
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces

_NUMPY_PICKLE_PATCHED = False


def _patch_numpy_cloudpickle_compat() -> None:
    """Patch NumPy internals referenced by some cloudpickled SB3 checkpoints."""

    global _NUMPY_PICKLE_PATCHED
    if _NUMPY_PICKLE_PATCHED:
        return

    import numpy.core as _core
    import numpy.core.numeric as _numeric
    import numpy.random._pickle as _np_pickle

    sys.modules.setdefault("numpy._core", _core)
    sys.modules.setdefault("numpy._core.numeric", _numeric)

    original_ctor = _np_pickle.__bit_generator_ctor

    def _patched_ctor(bit_generator_name):
        if isinstance(bit_generator_name, type):
            bit_generator_name = bit_generator_name.__name__
        return original_ctor(bit_generator_name)

    _np_pickle.__bit_generator_ctor = _patched_ctor

    _NUMPY_PICKLE_PATCHED = True


def read_sb3_zip_metadata(model_path: Path) -> dict[str, Any]:
    """Read the SB3 `data` JSON payload stored in a model `.zip` archive."""
    with zipfile.ZipFile(model_path, "r") as zf:
        raw = zf.read("data").decode("utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("SB3 model metadata must be a JSON object.")
    return data


def _read_policy_state_dict(model_path: Path) -> dict[str, Any]:
    import torch

    with zipfile.ZipFile(model_path, "r") as zf:
        payload = zf.read("policy.pth")
    state = torch.load(io.BytesIO(payload), map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError("policy.pth must contain a state_dict mapping.")
    return state


@dataclass(frozen=True)
class PpoModelSpaces:
    observation_space: spaces.Box
    action_space: spaces.Box
    window_size: int
    obs_dim: int
    action_dim: int
    meta: dict[str, Any]


def infer_ppo_model_spaces(model_path: Path) -> PpoModelSpaces:
    """Infer (observation_space, action_space) for an SB3 PPO model `.zip`."""
    meta = read_sb3_zip_metadata(model_path)
    state = _read_policy_state_dict(model_path)

    try:
        action_dim = int(state["action_net.bias"].numel())
    except Exception as exc:  # pragma: no cover - indicates corrupted checkpoint
        raise ValueError(f"Unable to infer action_dim from policy state: {exc}") from exc

    pk = (meta.get("policy_kwargs") or {}).get("features_extractor_kwargs") or {}
    market_dim = int(pk.get("market_dim") or 0)
    aux_dim = int(pk.get("auxiliary_dim") or 0)

    window_size: int
    obs_dim: int

    # CrossAttentionFeatureExtractor: indicator_norm is LayerNorm(seq_len).
    if "features_extractor.indicator_norm.weight" in state and market_dim and aux_dim:
        window_size = int(state["features_extractor.indicator_norm.weight"].numel())
        obs_dim = int(market_dim + aux_dim)
    # TransformerFeatureExtractor: input_proj maps num_features -> d_model.
    elif "features_extractor.input_proj.weight" in state:
        obs_dim = int(state["features_extractor.input_proj.weight"].shape[1])
        if "features_extractor.pos_embedding" in state:
            total_tokens = int(state["features_extractor.pos_embedding"].shape[1])
            window_size = (
                total_tokens - 1 if "features_extractor.cls_token" in state else total_tokens
            )
        else:  # pragma: no cover - unexpected for current repo policies
            raise ValueError("Unable to infer window_size for TransformerFeatureExtractor.")
    # CustomCNN (MultiScale-CNN): first conv in_channels == num_features; linear in_features == 64*window.
    elif (
        "features_extractor.short_conv.0.weight" in state
        and "features_extractor.linear.0.weight" in state
    ):
        obs_dim = int(state["features_extractor.short_conv.0.weight"].shape[1])
        linear_in = int(state["features_extractor.linear.0.weight"].shape[1])
        if linear_in % 64 != 0:
            raise ValueError(
                f"Unexpected CustomCNN linear_in={linear_in}; expected multiple of 64."
            )
        window_size = int(linear_in // 64)
    else:  # pragma: no cover - indicates unknown policy/checkpoint format
        raise ValueError(f"Unsupported/unknown PPO model format: {model_path.name}")

    observation_space = spaces.Box(
        low=-np.inf, high=np.inf, shape=(window_size, obs_dim), dtype=np.float32
    )
    action_space = spaces.Box(low=0.0, high=1.0, shape=(action_dim,), dtype=np.float32)
    return PpoModelSpaces(
        observation_space=observation_space,
        action_space=action_space,
        window_size=window_size,
        obs_dim=obs_dim,
        action_dim=action_dim,
        meta=meta,
    )


def load_ppo_model(model_path: Path, *, device: str = "cpu"):
    """Load an SB3 PPO model with compatibility patches + inferred spaces."""
    # Import SB3 first; patching NumPy aliases prior to torch import can crash on Windows.
    from stable_baselines3 import PPO

    _patch_numpy_cloudpickle_compat()

    spaces_info = infer_ppo_model_spaces(model_path)
    model = PPO.load(
        str(model_path),
        device=device,
        custom_objects={
            "observation_space": spaces_info.observation_space,
            "action_space": spaces_info.action_space,
        },
    )
    return model
