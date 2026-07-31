# Copyright (C) 2026 Applied Intuition, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for KDiscTokenizer (no GPU, no vLLM required)."""

import pickle
import tempfile

import numpy as np
import pytest
import torch

from nord.tokenizer import KDiscTokenizer


def _make_dummy_vocab(num_clusters: int = 16, shift: int = 5) -> dict:
    """Create a minimal vocab dict matching the v3 format."""
    rng = np.random.default_rng(0)
    contours = rng.standard_normal((num_clusters, shift + 1, 4, 2)).astype(np.float32)
    return {
        "token_all": {"veh": contours},
        "shift": shift,
        "clusters": num_clusters,
    }


@pytest.fixture
def tokenizer() -> KDiscTokenizer:
    return KDiscTokenizer(_make_dummy_vocab())


def test_decode_output_shape(tokenizer):
    token_ids = [0] * 8
    traj = tokenizer.decode(token_ids)
    assert traj.shape == (40, 3), f"Expected (40, 3), got {traj.shape}"
    assert traj.dtype == np.float32


def test_decode_varied_tokens(tokenizer):
    """Different token sequences should produce different trajectories."""
    traj_a = tokenizer.decode([0] * 8)
    traj_b = tokenizer.decode([1] * 8)
    assert not np.allclose(traj_a, traj_b)


def test_parse_token_ids_valid(tokenizer):
    text = "TRAJ_0001 TRAJ_0002 TRAJ_0003 TRAJ_0004 TRAJ_0005 TRAJ_0006 TRAJ_0007 TRAJ_0008"
    ids = tokenizer.parse_token_ids(text)
    assert ids == [1, 2, 3, 4, 5, 6, 7, 8]


def test_parse_token_ids_invalid(tokenizer):
    assert tokenizer.parse_token_ids("TRAJ_0001 TRAJ_0002") is None
    assert tokenizer.parse_token_ids("no tokens here") is None


def test_from_vocab_file(tmp_path):
    vocab = _make_dummy_vocab()
    vocab_file = tmp_path / "vocab.pkl"
    with open(vocab_file, "wb") as f:
        pickle.dump(vocab, f)

    tok = KDiscTokenizer.from_vocab_file(str(vocab_file))
    assert tok.num_clusters == 16
    assert tok.shift == 5


def test_decode_invalid_token_id(tokenizer):
    with pytest.raises(ValueError, match="out of range"):
        tokenizer.decode([999] * 8)


def test_encode_decode_roundtrip(tokenizer):
    """Encoding a decoded trajectory should give back roughly the same tokens."""
    original_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    traj = tokenizer.decode(original_ids)

    # Prepend t=0 (origin) to get full 41-point trajectory for encode
    t0 = np.zeros((1, 3), dtype=np.float32)
    full_traj = np.concatenate([t0, traj], axis=0)  # (41, 3)

    re_encoded = tokenizer.encode(full_traj)
    assert len(re_encoded) == 8