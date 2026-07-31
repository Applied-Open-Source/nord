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

"""Tests for NordAgent (no vLLM server required — uses mocked HTTP)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from nord.agent import NordAgent, NordInput, NordOutput
from nord.tokenizer import KDiscTokenizer

from tests.test_tokenizer import _make_dummy_vocab


def _make_agent() -> NordAgent:
    tokenizer = KDiscTokenizer(_make_dummy_vocab(num_clusters=10000))
    return NordAgent(tokenizer=tokenizer, use_guided_decoding=False)


def _dummy_images() -> list[Image.Image]:
    return [Image.fromarray(np.zeros((128, 256, 3), dtype=np.uint8)) for _ in range(3)]


def test_build_messages_structure():
    agent = _make_agent()
    scene = NordInput(
        cameras=_dummy_images(),
        ego_velocity_ms=(5.0, 0.0),
        ego_acceleration_ms2=(0.1, 0.0),
        driving_command="straight",
        past_token_ids=[10, 20, 30],
    )
    messages = agent._build_messages(scene)

    assert messages[0]["role"] == "system"
    user_content = messages[1]["content"]
    image_parts = [p for p in user_content if p["type"] == "image_url"]
    text_parts = [p for p in user_content if p["type"] == "text"]

    assert len(image_parts) == 3
    assert len(text_parts) == 1
    assert "TRAJ_0010 TRAJ_0020 TRAJ_0030" in text_parts[0]["text"]
    assert "straight" in text_parts[0]["text"]


def test_build_messages_default_past_tokens():
    agent = _make_agent()
    scene = NordInput(cameras=_dummy_images())
    messages = agent._build_messages(scene)
    text = messages[1]["content"][-1]["text"]
    assert "TRAJ_0000 TRAJ_0000 TRAJ_0000" in text


def test_predict_success():
    agent = _make_agent()
    scene = NordInput(cameras=_dummy_images())

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "TRAJ_0001 TRAJ_0002 TRAJ_0003 TRAJ_0004 TRAJ_0005 TRAJ_0006 TRAJ_0007 TRAJ_0008"}}]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_response):
        output = agent.predict(scene)

    assert isinstance(output, NordOutput)
    assert output.trajectory.shape == (40, 3)
    assert len(output.token_ids) == 8
    assert output.token_ids == [1, 2, 3, 4, 5, 6, 7, 8]


def test_predict_malformed_response_pads():
    """If model returns fewer than 8 tokens, agent pads with zeros gracefully."""
    agent = _make_agent()
    scene = NordInput(cameras=_dummy_images())

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "TRAJ_0001 TRAJ_0002"}}]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_response):
        output = agent.predict(scene)

    assert output.trajectory.shape == (40, 3)
    assert len(output.token_ids) == 8


def test_predict_server_down_raises():
    import requests as req_mod
    agent = _make_agent()
    scene = NordInput(cameras=_dummy_images())

    with patch("requests.post", side_effect=req_mod.exceptions.ConnectionError("refused")):
        with pytest.raises(RuntimeError, match="vLLM request failed"):
            agent.predict(scene)


def test_wrong_camera_count():
    agent = _make_agent()
    scene = NordInput(cameras=_dummy_images()[:1])  # only 1 image
    with pytest.raises(ValueError, match="3 camera images"):
        agent._build_messages(scene)