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

"""
NordNavsimAgent — NoRD inference agent for the NAVSIM benchmark.

Implements NAVSIM's AbstractAgent interface so it can be dropped directly
into the NAVSIM evaluation pipeline (run_pdm_score.py or eval_navsim.py).

Requirements
------------
- NAVSIM installed: ``pip install -e .`` in the navsim repo
- vLLM server running: ``vllm serve <model_dir> --served-model-name qwen --port 8000``
"""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING, Optional

import numpy as np
import requests
from PIL import Image

from nord.tokenizer import GUIDED_REGEX, SYSTEM_PROMPT, format_past_tokens, KDiscTokenizer

try:
    from navsim.agents.abstract_agent import AbstractAgent
    from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    _NAVSIM_AVAILABLE = True
except ImportError:
    _NAVSIM_AVAILABLE = False
    # Provide stub base class so the module is importable without NAVSIM
    class AbstractAgent:  # type: ignore[no-redef]
        requires_scene = False
        def __init__(self, *a, **kw): pass

if TYPE_CHECKING:
    from navsim.common.dataclasses import AgentInput, Trajectory


_DRIVING_COMMANDS = {0: "left", 1: "straight", 2: "right"}


class NordNavsimAgent(AbstractAgent):
    """
    NoRD inference agent for the NAVSIM benchmark.

    Wraps a vLLM-served Qwen2.5-VL model; translates NAVSIM's ``AgentInput``
    to the NoRD prompt format and returns a NAVSIM ``Trajectory``.

    Parameters
    ----------
    vocab_path
        Path to the K-Disc v3 vocab ``.pkl`` file
        (``tokenized_trajectories_shift5_clusters2048_interpolated.pkl``).
    server_url
        Base URL of the running vLLM OpenAI-compatible server.
    temperature
        Sampling temperature. Paper evaluation uses ``1.0``.
    use_guided_decoding
        Constrain model output to ``TRAJ_XXXX``-only format via guided regex.
    seed
        RNG seed passed to the vLLM API.
    """

    requires_scene: bool = False

    def __init__(
        self,
        vocab_path: str | None = None,
        server_url: str = "http://localhost:8000/v1",
        temperature: float = 1.0,
        use_guided_decoding: bool = True,
        seed: int = 42,
    ) -> None:
        if not _NAVSIM_AVAILABLE:
            raise ImportError(
                "NAVSIM is required to use NordNavsimAgent. "
                "Install it with: pip install -e <path-to-navsim-repo>"
            )
        from nord.hub import BUNDLED_VOCAB
        if vocab_path is None:
            vocab_path = str(BUNDLED_VOCAB)
        trajectory_sampling = TrajectorySampling(time_horizon=4, interval_length=0.1)
        import inspect
        _parent_sig = inspect.signature(AbstractAgent.__init__)
        if "trajectory_sampling" in _parent_sig.parameters:
            super().__init__(trajectory_sampling, requires_scene=False)  # NAVSIM v2
        else:
            super().__init__(requires_scene=False)                       # NAVSIM v1
        self._trajectory_sampling = trajectory_sampling
        self._tokenizer = KDiscTokenizer.from_vocab_file(vocab_path)
        self._server_url = server_url.rstrip("/")
        self._temperature = temperature
        self._use_guided_decoding = use_guided_decoding
        self._seed = seed

    # ------------------------------------------------------------------
    # AbstractAgent interface
    # ------------------------------------------------------------------

    def name(self) -> str:
        return "NordNavsimAgent"

    def initialize(self) -> None:
        pass

    def get_sensor_config(self) -> "SensorConfig":
        return SensorConfig(
            cam_f0=[3],
            cam_l0=[3],
            cam_l1=False,
            cam_l2=False,
            cam_r0=[3],
            cam_r1=False,
            cam_r2=False,
            cam_b0=False,
            lidar_pc=False,
        )

    def compute_trajectory(self, agent_input: "AgentInput", scene=None) -> "Trajectory":
        """
        Predict a 4-second, 10 Hz trajectory from a NAVSIM scene.

        Parameters
        ----------
        agent_input
            NAVSIM ``AgentInput`` with ego history and camera images.
        scene
            Unused (accepted for API compatibility).

        Returns
        -------
        Trajectory
            40-pose trajectory at 10 Hz in the ego frame.
        """
        try:
            return self._compute(agent_input)
        except Exception as e:
            print(f"[NordNavsimAgent] Error: {e} — returning stationary trajectory")
            return self._stationary_trajectory()

    # ------------------------------------------------------------------
    # Internal logic
    # ------------------------------------------------------------------

    def _compute(self, agent_input: "AgentInput") -> "Trajectory":
        current_ego = agent_input.ego_statuses[-1]
        current_cameras = agent_input.cameras[-1]

        # Driving command
        cmd = current_ego.driving_command
        if hasattr(cmd, "__len__") and len(cmd) > 1:
            cmd_idx = int(np.argmax(cmd))
        else:
            cmd_idx = int(cmd) if cmd is not None else 1
        driving_command = _DRIVING_COMMANDS.get(cmd_idx, "straight")

        # Current velocity and acceleration
        vel = current_ego.ego_velocity[:2].tolist()
        acc = current_ego.ego_acceleration[:2].tolist()

        # Past trajectory tokens (3 tokens = last 1.5 s)
        past_token_ids = self._encode_past_trajectory(agent_input)

        # Camera images: [front_left, front, front_right]
        images_b64 = [
            self._numpy_to_b64(current_cameras.cam_l0.image),
            self._numpy_to_b64(current_cameras.cam_f0.image),
            self._numpy_to_b64(current_cameras.cam_r0.image),
        ]

        # Build and send vLLM request
        messages = self._build_messages(images_b64, vel, acc, driving_command, past_token_ids)
        generated_text = self._call_vllm(messages)

        # Decode tokens → trajectory
        token_ids = self._tokenizer.parse_token_ids(generated_text)
        if token_ids is None:
            import re
            matches = re.findall(r"TRAJ_(\d{4})", generated_text)
            token_ids = [int(m) for m in matches[:8]] + [0] * max(0, 8 - len(matches))

        poses = self._tokenizer.decode(token_ids)
        return Trajectory(poses=poses, trajectory_sampling=self._trajectory_sampling)

    def _encode_past_trajectory(self, agent_input: "AgentInput") -> Optional[list]:
        """Encode past ego poses (2 Hz) to 3 K-Disc token IDs."""
        statuses = agent_input.ego_statuses
        if len(statuses) < 2:
            return None

        # Collect 2 Hz poses
        poses_2hz = np.array([s.ego_pose[:3] for s in statuses], dtype=np.float32)

        # Transform to local frame of the first pose
        origin = poses_2hz[0]
        cos_h, sin_h = np.cos(-origin[2]), np.sin(-origin[2])
        relative = []
        for p in poses_2hz:
            dx, dy = p[0] - origin[0], p[1] - origin[1]
            x_local = dx * cos_h - dy * sin_h
            y_local = dx * sin_h + dy * cos_h
            heading_local = np.arctan2(np.sin(p[2] - origin[2]), np.cos(p[2] - origin[2]))
            relative.append([x_local, y_local, heading_local])

        # Interpolate 2 Hz → 10 Hz
        traj_10hz = self._interpolate_to_10hz(np.array(relative, dtype=np.float32))
        try:
            return self._tokenizer.encode(traj_10hz) or None
        except Exception:
            return None

    @staticmethod
    def _interpolate_to_10hz(traj_2hz: np.ndarray) -> np.ndarray:
        """Linear interpolation from 2 Hz to 10 Hz (5× upsampling)."""
        if len(traj_2hz) < 2:
            return traj_2hz
        t_2hz = np.arange(len(traj_2hz)) * 0.5
        t_10hz = np.arange(0, t_2hz[-1] + 1e-6, 0.1)
        x = np.interp(t_10hz, t_2hz, traj_2hz[:, 0])
        y = np.interp(t_10hz, t_2hz, traj_2hz[:, 1])
        h_interp = np.interp(t_10hz, t_2hz, np.unwrap(traj_2hz[:, 2]))
        h = np.arctan2(np.sin(h_interp), np.cos(h_interp))
        return np.stack([x, y, h], axis=1).astype(np.float32)

    def _build_messages(
        self,
        images_b64: list,
        vel: list,
        acc: list,
        driving_command: str,
        past_token_ids: Optional[list],
    ) -> list:
        past_str = self._format_past_tokens(past_token_ids)
        user_text = (
            f"Past 1.5 seconds trajectory: {past_str}\n"
            f"Current [x, y] velocity: [{vel[0]:.3f}, {vel[1]:.3f}] m/s\n"
            f"Current [x, y] acceleration: [{acc[0]:.3f}, {acc[1]:.3f}] m/s^2\n"
            f"Driving command: {driving_command}"
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{images_b64[0]}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{images_b64[1]}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{images_b64[2]}"}},
                    {"type": "text", "text": user_text},
                ],
            },
        ]

    def _call_vllm(self, messages: list[dict]) -> str:
        payload: dict = {
            "model": "qwen",
            "messages": messages,
            "temperature": self._temperature,
            "repetition_penalty": 1.0,
            "seed": self._seed,
        }
        if self._use_guided_decoding:
            payload["guided_regex"] = GUIDED_REGEX

        resp = requests.post(
            f"{self._server_url}/chat/completions",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    @staticmethod
    def _numpy_to_b64(image: np.ndarray) -> str:
        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 255) if image.max() > 1.0 else image * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(image).convert("RGB").save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def _format_past_tokens(token_ids: list[int] | None) -> str:
        return format_past_tokens(token_ids)

    def _stationary_trajectory(self) -> "Trajectory":
        poses = np.zeros((self._trajectory_sampling.num_poses, 3), dtype=np.float32)
        return Trajectory(poses=poses, trajectory_sampling=self._trajectory_sampling)