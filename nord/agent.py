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

"""NoRD inference agent — wraps a vLLM-served Qwen2.5-VL model."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from nord.tokenizer import GUIDED_REGEX, SYSTEM_PROMPT, format_past_tokens

if TYPE_CHECKING:
    from nord.tokenizer import KDiscTokenizer


@dataclass
class NordInput:
    """
    One driving scene for inference.

    Parameters
    ----------
    cameras
        List of exactly 3 PIL Images in order: [front_left, front, front_right].
    ego_velocity_ms
        Current ego velocity ``(vx, vy)`` in m/s in the ego frame.
    ego_acceleration_ms2
        Current ego acceleration ``(ax, ay)`` in m/s² in the ego frame.
    driving_command
        High-level route command: ``"left"``, ``"straight"``, or ``"right"``.
    past_token_ids
        3 K-Disc token IDs representing the past 1.5 s of trajectory.
        If ``None``, the prompt uses ``TRAJ_0000`` placeholders.
    """

    cameras: list[Image.Image]
    ego_velocity_ms: tuple[float, float] = (0.0, 0.0)
    ego_acceleration_ms2: tuple[float, float] = (0.0, 0.0)
    driving_command: str = "straight"
    past_token_ids: list[int] | None = None


@dataclass
class NordOutput:
    """
    Predicted trajectory for one driving scene.

    Parameters
    ----------
    trajectory
        ``(40, 3)`` float32 array of ``[x, y, heading]`` waypoints at 10 Hz
        (t = 0.1 … 4.0 s) in the ego frame.
        x = forward, y = left. Units: metres / radians.
    token_ids
        The 8 raw K-Disc codebook token indices produced by the model.
    """

    trajectory: np.ndarray  # (40, 3) float32
    token_ids: list[int]    # length 8


class NordAgent:
    """
    NoRD inference agent.

    Sends a structured vision-language prompt to a vLLM-served Qwen2.5-VL
    model and decodes the 8 trajectory tokens into a 4-second, 10 Hz
    trajectory.

    The model must be running before calling ``predict()``. Start it with::

        vllm serve <model-dir> --served-model-name qwen --port 8000

    Parameters
    ----------
    tokenizer
        A loaded :class:`~nord.tokenizer.KDiscTokenizer`.
    base_url
        Base URL of the OpenAI-compatible vLLM server
        (default: ``"http://localhost:8000/v1"``).
    temperature
        Sampling temperature. The paper evaluates at ``temperature=1.0``.
    top_p
        Nucleus sampling cutoff.
    use_guided_decoding
        If True (recommended), constrains the model output to the
        ``TRAJ_XXXX``-only format via vLLM's ``guided_regex`` parameter.
    """

    _GUIDED_REGEX = GUIDED_REGEX
    _SYSTEM_PROMPT = SYSTEM_PROMPT

    def __init__(
        self,
        tokenizer: "KDiscTokenizer",
        base_url: str = "http://localhost:8000/v1",
        temperature: float = 1.0,
        top_p: float = 1.0,
        use_guided_decoding: bool = True,
    ) -> None:
        self._tokenizer = tokenizer
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._top_p = top_p
        self._use_guided_decoding = use_guided_decoding

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "nord",
        *,
        base_url: str = "http://localhost:8000/v1",
        vocab_path: str | None = None,
        cache_dir: str | None = None,
        temperature: float = 1.0,
        use_guided_decoding: bool = True,
    ) -> "NordAgent":
        """
        Create a NordAgent by loading the vocab from HuggingFace Hub (or a
        local path).

        This does **not** start the vLLM server. Start it separately before
        calling ``predict()``.

        Parameters
        ----------
        model_name
            ``"nord"`` or ``"nord-base"``.
        base_url
            vLLM server base URL.
        vocab_path
            Path to the ``.pkl`` vocab file. If ``None``, the vocab is
            downloaded from the HF Hub alongside the model.
        cache_dir
            HF Hub cache directory.
        temperature
            Sampling temperature.
        use_guided_decoding
            Enable vLLM guided decoding (strongly recommended).
        """
        from nord.tokenizer import KDiscTokenizer
        from nord.hub import BUNDLED_VOCAB

        if vocab_path is None:
            vocab_path = str(BUNDLED_VOCAB)

        tokenizer = KDiscTokenizer.from_vocab_file(vocab_path)
        return cls(
            tokenizer=tokenizer,
            base_url=base_url,
            temperature=temperature,
            use_guided_decoding=use_guided_decoding,
        )

    def predict(self, scene: NordInput) -> NordOutput:
        """
        Run one trajectory prediction against the running vLLM server.

        Parameters
        ----------
        scene
            A :class:`NordInput` with camera images and ego state.

        Returns
        -------
        NordOutput
            Predicted ``(40, 3)`` trajectory and raw token IDs.
        """
        import requests

        messages = self._build_messages(scene)
        payload: dict = {
            "model": "qwen",
            "messages": messages,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "repetition_penalty": 1.0,
        }
        if self._use_guided_decoding:
            payload["guided_regex"] = self._GUIDED_REGEX

        try:
            resp = requests.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            generated_text = resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(
                f"vLLM request failed: {e}\n"
                "Make sure the model is running at: "
                f"{self._base_url}"
            ) from e

        token_ids = self._tokenizer.parse_token_ids(generated_text)
        if token_ids is None:
            # Graceful degradation: pad/trim to 8 tokens
            import re
            matches = re.findall(r"TRAJ_(\d{4})", generated_text)
            token_ids = [int(m) for m in matches[:8]]
            token_ids += [0] * (8 - len(token_ids))

        trajectory = self._tokenizer.decode(token_ids)
        return NordOutput(trajectory=trajectory, token_ids=token_ids)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_messages(self, scene: NordInput) -> list[dict]:
        if len(scene.cameras) != 3:
            raise ValueError(
                f"Expected 3 camera images (front_left, front, front_right), "
                f"got {len(scene.cameras)}."
            )

        images_b64 = [self._image_to_b64(img) for img in scene.cameras]
        past_traj_str = self._format_past_tokens(scene.past_token_ids)

        vx, vy = scene.ego_velocity_ms
        ax, ay = scene.ego_acceleration_ms2
        user_text = (
            f"Past 1.5 seconds trajectory: {past_traj_str}\n"
            f"Current [x, y] velocity: [{vx:.3f}, {vy:.3f}] m/s\n"
            f"Current [x, y] acceleration: [{ax:.3f}, {ay:.3f}] m/s^2\n"
            f"Driving command: {scene.driving_command}"
        )

        return [
            {"role": "system", "content": self._SYSTEM_PROMPT},
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

    @staticmethod
    def _image_to_b64(image: Image.Image) -> str:
        if not isinstance(image, Image.Image):
            # Accept numpy arrays for convenience
            image = Image.fromarray(np.asarray(image).astype(np.uint8))
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def _format_past_tokens(token_ids: list[int] | None) -> str:
        return format_past_tokens(token_ids)