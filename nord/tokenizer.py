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

"""K-Disc trajectory tokenizer for NoRD (v3 contour-based format)."""

from __future__ import annotations

import pickle
import re
from typing import Sequence

import numpy as np
import torch


GUIDED_REGEX = r"^((TRAJ_\d{4} ){7}TRAJ_\d{4})$"

SYSTEM_PROMPT = (
    "You are an expert driver. The current position [x, y, yaw] of the vehicle is [0, 0, 0].\n"
    "Input:\n"
    "- 3 frames of multi-view images collected from the ego-vehicle at the present timestep. "
    "The images are in the order: 1. front left, 2. front, 3. front right.\n"
    "- 3 tokens of past 1.5 seconds trajectory\n"
    "- Current speed [x, y] m/s\n"
    "- Current acceleration [x, y] m/s^2\n"
    "- High level driving command (left, right, straight)\n"
    "Given the inputs, predict the optimal 4-second future trajectory at 10Hz containing exactly 8 tokens. "
    "Output format is raw text."
)


def format_past_tokens(token_ids: list | None) -> str:
    """Format 3 past trajectory token IDs as ``TRAJ_XXXX TRAJ_XXXX TRAJ_XXXX``."""
    if not token_ids:
        return "TRAJ_0000 TRAJ_0000 TRAJ_0000"
    ids = (list(token_ids) + [0, 0, 0])[:3]
    return " ".join(f"TRAJ_{tid:04d}" for tid in ids)


class KDiscTokenizer:
    """
    K-means trajectory tokenizer (v3 contour-based format).

    Vocab: 2048 clusters, shift=5, segments of 6 timesteps at 10 Hz.
    Each cluster centroid encodes a vehicle-contour trajectory segment as
    (shift+1, 4, 2) — timesteps × bounding-box corners × xy.

    Load via ``KDiscTokenizer.from_vocab_file(path)``. The vocab ``.pkl``
    ships with each NoRD checkpoint on HuggingFace Hub.
    """

    _TOKEN_RE = re.compile(r"TRAJ_(\d{4})")

    def __init__(self, vocab: dict) -> None:
        if "token_all" not in vocab or "veh" not in vocab["token_all"]:
            raise ValueError(
                "Vocab file is not in v3 format. "
                "Expected a dict with keys 'token_all' → {'veh': ...}."
            )
        self._vocab = vocab
        self._contours: torch.Tensor = torch.tensor(
            vocab["token_all"]["veh"], dtype=torch.float32
        )  # (num_clusters, shift+1, 4, 2)
        self.shift: int = int(vocab.get("shift", 5))
        self.num_clusters: int = self._contours.shape[0]
        self._contours_last: torch.Tensor = self._contours[:, -1, :, :]  # (K, 4, 2)

    @classmethod
    def from_vocab_file(cls, vocab_path: str) -> "KDiscTokenizer":
        """Load tokenizer from a ``.pkl`` vocab file."""
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        return cls(vocab)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decode(self, token_ids: Sequence[int]) -> np.ndarray:
        """
        Decode token IDs to a 10 Hz trajectory.

        Parameters
        ----------
        token_ids
            Sequence of 8 integer token indices (0 ≤ id < num_clusters).

        Returns
        -------
        poses : np.ndarray, shape (40, 3)
            ``[x, y, heading]`` at 10 Hz, t = 0.1 … 4.0 s, in the ego frame
            at the current timestep (x = forward, y = left). Units: metres /
            radians.
        """
        ids = list(token_ids)
        for tid in ids:
            if tid < 0 or tid >= self.num_clusters:
                raise ValueError(
                    f"Token ID {tid} out of range [0, {self.num_clusters})."
                )

        n_tokens = len(ids)
        max_len = self.shift * n_tokens + 1

        reconstructed = torch.zeros((max_len, 3))
        counts = torch.zeros((max_len, 1))

        current_pos = torch.zeros(2)
        current_head = torch.tensor(0.0)

        for seg_idx, token_idx in enumerate(ids):
            vocab_local = self._contours[token_idx]  # (shift+1, 4, 2)
            start_idx = seg_idx * self.shift

            R = self._rotation_matrix(current_head, dtype=vocab_local.dtype)

            # Transform local contour to global: (shift+1)*4 points
            flat = vocab_local.flatten(0, 1)           # ((shift+1)*4, 2)
            flat_global = flat @ R + current_pos        # ((shift+1)*4, 2)
            contour_global = flat_global.view(-1, 4, 2) # (shift+1, 4, 2)

            # Centre and heading from contour geometry
            pos_global = contour_global.mean(dim=1)     # (shift+1, 2)
            dxy = contour_global[:, 0] - contour_global[:, 3]
            head_global = self._wrap_angle(torch.atan2(dxy[:, 1], dxy[:, 0]))

            L = pos_global.shape[0]
            write_start = 0 if seg_idx == 0 else 1
            if write_start >= L:
                continue
            write_offset = start_idx + write_start
            end_idx = min(start_idx + L, max_len)
            actual_L = end_idx - write_offset

            reconstructed[write_offset:end_idx, :2] += pos_global[write_start : write_start + actual_L]
            reconstructed[write_offset:end_idx, 2] += head_global[write_start : write_start + actual_L]
            counts[write_offset:end_idx] += 1

            current_pos = pos_global[L - 1]
            current_head = head_global[L - 1]

        mask = counts > 0
        reconstructed = reconstructed / torch.where(mask, counts, torch.ones_like(counts))

        # Drop t=0; return t=0.1 … 4.0
        return reconstructed.numpy()[1:].astype(np.float32)

    def parse_token_ids(self, response_text: str) -> list[int] | None:
        """
        Parse ``TRAJ_XXXX`` tokens from a model response string.

        Returns a list of 8 integers, or ``None`` if the text does not contain
        exactly 8 valid tokens.
        """
        matches = self._TOKEN_RE.findall(response_text)
        if len(matches) != 8:
            return None
        return [int(m) for m in matches]

    def encode(self, trajectory: np.ndarray) -> list[int]:
        """
        Encode a 10 Hz trajectory to token IDs (for past-trajectory conditioning).

        Parameters
        ----------
        trajectory
            ``(T, 3)`` array of ``[x, y, heading]`` in the ego frame at 10 Hz.
            Typically T = 16 for 1.5 s of history (gives 3 tokens with shift=5).

        Returns
        -------
        token_ids : list[int]
        """
        traj = torch.tensor(trajectory, dtype=torch.float32)
        T = traj.shape[0]
        pos = traj[:, :2]
        head = self._wrap_angle(traj[:, 2])

        gt_contours = self._cal_polygon_contour(pos, head)  # (T, 4, 2)

        n_tokens = (T - 1) // self.shift
        token_ids: list[int] = []

        current_pos = torch.zeros(2)
        current_head = torch.tensor(0.0)

        for seg_idx in range(n_tokens):
            start = seg_idx * self.shift
            end = min(start + self.shift + 1, T)
            gt_last = gt_contours[end - 1]  # (4, 2)

            R = self._rotation_matrix(current_head, dtype=self._contours.dtype)

            # Vocab last frame in global coords
            flat = self._contours_last.flatten(0, 1)                # (K*4, 2)
            flat_global = flat @ R + current_pos
            token_last_global = flat_global.view(self.num_clusters, 4, 2)

            diff = token_last_global - gt_last.unsqueeze(0)
            dists = torch.norm(diff, dim=-1).sum(-1)
            best = int(torch.argmin(dists).item())
            token_ids.append(best)

            # Update rollout state
            chosen_local = self._contours[best]                     # (shift+1, 4, 2)
            flat_c = chosen_local.flatten(0, 1)
            flat_c_global = flat_c @ R + current_pos
            contour_g = flat_c_global.view(-1, 4, 2)
            pos_g = contour_g.mean(dim=1)
            dxy = contour_g[:, 0] - contour_g[:, 3]
            head_g = self._wrap_angle(torch.atan2(dxy[:, 1], dxy[:, 0]))
            current_pos = pos_g[-1]
            current_head = head_g[-1]

        return token_ids

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap_angle(theta: torch.Tensor) -> torch.Tensor:
        return (theta + torch.pi) % (2 * torch.pi) - torch.pi

    @staticmethod
    def _rotation_matrix(head: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
        """Return 2×2 rotation matrix for heading ``head``."""
        cos_h = torch.cos(head)
        sin_h = torch.sin(head)
        R = torch.zeros((2, 2), dtype=dtype)
        R[0, 0] = cos_h
        R[0, 1] = sin_h
        R[1, 0] = -sin_h
        R[1, 1] = cos_h
        return R

    @staticmethod
    def _cal_polygon_contour(
        pos: torch.Tensor,
        head: torch.Tensor,
        width_length: tuple[float, float] = (2.0, 4.8),
    ) -> torch.Tensor:
        """Compute (T, 4, 2) bounding-box corners from positions and headings."""
        w, l = width_length
        corners = torch.tensor(
            [[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]],
            dtype=pos.dtype,
        )
        result = []
        for p, h in zip(pos, head):
            R = torch.tensor(
                [[torch.cos(h), -torch.sin(h)], [torch.sin(h), torch.cos(h)]],
                dtype=pos.dtype,
            )
            result.append((R @ corners.T).T + p)
        return torch.stack(result)