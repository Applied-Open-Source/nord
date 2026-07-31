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

"""Model registry and HuggingFace Hub helpers for NoRD."""

from __future__ import annotations

import os
from pathlib import Path

# Vocab is bundled with the package — no separate download needed.
BUNDLED_VOCAB = Path(__file__).parent / "data" / "vocab.pkl"

NORD_MODELS: dict[str, dict[str, str]] = {
    "nord": {
        "hf_repo": "nord-vla-ai/nord",
        "description": "NoRD: Dr. GRPO fine-tuned (paper's main result)",
    },
    "nord-base": {
        "hf_repo": "nord-vla-ai/nord-base",
        "description": "NoRD-Base: SFT-only trained (ablation baseline)",
    },
}


def list_models() -> list[str]:
    return list(NORD_MODELS.keys())


def download_model(
    model_name: str,
    cache_dir: str | None = None,
    local_files_only: bool = False,
) -> str:
    """
    Download a NoRD model checkpoint from HuggingFace Hub.

    Returns ``model_dir`` — a local path suitable for ``vllm serve --model``.
    The vocab is bundled with this package; no separate vocab download needed.
    """
    if model_name not in NORD_MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list_models()}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError("pip install huggingface_hub") from e

    return snapshot_download(
        repo_id=NORD_MODELS[model_name]["hf_repo"],
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )