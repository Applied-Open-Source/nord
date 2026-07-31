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
Minimal NoRD inference example.

Requirements
------------
1. Start the vLLM model server in a separate terminal:

    vllm serve <path-to-model-or-hf-repo> \\
        --served-model-name qwen \\
        --dtype bfloat16 \\
        --port 8000

   To download a checkpoint first:
       python -c "import nord; nord.download_model('nord')"

2. Run this script:

    python examples/run_inference.py \\
        --front-left  images/fl.jpg \\
        --front       images/front.jpg \\
        --front-right images/fr.jpg \\
        --model nord
"""

import argparse

import numpy as np
from PIL import Image

import nord


def main() -> None:
    parser = argparse.ArgumentParser(description="NoRD single-scene inference")
    parser.add_argument("--front-left",  required=True, help="Front-left camera image path")
    parser.add_argument("--front",       required=True, help="Front camera image path")
    parser.add_argument("--front-right", required=True, help="Front-right camera image path")
    parser.add_argument("--model",       default="nord", choices=nord.list_models())
    parser.add_argument("--server",      default="http://localhost:8000/v1",
                        help="vLLM server base URL")
    parser.add_argument("--vocab",       default=None,
                        help="Path to vocab .pkl (optional; downloads from HF Hub if omitted)")
    parser.add_argument("--speed-kmh",   type=float, default=30.0,
                        help="Approximate forward speed in km/h")
    parser.add_argument("--command",     default="straight",
                        choices=["left", "straight", "right"])
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    # Load images
    images = [
        Image.open(args.front_left).convert("RGB"),
        Image.open(args.front).convert("RGB"),
        Image.open(args.front_right).convert("RGB"),
    ]

    # Load agent (downloads vocab from HF Hub if --vocab not provided)
    agent = nord.NordAgent.from_pretrained(
        model_name=args.model,
        base_url=args.server,
        vocab_path=args.vocab,
        temperature=args.temperature,
    )

    # Build scene input
    vx = args.speed_kmh / 3.6
    scene = nord.NordInput(
        cameras=images,
        ego_velocity_ms=(vx, 0.0),
        ego_acceleration_ms2=(0.0, 0.0),
        driving_command=args.command,
    )

    # Run inference
    output = agent.predict(scene)

    print(f"Token IDs : {output.token_ids}")
    print(f"Trajectory: shape={output.trajectory.shape}")
    endpoint = output.trajectory[-1]
    print(f"Endpoint  : x={endpoint[0]:.2f} m (fwd),  y={endpoint[1]:.2f} m (left),  "
          f"heading={np.degrees(endpoint[2]):.1f} deg")


if __name__ == "__main__":
    main()