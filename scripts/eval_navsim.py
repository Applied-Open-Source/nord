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
Evaluate NoRD on the NAVSIM benchmark.

Prerequisites
-------------
1. Install NAVSIM:
       git clone https://github.com/autonomousvision/navsim
       cd navsim && pip install -e .
   Download the navtest dataset and metric cache (see NAVSIM README).

2. Start vLLM:
       vllm serve AppliedIntuitionResearch/nord --served-model-name qwen --dtype bfloat16 --port 8000

3. Run this script:
       python scripts/eval_navsim.py \\
           --model nord \\
           --navsim-logs ~/data/navsim/navsim_logs/test \\
           --sensor-blobs ~/data/navsim/sensor_blobs/test \\
           --metric-cache ~/data/navsim/metric_cache
"""

from __future__ import annotations

import argparse
import inspect
import lzma
import pickle
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from tqdm import tqdm


def _check_navsim() -> None:
    try:
        import navsim  # noqa: F401
    except ImportError:
        sys.exit(
            "NAVSIM is not installed.\n"
            "  git clone https://github.com/autonomousvision/navsim\n"
            "  cd navsim && pip install -e ."
        )


def _get_metric(obj, *keys: str, default: float = 0.0) -> float:
    """Extract a float from a PDMResult (handles v1 dataclass and v2 DataFrame row)."""
    for k in keys:
        v = obj.get(k) if hasattr(obj, "get") else getattr(obj, k, None)
        if v is not None:
            return float(v)
    return default


def build_simulator_and_scorer():
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig

    proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    simulator = PDMSimulator(proposal_sampling=proposal_sampling)
    scorer = PDMScorer(proposal_sampling=proposal_sampling, config=PDMScorerConfig())
    return simulator, scorer


def run_evaluation(
    model: str,
    vocab_path: Optional[str],
    navsim_logs: str,
    sensor_blobs: str,
    metric_cache: str,
    server_url: str,
    split: str,
    output_dir: str,
    max_scenes: Optional[int],
) -> pd.DataFrame:
    from navsim.common.dataclasses import SceneFilter
    from navsim.common.dataloader import SceneLoader, MetricCacheLoader
    from navsim.evaluate.pdm_score import pdm_score as pdm_score_fn
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    from nord.navsim_agent import NordNavsimAgent

    agent = NordNavsimAgent(vocab_path=vocab_path, server_url=server_url)
    agent.initialize()

    # Filter to .pkl files only (some dirs have non-pkl symlinks that confuse the loader)
    pkl_log_names = [p.stem for p in Path(navsim_logs).iterdir() if p.suffix == ".pkl"]

    scene_filter = SceneFilter(
        num_history_frames=4,
        num_future_frames=10,
        frame_interval=1,
        has_route=True,
        log_names=pkl_log_names,
    )

    # SceneLoader sensor-path arg varies by NAVSIM version
    _sensor_arg = (
        "sensor_blobs_path"
        if "sensor_blobs_path" in inspect.signature(SceneLoader).parameters
        else "original_sensor_path"
    )
    scene_loader = SceneLoader(
        data_path=Path(navsim_logs),
        **{_sensor_arg: Path(sensor_blobs)},
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )
    metric_cache_loader = MetricCacheLoader(Path(metric_cache))

    tokens = sorted(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    if max_scenes is not None:
        tokens = tokens[:max_scenes]
    print(f"Evaluating {model} on {split} ({len(tokens)} scenes) ...")

    simulator, scorer = build_simulator_and_scorer()
    future_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)

    # Detect whether this NAVSIM version's pdm_score requires a traffic_agents_policy arg
    _needs_traffic = "traffic_agents_policy" in inspect.signature(pdm_score_fn).parameters
    if _needs_traffic:
        from navsim.traffic_agents_policies.navsim_IDM_traffic_agents import NavsimIDMTrafficAgents
        from navsim.planning.simulation.observation.navsim_idm_agents import NavsimIDMAgents
        traffic_policy = NavsimIDMTrafficAgents(
            future_trajectory_sampling=future_sampling,
            idm_agents_observation=NavsimIDMAgents(
                target_velocity=10, min_gap_to_lead_agent=1.0, headway_time=1.5,
                accel_max=1.0, decel_max=2.0, open_loop_detections_types=[],
                minimum_path_length=20, planned_trajectory_samples=None,
                planned_trajectory_sample_interval=None, radius=100,
                add_open_loop_parked_vehicles=True, idm_snap_threshold=3.0,
            ),
        )
    else:
        traffic_policy = None

    rows: list[dict[str, Any]] = []
    for token in tqdm(tokens, desc=model):
        try:
            mc_path = metric_cache_loader.metric_cache_paths[token]
            with lzma.open(mc_path, "rb") as f:
                mc = pickle.load(f)

            trajectory = agent.compute_trajectory(scene_loader.get_agent_input_from_token(token))

            if _needs_traffic:
                result_out = pdm_score_fn(mc, trajectory, future_sampling, simulator, scorer, traffic_policy)
                r = (result_out[0] if isinstance(result_out, tuple) else result_out).iloc[0]
            else:
                r = pdm_score_fn(mc, trajectory, future_sampling, simulator, scorer)

            row = {
                "token": token,
                "score": _get_metric(r, "pdm_score", "score"),
                "no_at_fault_collisions": _get_metric(r, "no_at_fault_collisions"),
                "drivable_area_compliance": _get_metric(r, "drivable_area_compliance"),
                "ego_progress": _get_metric(r, "ego_progress"),
                "time_to_collision_within_bound": _get_metric(r, "time_to_collision_within_bound"),
                "comfort": _get_metric(r, "history_comfort", "comfort"),
            }
        except AttributeError as e:
            if "current_tracked_objects" in str(e) or "future_tracked_objects" in str(e):
                sys.exit(
                    f"\nERROR: Metric cache is incompatible with the installed NAVSIM version.\n"
                    f"  ({e})\n\n"
                    "Your metric cache was built with an older NAVSIM version that used a different\n"
                    "MetricCache format. Please rebuild it:\n\n"
                    "  cd /path/to/navsim\n"
                    "  python navsim/planning/script/run_dataset_caching.py\n\n"
                    "See the NAVSIM README for the full caching instructions."
                )
            print(f"  [skip] {token}: {e}")
            row = {"token": token, "score": 0.0, "no_at_fault_collisions": 0.0,
                   "drivable_area_compliance": 0.0, "ego_progress": 0.0,
                   "time_to_collision_within_bound": 0.0, "comfort": 0.0}
        except Exception as e:
            print(f"  [skip] {token}: {e}")
            row = {"token": token, "score": 0.0, "no_at_fault_collisions": 0.0,
                   "drivable_area_compliance": 0.0, "ego_progress": 0.0,
                   "time_to_collision_within_bound": 0.0, "comfort": 0.0}
        rows.append(row)

    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame) -> None:
    metrics = [
        ("PDMS", "score"),
        ("  no_at_fault_collisions   ", "no_at_fault_collisions"),
        ("  drivable_area_compliance ", "drivable_area_compliance"),
        ("  ego_progress             ", "ego_progress"),
        ("  time_to_collision        ", "time_to_collision_within_bound"),
        ("  comfort                  ", "comfort"),
    ]
    sep = "─" * 44
    print(sep)
    for label, col in metrics:
        print(f"{label:<36} {df[col].mean():.4f}")
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NoRD on NAVSIM navtest")
    parser.add_argument("--model",        choices=["nord", "nord-base"], default="nord")
    parser.add_argument("--vocab",        default=None, help="Path to vocab .pkl (default: bundled)")
    parser.add_argument("--navsim-logs",  required=True)
    parser.add_argument("--sensor-blobs", required=True)
    parser.add_argument("--metric-cache", required=True)
    parser.add_argument("--server",       default="http://localhost:8000/v1")
    parser.add_argument("--split",        default="navtest")
    parser.add_argument("--output-dir",   default="results/")
    parser.add_argument("--max-scenes",   type=int, default=None)
    args = parser.parse_args()

    _check_navsim()

    df = run_evaluation(
        model=args.model,
        vocab_path=args.vocab,
        navsim_logs=args.navsim_logs,
        sensor_blobs=args.sensor_blobs,
        metric_cache=args.metric_cache,
        server_url=args.server,
        split=args.split,
        output_dir=args.output_dir,
        max_scenes=args.max_scenes,
    )

    print_summary(df)

    out_dir = Path(args.output_dir) / f"{args.model}_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "scores.csv"
    df.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()