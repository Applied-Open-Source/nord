# NoRD: No Reasoning for Driving

**CVPR 2026** | [arXiv](https://arxiv.org/abs/2602.21172) | [Project Page](https://nord-vla-ai.github.io/)

*Ishaan Rawal · Shubh Gupta · Yihan Hu · Wei Zhan*

---

A reasoning-free Vision-Language-Action model for autonomous driving. NoRD predicts trajectories directly from surround-view cameras using discrete trajectory tokens — no chain-of-thought, 3× fewer tokens, and competitive PDM scores on NAVSIM with <60% of typical training data.

## Models

| Model | Training | NAVSIM PDMS | HuggingFace |
|---|---|---|---|
| `nord` | SFT + Dr. GRPO | 0.8626 | `nord-vla-ai/nord` |
| `nord-base` | SFT only | 0.7273 | `nord-vla-ai/nord-base` |

Weights will be released at a later date.

## Install

```bash
git clone https://github.com/nord-vla-ai/nord
cd nord
pip install -e ".[serve]"   # includes vllm + transformers
```

Requires Python 3.9+, CUDA GPU with ≥16 GB VRAM.

## Inference

Start the vLLM server, then call the API.

```bash
# Terminal 1
vllm serve nord-vla-ai/nord --served-model-name qwen --dtype bfloat16 --port 8000
```

```python
# Terminal 2
from PIL import Image
import nord

agent = nord.NordAgent.from_pretrained("nord")

output = agent.predict(nord.NordInput(
    cameras=[Image.open("fl.jpg"), Image.open("front.jpg"), Image.open("fr.jpg")],
    ego_velocity_ms=(8.3, 0.0),     # vx, vy in m/s
    driving_command="straight",      # "left" | "straight" | "right"
))

print(output.trajectory.shape)   # (40, 3) — x, y, heading at 10 Hz, 4 seconds
print(output.token_ids)          # [int × 8]
```

Or from the command line:
```bash
python examples/run_inference.py --front-left fl.jpg --front front.jpg --front-right fr.jpg
```

## NAVSIM Evaluation

```bash
# 1. Install NAVSIM v1.1 (the version used in the paper) and download navtest data
git clone --branch v1.1 https://github.com/autonomousvision/navsim && cd navsim && pip install -e .
pip install -e /path/to/nord   # install nord in the same environment

# 2. Start vLLM (as above)

# 3. Run eval
python scripts/eval_navsim.py \
    --model nord \
    --navsim-logs ~/data/navsim/navsim_logs/test \
    --sensor-blobs ~/data/navsim/sensor_blobs/test \
    --metric-cache ~/data/navsim/metric_cache \
    --output-dir results/
```

Expected output (12,147 scenes, ~100 scenes/min on A100):
```
Evaluating nord on navtest (12147 scenes) ...
────────────────────────────────────────────
PDMS                                 0.8626
  no_at_fault_collisions             0.9737
  drivable_area_compliance           0.9522
  ego_progress                       0.8156
  time_to_collision                  0.9253
  comfort                            0.9997
────────────────────────────────────────────
Results saved to results/nord_navtest/scores.csv
```

Add `--max-scenes 50` for a quick smoke-test.

## Citation

```bibtex
@inproceedings{rawal2026nord,
  title={NoRD: A Data-Efficient Vision-Language-Action Model that Drives without Reasoning},
  author={Rawal, Ishaan and Gupta, Shubh and Hu, Yihan and Zhan, Wei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
