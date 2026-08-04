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

"""Upload a NoRD checkpoint directory to a HuggingFace Hub model repo.

Example:
    python scripts/upload_to_hub.py \\
        --checkpoint-dir /path/to/staged/nord \\
        --repo-id AppliedIntuitionResearch/nord \\
        --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        type=Path,
        help="Local directory containing the checkpoint (and model card) to upload",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target HF Hub repo, e.g. AppliedIntuitionResearch/nord",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=True,
        help="Create/keep the repo private (default)",
    )
    parser.add_argument(
        "--public",
        dest="private",
        action="store_false",
        help="Create/keep the repo public",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the exact upload target and file list without calling the Hub API",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint_dir.is_dir():
        raise ValueError(f"Checkpoint directory does not exist: {args.checkpoint_dir}")

    files = sorted(p for p in args.checkpoint_dir.rglob("*") if p.is_file())
    total_bytes = sum(p.stat().st_size for p in files)

    print(f"Target repo:     {args.repo_id}")
    print(f"Visibility:      {'private' if args.private else 'public'}")
    print(f"Source dir:      {args.checkpoint_dir}")
    print(f"Files to upload: {len(files)} ({total_bytes / 1e9:.2f} GB)")
    for p in files:
        print(f"  {p.relative_to(args.checkpoint_dir)}  ({p.stat().st_size / 1e6:.1f} MB)")

    if args.dry_run:
        print("\nDry run — no changes made.")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo_id, private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=str(args.checkpoint_dir),
        repo_id=args.repo_id,
        commit_message="Upload NoRD checkpoint",
    )
    print(f"\nUploaded to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
