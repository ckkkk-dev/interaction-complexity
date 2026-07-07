#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") != "0":
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from interaction_complexity.engine import compute_ic_single


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IC-V5 for one CommonRoad scenario.")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--config", default="configs/ic_v5_alpha1.json")
    parser.add_argument("--fusion-config", default="configs/normalized_fusion.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-step", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-vis", action="store_true")
    args = parser.parse_args()
    summary = compute_ic_single(
        scenario=args.scenario,
        config_path=args.config,
        fusion_config_path=args.fusion_config,
        output_dir=args.output_dir,
        no_vis=args.no_vis,
        n_step=args.n_step,
        seed=2025 if args.seed is None else args.seed,
    )
    print(summary)


if __name__ == "__main__":
    main()
