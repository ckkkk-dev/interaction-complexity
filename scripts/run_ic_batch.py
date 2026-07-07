#!/usr/bin/env python
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from interaction_complexity.config import write_json
from interaction_complexity.utils import load_scenario_list, scenario_id_from_path
from interaction_complexity.evaluation.normalized_fusion import add_normalized_fusion
from interaction_complexity.config import load_config


def scenario_arg(sid: str) -> str:
    if sid.endswith(".xml") or "/" in sid:
        return sid
    return f"data/sind_left_turn/{sid}.xml"


def run_one(sid: str, args) -> dict:
    scenario_id = scenario_id_from_path(sid)
    out_dir = Path(args.output_dir) / "scenes" / scenario_id
    log_dir = Path(args.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "scripts/run_ic_single.py"),
        "--scenario", scenario_arg(sid),
        "--config", args.config,
        "--fusion-config", args.fusion_config,
        "--output-dir", str(out_dir),
        "--no-vis",
    ]
    if args.resume and (out_dir / "scene_complexity.pkl").exists():
        return {"scenario_id": scenario_id, "status": "skipped"}
    import os
    env = os.environ.copy()
    env.setdefault("PYTHONHASHSEED", "0")
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env.setdefault(var, "1")
    with (log_dir / f"{scenario_id}.log").open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
    return {"scenario_id": scenario_id, "status": "ok" if proc.returncode == 0 else "failed", "returncode": proc.returncode}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IC-V5 for a scenario list.")
    parser.add_argument("--scenario-list", required=True)
    parser.add_argument("--config", default="configs/ic_v5_alpha1.json")
    parser.add_argument("--fusion-config", default="configs/normalized_fusion.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ids = load_scenario_list(args.scenario_list)
    rows = []
    with cf.ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futs = [ex.submit(run_one, sid, args) for sid in ids]
        for fut in cf.as_completed(futs):
            row = fut.result()
            rows.append(row)
            print(row, flush=True)
    failed = sorted([r["scenario_id"] for r in rows if r["status"] == "failed"])
    (output_dir / "failed_scenarios.txt").write_text("\n".join(failed) + ("\n" if failed else ""), encoding="utf-8")
    write_json(output_dir / "run_summary.json", {"scenario_count": len(ids), "failed_count": len(failed), "failed": failed})
    if failed:
        raise SystemExit(f"failed_scenarios={len(failed)}")
    final_dir = output_dir / "final_auc_summary"
    cmd = [
        sys.executable,
        "-m", "interaction_complexity.evaluation.summarize_ic_auc",
        "--scene-root", str(output_dir / "scenes"),
        "--output-dir", str(final_dir),
    ]
    
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)
    scores_path = final_dir / "social_force_scores_and_labels.csv"
    add_normalized_fusion(pd_read(scores_path), load_config(args.fusion_config), output_dir)


def pd_read(path: Path):
    import pandas as pd
    return pd.read_csv(path)


if __name__ == "__main__":
    main()
