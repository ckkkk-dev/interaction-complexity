from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from interaction_complexity.config import load_config, write_json
from interaction_complexity.utils import robust_z, stable_split

PLANNERS = {"FRL": "rl_fail", "CRP": "cr_fail", "CRS": "cs_fail"}


def _auc(df: pd.DataFrame, score: str, fail_col: str) -> float:
    d = df[[score, fail_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if d.empty or d[fail_col].nunique() < 2 or d[score].nunique() < 2:
        return np.nan
    return float(roc_auc_score(d[fail_col].astype(int), d[score].astype(float)))


def add_normalized_fusion(scores: pd.DataFrame, fusion_config: Dict, output_dir: Path) -> None:
    out = scores.copy()
    if "split" not in out.columns:
        out["split"] = out["scenario_id"].astype(str).map(stable_split)
    clip = float(fusion_config.get("normalization", {}).get("clip", 5.0))
    stats = fusion_config.get("normalization", {}).get("stats", {}) or {}
    if stats:
        area_med = float(stats["IC-Area"]["median"])
        area_mad = float(stats["IC-Area"]["mad"])
        action_med = float(stats["IC-Action"]["median"])
        action_mad = float(stats["IC-Action"]["mad"])
    else:
        train = out[out["split"] == "train"]
        area_med = float(np.nanmedian(train["IC-Area"]))
        area_mad = float(np.nanmedian(np.abs(train["IC-Area"] - area_med)))
        action_med = float(np.nanmedian(train["IC-Action"]))
        action_mad = float(np.nanmedian(np.abs(train["IC-Action"] - action_med)))
    out["z_area"] = robust_z(out["IC-Area"], area_med, area_mad, clip)
    out["z_action"] = robust_z(out["IC-Action"], action_med, action_mad, clip)
    grid = fusion_config.get("fusion", {}).get("sensitivity_grid", [round(float(x), 1) for x in np.arange(0, 1.01, 0.1)])
    rows = []
    for w in grid:
        w = float(w)
        col = f"IC-Normalized-w_area={w:.1f}"
        out[col] = w * out["z_area"] + (1.0 - w) * out["z_action"]
        row = {"method": col, "w_area": w}
        vals = []
        for planner, fail_col in PLANNERS.items():
            val = _auc(out, col, fail_col)
            row[f"auc_{planner}"] = val
            vals.append(val)
        row["auc_mean"] = float(np.nanmean(vals))
        rows.append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_dir / "scores_and_labels.csv", index=False)
    pd.DataFrame(rows).to_csv(output_dir / "normalized_fusion_grid.csv", index=False)
    write_json(output_dir / "normalized_fusion_summary.json", {"n": int(len(out)), "stats": {"IC-Area": {"median": area_med, "mad": area_mad}, "IC-Action": {"median": action_med, "mad": action_mad}}})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--fusion-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    add_normalized_fusion(pd.read_csv(args.scores), load_config(args.fusion_config), args.output_dir)


if __name__ == "__main__":
    main()
