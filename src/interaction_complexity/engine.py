from __future__ import annotations

import os
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .config import load_config, write_json, to_jsonable
from .utils import dynamic_n_step, legacy_root, repo_root, resolve_scenario_for_legacy, scenario_id_from_path, robust_z


def _prepare_legacy_imports() -> Path:
    legacy_dir = repo_root() / "src/interaction_complexity/legacy"
    for p in [legacy_root(), legacy_root() / "LeastAction", legacy_dir]:
        sp = str(p)
        if sp in sys.path:
            sys.path.remove(sp)
        sys.path.insert(0, sp)
    return legacy_dir


def _legacy_action_params(config: Dict[str, Any], config_path: str | Path, output_dir: Path) -> Path:
    data = dict(config)
    if "social_config" in data:
        out = data
    else:
        la = data.get("least_action", {})
        pc = data.get("path_conflict", {})
        af = data.get("action_field", {})
        social_config = {
            "path_conflict_enabled": bool(pc.get("enabled", True)),
            "path_conflict_mode": pc.get("mode", "per_step"),
            "path_tube_width": pc.get("path_tube_width", 1.5),
            "conflict_s_margin": pc.get("conflict_s_margin", 4.0),
            "sigma_l": pc.get("sigma_l", 0.6),
            "tau_safe": pc.get("tau_safe", 0.3),
            "beta_time": pc.get("beta_time", 0.8),
            "lambda_path_conflict": pc.get("lambda_path_conflict", 8.0),
            "w_overlap": pc.get("w_overlap", 1.0),
            "ped_bike_weight": pc.get("ped_bike_weight", 3.0),
            "vehicle_weight": pc.get("vehicle_weight", 1.0),
            "crossing_speed_scale": pc.get("crossing_speed_scale", 0.5),
            "sigma_s": pc.get("sigma_s", 4.0),
            "action_field_component_scaled": bool(af.get("component_scaled", True)),
            "action_field_alpha": float(af.get("alpha", 1.0)),
            "action_field_scale_physical": float(af.get("scale_physical", 1.0)),
            "action_field_scale_pc": float(af.get("scale_path_conflict", af.get("scale_pc", 1.0))),
            "action_field_clip": float(af.get("clip", 5.0)),
        }
        out = {
            "w_goal": la.get("w_goal", 150.0),
            "n_soc": la.get("n_soc", 2.0),
            "b_l": la.get("b_l", 1.0),
            "r_coeff": la.get("r_coeff", 100.0),
            "v_ref": la.get("v_ref", 10.0),
            "social_config": social_config,
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp = output_dir / "_legacy_action_params.json"
    write_json(tmp, out)
    return tmp


def compute_ic_single(
    scenario: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    fusion_config_path: str | Path | None = None,
    no_vis: bool = True,
    n_step: Optional[int] = None,
    seed: int = 2025,
) -> Dict[str, Any]:
    np.random.seed(int(seed))
    _prepare_legacy_imports()
    from Interaction_complexity import compute_scene_complexity_for_scenario  # type: ignore

    cfg = load_config(config_path)
    seed = int(cfg.get("seed", seed))
    np.random.seed(seed)
    out_dir = Path(output_dir).resolve()
    sid = scenario_id_from_path(scenario)
    steps = int(n_step) if n_step is not None else dynamic_n_step(sid, 0 if cfg.get("horizon", "full") == "full" else 30)
    legacy_scenario = resolve_scenario_for_legacy(scenario)
    legacy_params = _legacy_action_params(cfg, config_path, out_dir)
    old_cwd = Path.cwd()
    os.chdir(legacy_root())
    try:
        res = compute_scene_complexity_for_scenario(
            scenario_path=legacy_scenario,
            save_vis=not no_vis,
            save_res=True,
            output_root=str(out_dir.parent),
            n_step=steps,
            action_params_json=str(legacy_params),
            action_gp_target=str(cfg.get("action_gp_target", "incremental")),
            dp_mode=str(cfg.get("dp_mode", "incremental_physical")),
            dp_k=int(cfg.get("dp_k", 80)),
            dp_samples=int(cfg.get("dp_samples", 2000)),
            dp_seed=int(seed),
        )
    finally:
        os.chdir(old_cwd)
    actual_dir = out_dir.parent / sid
    if actual_dir.exists() and actual_dir != out_dir:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        actual_dir.rename(out_dir)
    area = np.asarray(res.get("area_penalty", []), dtype=float)
    action = np.asarray(res.get("action_penalty", []), dtype=float)
    summary = {
        "scenario_id": sid,
        "n_step": steps,
        "ic_area": float(np.nanmean(area)) if area.size else None,
        "ic_action": float(np.nanmean(action)) if action.size else None,
        "ic_combined_raw_0p5": float(res.get("overall_score", np.nan)),
        "action_field_component_scaled": bool(res.get("action_params", {}).get("social_config", {}).get("action_field_component_scaled", False)),
        "action_field_alpha": res.get("action_params", {}).get("social_config", {}).get("action_field_alpha"),
        "path_conflict_event_count": int(res.get("actual_path_conflict_event_count", res.get("path_conflict_event_count", 0)) or 0),
        "small_actor_event_count": int(res.get("actual_small_actor_event_count", res.get("path_conflict_small_actor_event_count", 0)) or 0),
    }
    if fusion_config_path and summary["ic_area"] is not None and summary["ic_action"] is not None:
        fusion = load_config(fusion_config_path)
        norm = fusion.get("normalization", {}) or {}
        stats = norm.get("stats", {}) or {}
        clip = float(norm.get("clip", 5.0))
        if "IC-Area" in stats and "IC-Action" in stats:
            z_area = float(robust_z(pd.Series([summary["ic_area"]]), stats["IC-Area"]["median"], stats["IC-Area"]["mad"], clip).iloc[0])
            z_action = float(robust_z(pd.Series([summary["ic_action"]]), stats["IC-Action"]["median"], stats["IC-Action"]["mad"], clip).iloc[0])
            default_w = float((fusion.get("fusion", {}) or {}).get("default_w_area", 0.5))
            paper_w = 0.7
            summary.update(
                {
                    "z_area": z_area,
                    "z_action": z_action,
                    "ic_combined_normalized_default": default_w * z_area + (1.0 - default_w) * z_action,
                    "ic_combined_normalized_w_area_0p7": paper_w * z_area + (1.0 - paper_w) * z_action,
                }
            )
    write_json(out_dir / "scene_complexity.json", summary)
    write_json(out_dir / "metadata.json", {"scenario": str(scenario), "legacy_scenario": legacy_scenario, "config": str(config_path), "fusion_config": str(fusion_config_path) if fusion_config_path else None})
    diag = res.get("path_conflict_diagnostics", {}) or {k: res.get(k) for k in ["actual_path_conflict_cost", "ideal_path_conflict_cost_on_actual_field", "path_conflict_offset", "actual_dp_min_clearance", "ideal_path_min_clearance"] if k in res}
    write_json(out_dir / "path_conflict_diagnostics.json", diag)
    write_json(out_dir / "action_field_diagnostics.json", {"action_gp_target": cfg.get("action_gp_target"), "dp_mode": cfg.get("dp_mode"), "dp_k": cfg.get("dp_k"), "dp_samples": cfg.get("dp_samples"), "action_params": res.get("action_params")})
    return summary
