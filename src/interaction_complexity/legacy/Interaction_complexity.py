"""Compute interaction complexity for a CommonRoad scenario.

This legacy-compatible module contains the full-horizon reachable-area and
least-action GP/DP implementation used by the public IC-V5 API.
"""
import os
import sys
import json
import pathlib
import argparse
import pickle
from typing import Optional

# --------------------------------------------------------------------------
# 1) 第三方 / 项目内部依赖
# --------------------------------------------------------------------------
sys.path.append("../")  # 若脚本与工程根目录平级，可按需调整
import numpy as np
import pandas as pd
import commonroad_reach.utility.logger as util_logger
from commonroad_reach.data_structure.configuration_builder import ConfigurationBuilder
from commonroad_reach.data_structure.reach.reach_interface import ReachableSetInterface
from commonroad_reach.utility import visualization as util_visual

from LeastAction_calculate import (
    main as least_action_main,
    compute_action_offset,
    compute_action_offset_incremental,
    eval_ideal_on_actual_field,
    eval_ideal_on_actual_field_incremental,
    build_path_conflict_events,
    path_conflict_cost_array,
    path_conflict_potential_at_points,
    normalize_social_config,
)
from interaction_complexity_calculate import compute_scene_complexity

# --------------------------------------------------------------------------
# 2) 固定超参数 (可根据需要修改或做成 CLI 参数)
# --------------------------------------------------------------------------
N_STEP = 30             # 规划步数
M_I    = 1500.0         # 自车质量 (kg)
G      = 9.81           # 重力加速度
N_G    = 0.10           # 经验系数
L_W    = 3.5            # 车道宽 (m)
V_LIM  = 13.9           # 限速 ≈50 km/h
V_DES  = 13.9           # 期望速度
V_REF  = 6.0            # 经验预设
B_L    = 2              # Lane type: 虚线→2, 实线→3
N_SOC  = 2.5
W_GOAL = 127.42749857031335
R_COEFF = 20.0


def _load_action_params(path: Optional[str]) -> dict:
    params = {
        "v_ref": V_REF,
        "b_l": B_L,
        "n_soc": N_SOC,
        "w_goal": W_GOAL,
        "r_coeff": R_COEFF,
        "social_config": None,
    }
    if path is None:
        return params
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mapping = {
        "v_ref": "v_ref",
        "b_l": "b_l",
        "n_soc": "n_soc",
        "w_goal": "w_goal",
        "r_coeff": "r_coeff",
    }
    for dst, src in mapping.items():
        if src in data:
            params[dst] = float(data[src])
    social_keys = {
        "filter_mode",
        "coupling",
        "k",
        "T0",
        "r_safe",
        "front_weight",
        "back_weight",
        "lateral_weight",
        "v_floor_s",
        "v_floor_l",
        "slack_back",
        "front_s_window",
        "back_s_window",
        "lat_window",
        "goal_s_window",
        "goal_l_window",
        "path_conflict_enabled",
        "path_tube_width",
        "conflict_s_margin",
        "sigma_l",
        "tau_safe",
        "beta_time",
        "lambda_path_conflict",
        "w_overlap",
        "ped_bike_weight",
        "vehicle_weight",
        "crossing_speed_scale",
        "sigma_s",
        "path_conflict_mode",
        "rho_conflict_drive",
        "action_field_component_scaled",
        "action_field_alpha",
        "action_field_scale_physical",
        "action_field_scale_pc",
        "action_field_clip",
    }
    social_config = {}
    if isinstance(data.get("social_config"), dict):
        social_config.update(data["social_config"])
    for key in social_keys:
        if key in data:
            social_config[key] = data[key]
    if "lambda_path_conflict" in data:
        social_config["path_conflict_enabled"] = True
    if social_config.get("path_conflict_enabled"):
        social_config.setdefault("path_conflict_mode", "per_step")
    if social_config:
        params["social_config"] = social_config
    return params


def _path_to_profile(path, dt: float) -> dict:
    if isinstance(path, dict):
        s = np.asarray(path["s"], dtype=float)
        l = np.asarray(path["l"], dtype=float)
    else:
        arr = np.asarray([(p[0], p[1]) for p in path], dtype=float)
        s = arr[:, 0]
        l = arr[:, 1]
    return {
        "s": s,
        "l": l,
        "v_x": np.gradient(s, dt) if len(s) > 1 else np.zeros_like(s),
        "v_y": np.gradient(l, dt) if len(l) > 1 else np.zeros_like(l),
    }


def _eval_ideal_on_actual_field_incremental_with_residual(opt_path_ideal, gp_models, events, dt, n_soc, social_config, eta_pc_residual):
    base = eval_ideal_on_actual_field_incremental(opt_path_ideal, gp_models)
    if not events or float(eta_pc_residual) == 0.0:
        return base
    profile = _path_to_profile(opt_path_ideal, dt)
    pts = np.column_stack([profile["s"], profile["l"]])
    residual = np.array([
        path_conflict_potential_at_points(pts[t:t+1], t, events, dt, n_soc, social_config)[0]
        for t in range(len(pts))
    ], dtype=float) * float(dt) * float(eta_pc_residual)
    return base + residual


def _first_event_timing(profile: dict, events: list, dt: float) -> dict:
    if not events:
        return {
            "arrival_time_at_first_event": np.nan,
            "clearance_after_first_event": np.nan,
            "wait_time_before_first_event": 0.0,
        }
    event = sorted(events, key=lambda ev: float(ev.get("t_enter", 0.0)))[0]
    s_arr = np.asarray(profile["s"], dtype=float)
    vx = np.asarray(profile.get("v_x", np.gradient(s_arr, dt) if len(s_arr) > 1 else np.zeros_like(s_arr)), dtype=float)
    in_zone = (s_arr >= float(event["s_min"])) & (s_arr <= float(event["s_max"]))
    idx = np.flatnonzero(in_zone)
    if idx.size:
        arrival_time = float(idx[0]) * float(dt)
        clearance = arrival_time - float(event["t_exit"])
    else:
        arrival_time = np.nan
        clearance = np.nan
    pre_zone = (s_arr >= float(event["s_min"]) - 5.0) & (s_arr < float(event["s_min"]))
    wait_time = float(np.sum(pre_zone & (np.abs(vx) < 0.5)) * float(dt))
    return {
        "arrival_time_at_first_event": float(arrival_time) if np.isfinite(arrival_time) else np.nan,
        "clearance_after_first_event": float(clearance) if np.isfinite(clearance) else np.nan,
        "wait_time_before_first_event": wait_time,
    }


def _min_path_clearance_after(profile: dict, events: list, dt: float) -> float:
    s_arr = np.asarray(profile["s"], dtype=float)
    min_clearance = np.inf
    for event in events:
        in_zone = (s_arr >= float(event["s_min"])) & (s_arr <= float(event["s_max"]))
        idx = np.flatnonzero(in_zone)
        if idx.size == 0:
            continue
        clearance = float(idx[0]) * float(dt) - float(event["t_exit"])
        min_clearance = min(min_clearance, clearance)
    return float(min_clearance) if np.isfinite(min_clearance) else np.nan


def _path_conflict_sum(profile: dict, events: list, dt: float, n_soc: float, social_config: dict | None) -> float:
    arr = path_conflict_cost_array(profile, events, dt, n_soc, social_config)
    return float(np.nansum(arr) * float(dt))


def _compute_path_conflict_diagnostics(config, n_step, opt_path_actual, opt_path_ideal, n_soc, social_config):
    cfg = normalize_social_config(social_config)
    if not bool(cfg.get("path_conflict_enabled", False)):
        return {}
    dt = float(config.planning.dt)
    events = build_path_conflict_events(config, n_step, cfg)
    actual_profile = _path_to_profile(opt_path_actual, dt)
    ideal_profile = _path_to_profile(opt_path_ideal, dt)
    actual_cost = _path_conflict_sum(actual_profile, events, dt, n_soc, cfg)
    ideal_cost = _path_conflict_sum(ideal_profile, events, dt, n_soc, cfg)
    actual_timing = _first_event_timing(actual_profile, events, dt)
    ideal_timing = _first_event_timing(ideal_profile, events, dt)
    return {
        "actual_path_conflict_cost": actual_cost,
        "ideal_path_conflict_cost_on_actual_field": ideal_cost,
        "path_conflict_offset": ideal_cost - actual_cost,
        "actual_dp_min_clearance": _min_path_clearance_after(actual_profile, events, dt),
        "ideal_path_min_clearance": _min_path_clearance_after(ideal_profile, events, dt),
        "actual_arrival_time_at_first_event": actual_timing["arrival_time_at_first_event"],
        "ideal_arrival_time_at_first_event": ideal_timing["arrival_time_at_first_event"],
        "actual_clearance_after_first_event": actual_timing["clearance_after_first_event"],
        "ideal_clearance_after_first_event": ideal_timing["clearance_after_first_event"],
        "actual_wait_time_before_event": actual_timing["wait_time_before_first_event"],
        "actual_path_conflict_event_count": int(len(events)),
        "actual_small_actor_event_count": int(sum(int(ev.get("is_small_actor", 0)) for ev in events)),
    }


def _add_action_normalizations(res: dict, eps: float = 1e-6) -> None:
    delta = np.asarray(res.get("delta_S", []), dtype=float)
    if delta.size == 0:
        return
    med = np.nanmedian(delta)
    mad = np.nanmedian(np.abs(delta - med))
    q25, q75 = np.nanpercentile(delta, [25, 75])
    iqr = q75 - q25
    res["action_penalty_raw_delta"] = delta
    res["action_penalty_mad"] = delta / (max(float(mad), eps))
    res["action_penalty_iqr"] = delta / (max(float(iqr), eps))


def _build_configuration_for_scenario(scenario_path: str):
    """Build a CommonRoad-Reach configuration for an explicit scenario path.

    Some CommonRoad-Reach installations ship a default `path_scenarios` pointing
    to the developer's local data root. We therefore build by scenario name and
    then explicitly override `general.path_scenario` when a real XML path is
    provided.
    """
    xml_path = scenario_path if scenario_path.endswith(".xml") else f"{scenario_path}.xml"
    name_scenario = os.path.basename(scenario_path)
    config = ConfigurationBuilder(path_root=None).build_configuration(name_scenario)
    if os.path.exists(xml_path):
        config.general.path_scenario = os.path.abspath(xml_path)
    return config

# --------------------------------------------------------------------------
# 3) 核心流程函数
# --------------------------------------------------------------------------
def compute_scene_complexity_for_scenario(
    scenario_path: str,
    save_vis: bool = True,
    save_res: bool = True,
    output_root: str = "outputs/scenarios",
    n_step: Optional[int] = None,
    action_params_json: Optional[str] = None,
    action_gp_target: str = "cumulative",
    dp_mode: str = "legacy_current",
    dp_k: int = 80,
    dp_samples: int = 2000,
    dp_seed: int = 2025,
    eta_pc_residual: float = 0.0,
    rho_conflict_drive: float = 0.0,
    lambda_acc: float = 0.5,
    hold_margin: float = 3.0,
    yield_delays: Optional[str] = None,
):
    """
    计算单个场景复杂度并可选保存结果与可视化

    Parameters
    ----------
    scenario_path : str
        形如 "data/scenarios/left_turn_scenarios/CHN_SIND-821_8_LEFT_T-56"
        或绝对路径；可包含/不包含 .xml
    save_vis : bool
        是否生成 reachable set 可视化 (PNG)
    save_res : bool
        是否保存整体结果 dict (pkl & json)
    output_root : str
        保存目录根路径
    """
    n_step = N_STEP if n_step is None else int(n_step)
    if n_step < 1:
        raise ValueError(f"n_step must be positive, got {n_step}")
    params = _load_action_params(action_params_json)
    if params.get("social_config") is not None:
        params["social_config"] = dict(params["social_config"])
        params["social_config"]["rho_conflict_drive"] = float(rho_conflict_drive)
    incremental_mode = action_gp_target == "incremental" or dp_mode in {"incremental_physical", "time_velocity"}

    scenario_path = pathlib.Path(scenario_path).as_posix()
    name_scenario = os.path.basename(scenario_path)
    out_dir = pathlib.Path(output_root) / name_scenario
    out_dir.mkdir(parents=True, exist_ok=True)

    # ===== 1) 理想 (无动态障碍物) 配置 =====
    config_ideal = _build_configuration_for_scenario(scenario_path)
    config_ideal.planning.steps_computation = n_step
    config_ideal.update()

    # 移除所有动态障碍物
    for oid in config_ideal.scenario.dynamic_obstacles:
        config_ideal.scenario.remove_obstacle(oid)
    config_ideal.update(
        scenario=config_ideal.scenario,
        planning_problem_set=config_ideal.planning_problem_set,
    )

    reach_if_ideal = ReachableSetInterface(config_ideal)
    reach_if_ideal.compute_reachable_sets()
    corridors_ideal = reach_if_ideal.extract_driving_corridors()

    # ideal 最优轨迹
    ideal_out = least_action_main(
        config_ideal,
        reach_if_ideal,
        corridors_ideal,
        M_I,
        G,
        N_G,
        L_W,
        V_LIM,
        V_DES,
        params["v_ref"],
        params["b_l"],
        params["n_soc"],
        params["w_goal"],
        n_step,
        r_coeff=params["r_coeff"],
        action_gp_target=action_gp_target,
        dp_mode=dp_mode,
        dp_k=dp_k,
        dp_samples=dp_samples,
        dp_seed=dp_seed,
        dp_lambda_acc=lambda_acc,
        eta_pc_residual=eta_pc_residual,
        hold_margin=hold_margin,
        yield_delays=yield_delays,
        return_diagnostics=True,
        social_config=params.get("social_config"),
    )
    _, _, _, opt_path_ideal, _, dp_diag_ideal = ideal_out

    # ===== 2) 实际 (含动态障碍物) 配置 =====
    config = _build_configuration_for_scenario(scenario_path)
    config.debug.draw_ref_path = True
    config.planning.steps_computation = n_step
    config.update()

    reach_if = ReachableSetInterface(config)
    reach_if.compute_reachable_sets()
    corridors = reach_if.extract_driving_corridors()

    (
        gp_models,
        all_speed_profile,
        S_all,
        opt_path_actual,
        final,
        dp_diag_actual,
    ) = least_action_main(
        config,
        reach_if,
        corridors,
        M_I,
        G,
        N_G,
        L_W,
        V_LIM,
        V_DES,
        params["v_ref"],
        params["b_l"],
        params["n_soc"],
        params["w_goal"],
        n_step,
        r_coeff=params["r_coeff"],
        action_gp_target=action_gp_target,
        dp_mode=dp_mode,
        dp_k=dp_k,
        dp_samples=dp_samples,
        dp_seed=dp_seed,
        dp_lambda_acc=lambda_acc,
        eta_pc_residual=eta_pc_residual,
        hold_margin=hold_margin,
        yield_delays=yield_delays,
        return_diagnostics=True,
        social_config=params.get("social_config"),
    )

    # ===== 3) 场景复杂度计算 =====
    if dp_mode == "time_velocity":
        actual_events_for_eval = build_path_conflict_events(config, n_step, normalize_social_config(params.get("social_config")))
        eval_action_func = lambda traj, models: _eval_ideal_on_actual_field_incremental_with_residual(
            traj,
            models,
            actual_events_for_eval,
            float(config.planning.dt),
            params["n_soc"],
            normalize_social_config(params.get("social_config")),
            eta_pc_residual,
        )
    else:
        eval_action_func = eval_ideal_on_actual_field_incremental if incremental_mode else eval_ideal_on_actual_field

    res = compute_scene_complexity(
        corridors_ideal,
        corridors,
        opt_path_ideal,
        opt_path_actual,
        gp_models,
        n_step,
        eval_action_func,
        compute_action_offset_incremental if incremental_mode else compute_action_offset,
        w_area=0.5,
        w_action=0.5,
    )

    print(f"[{name_scenario}] 场景总体复杂度分数 = {res['overall_score']:.4f}")
    res["n_step"] = n_step
    res["action_params"] = params
    res["action_gp_target"] = action_gp_target
    res["dp_mode"] = dp_mode
    res["dp_diagnostics"] = {
        "ideal": dp_diag_ideal,
        "actual": dp_diag_actual,
    }
    path_conflict_diag = _compute_path_conflict_diagnostics(
        config,
        n_step,
        opt_path_actual,
        opt_path_ideal,
        params["n_soc"],
        params.get("social_config"),
    )
    if path_conflict_diag:
        res["path_conflict_diagnostics"] = path_conflict_diag
        res.update(path_conflict_diag)
    for key in ["yielding_node_usage_count", "yielding_node_usage_ratio", "dp_velocity_profile", "dp_acceleration_profile", "path_conflict_residual_sum", "transition_cost_sum"]:
        if key in dp_diag_actual:
            res[key] = dp_diag_actual[key]
    _add_action_normalizations(res)

    # ===== 4) 可视化 =====
    if save_vis:
        util_visual.plot_scenario_with_driving_corridor(
            corridors[0],
            0,
            reach_if,
            save_gif=False,
        )

    # ===== 5) 保存结果 =====
    if save_res:
        pkl_path = out_dir / "scene_complexity.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(res, f)
        print(f"[{name_scenario}] 结果已保存 -> {pkl_path}")

    return res


# --------------------------------------------------------------------------
# 4) CLI
# --------------------------------------------------------------------------
def _parse_args():
    parser = argparse.ArgumentParser(
        description="Compute scene complexity and visualize reachable set."
    )
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        help="Path (absolute or relative) to a CommonRoad scenario directory or XML.",
    )
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="Skip reachable set visualization.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving result files.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs/scenarios",
        help="Root directory where results will be stored.",
    )
    parser.add_argument(
        "--n-step",
        type=int,
        default=N_STEP,
        help="Planning horizon in discrete steps. This value is propagated to terminal sampling and action evaluation.",
    )
    parser.add_argument(
        "--action-params-json",
        type=str,
        default=None,
        help="Optional JSON with calibrated action params: w_goal, n_soc, b_l, r_coeff, v_ref.",
    )
    parser.add_argument(
        "--action-gp-target",
        choices=["cumulative", "incremental"],
        default="cumulative",
        help="Fit GP to cumulative S_total or per-step incremental dS.",
    )
    parser.add_argument(
        "--dp-mode",
        choices=["legacy_current", "current", "incremental_physical", "time_velocity"],
        default="legacy_current",
        help="DP search mode for least-action path.",
    )
    parser.add_argument("--dp-k", type=int, default=80, help="Number of low-cost nodes kept per time step.")
    parser.add_argument("--dp-samples", type=int, default=2000, help="Random samples per time step before top-K pruning.")
    parser.add_argument("--dp-seed", type=int, default=2025, help="Random seed for DP node sampling.")
    parser.add_argument("--eta-pc-residual", type=float, default=0.0, help="Analytic path-conflict residual multiplier for time_velocity DP.")
    parser.add_argument("--rho-conflict-drive", type=float, default=0.0, help="Conflict gate multiplier that weakens drive reward inside path-conflict windows.")
    parser.add_argument("--lambda-acc", type=float, default=0.5, help="Longitudinal acceleration regularization for time_velocity DP.")
    parser.add_argument("--hold-margin", type=float, default=3.0, help="Meters before a conflict interval used for yielding support profiles.")
    parser.add_argument("--yield-delays", type=str, default="0.5,1.0,1.5,2.0,3.0", help="Comma-separated yielding delays in seconds.")
    return parser.parse_args()


def main_cli():
    args = _parse_args()
    compute_scene_complexity_for_scenario(
        scenario_path=args.scenario,
        save_vis=not args.no_vis,
        save_res=not args.no_save,
        output_root=args.output_root,
        n_step=args.n_step,
        action_params_json=args.action_params_json,
        action_gp_target=args.action_gp_target,
        dp_mode=args.dp_mode,
        dp_k=args.dp_k,
        dp_samples=args.dp_samples,
        dp_seed=args.dp_seed,
        eta_pc_residual=args.eta_pc_residual,
        rho_conflict_drive=args.rho_conflict_drive,
        lambda_acc=args.lambda_acc,
        hold_margin=args.hold_margin,
        yield_delays=args.yield_delays,
    )


if __name__ == "__main__":
    main_cli()
