import numpy as np
from typing import List, Tuple, Dict

def _infer_last_corridor_step(corridors, max_probe: int = 1000) -> int:
    last_step = None
    for k in range(max_probe + 1):
        try:
            nodes = corridors[0].reach_nodes_at_step(k)
        except Exception:
            break
        if not nodes:
            break
        last_step = k
    if last_step is None:
        raise ValueError("No reachable corridor nodes found.")
    return last_step


def _corridor_step_areas(corridors, max_step=None) -> np.ndarray:
    """返回 shape (N_step,)，每帧 corridor 的面积和。"""
    if max_step is None:
        max_step = _infer_last_corridor_step(corridors)
    main = corridors[0]                     # 只用最内层 corridor
    areas = np.zeros(max_step+1, dtype=float)
    for k in range(max_step+1):
        A = 0.0
        for node in main.reach_nodes_at_step(k):
            A += (node.p_lon_max - node.p_lon_min) * (node.p_lat_max - node.p_lat_min)
        areas[k] = A
    return areas

# def compute_scene_complexity(
#     corridors_ideal, corridors_real,
#     offset_dict,
#     w_area:   float = 0.7,
#     w_action: float = 0.3,
#     eps: float = 1e-6
# ) -> dict:
#     """
#     返回：
#       {
#         'area_penalty'   : ndarray(N,),
#         'action_penalty' : ndarray(N,),
#         'complexity'     : ndarray(N,),
#         'overall_score'  : float
#       }
#     """
#     # ---------- 1) 逐帧面积 ----------------------------
#     A_ideal = _corridor_step_areas(corridors_ideal)
#     A_real  = _corridor_step_areas(corridors_real)
#     # 防止除零
#     area_penalty = 1.0 - np.divide(A_real, A_ideal+eps)
#     # ---------- 2) 逐帧作用量增量 ----------------------
#     delta_S   = offset_dict['offset']            # S_ideal - S_best, shape (N,)
#     S_best    = offset_dict['S_best']
#     action_penalty = delta_S / (np.abs(S_best) + eps)
#     # ---------- 3) 合成逐帧复杂度 ----------------------
#     complexity = w_area*area_penalty + w_action*action_penalty

#     # ---------- 4) 场景 30 帧总体复杂度 -----------------
#     overall = complexity.mean()          # 也可加权

#     return dict(area_penalty   = area_penalty,
#                 action_penalty = action_penalty,
#                 complexity     = complexity,
#                 overall_score  = overall)


def compute_scene_complexity(
    corridors_ideal,
    corridors_real,
    opt_path_ideal: List[Tuple[float,float]],
    opt_path_actual: List[Tuple[float,float]],
    gp_models: List,
    N_STEP,
    eval_ideal_on_actual_field,      # 你已有的函数
    compute_action_offset,           # 你已有的函数
    w_area: float   = 0.7,
    w_action: float = 0.3,
    eps: float      = 1e-6
) -> Dict[str, np.ndarray]:
    """
    计算场景复杂度：
      - area_penalty[k]   = 1 - A_real[k]/A_ideal[k]
      - action_penalty[k] = (S_ideal[k] - S_best[k]) / (|S_best[k]|+eps)
      - complexity[k]     = w_area*area_penalty + w_action*action_penalty
    返回 dict 包含：
      'area_penalty','action_penalty','complexity' (ndarrays of shape (N,)),
      'overall_score' (float)
    """
    # —— 1) 格式化理想/实际最优轨迹为 dict{s,l} —— #
    def _traj_dict(path: List[Tuple[float,float]]):
        arr = np.array(path)
        return {'s': arr[:,0], 'l': arr[:,1]}
    traj_ideal = _traj_dict(opt_path_ideal)
    traj_best  = _traj_dict(opt_path_actual)

    # —— 2) 在实际场中评估理想轨迹作用量 S_ideal —— #
    #    假定 eval_ideal_on_actual_field(traj_ideal, gp_models) → ndarray shape (N,)
    S_ideal = eval_ideal_on_actual_field(traj_ideal, gp_models)

    # —— 3) 计算 offset_dict 包括 S_best 和 ΔS = S_ideal-S_best —— #
    offset_dict = compute_action_offset(
        mu_ideal        = S_ideal,
        opt_path_actual = opt_path_actual
    )

    # —— 4) 逐帧面积 —— #
    A_ideal = _corridor_step_areas(corridors_ideal, max_step=N_STEP)
    A_real  = _corridor_step_areas(corridors_real, max_step=N_STEP)
    area_penalty = 1.0 - np.divide(A_real, A_ideal + eps)

    # —— 5) 逐帧作用量恶化 —— #
    delta_S       = offset_dict['offset']     # S_ideal - S_best
    S_best        = offset_dict['S_best']
    action_penalty = delta_S / (np.abs(S_best) + eps)

    # —— 6) 合成复杂度 & 整体得分 —— #
    complexity   = w_area*area_penalty + w_action*action_penalty
    overall_score = complexity.mean()

    return {
        'area_penalty':   area_penalty,
        'action_penalty': action_penalty,
        'complexity':     complexity,
        'overall_score':  overall_score,
        'delta_S':        delta_S,
        'S_best':         S_best,
        'S_ideal':        offset_dict['mu_ideal'],
    }
