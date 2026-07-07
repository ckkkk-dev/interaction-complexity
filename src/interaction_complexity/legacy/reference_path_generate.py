import os
import pandas as pd
import re
import numpy as np
from commonroad.scenario.lanelet import LaneletNetwork
import commonroad_dc.pycrccosy as pycrccosy
from collections import Counter
from typing import List

# 从原始数据中提取dynamic obstacle的指标
def get_obstacle_trajectory_and_start_v(
    name_scenario: str,
    obstacle_id: int,
    sind_master_root: str | None = None,
) -> dict:

    # 1) 提取 event code 和 start_frame
    m_event = re.search(r"CHN_SIND-([0-9]+)_", name_scenario)
    m_frame = re.search(r"CHN_SIND-[0-9]+_([0-9]+)_", name_scenario)
    if not m_event or not m_frame:
        raise ValueError(f"无法解析 name_scenario: {name_scenario}")
    ev   = m_event.group(1)      # e.g. "821" or "8111"
    sf   = int(m_frame.group(1)) # e.g. 219
    subfolder = f"{ev[0]}_{ev[1:-1]}_{ev[-1]}"

    # 2) CSV 路径
    if sind_master_root is None:
        sind_master_root = os.environ.get("SIND_MASTER_ROOT")
    if sind_master_root is None:
        raise FileNotFoundError(
            "SIND_MASTER_ROOT is required to load raw SinD track CSV files. "
            "Set it only if you need raw-track reference-path utilities."
        )
    csv_path = os.path.join(sind_master_root, subfolder, "Veh_smoothed_tracks.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"找不到文件: {csv_path}")

    # 3) 读取并筛选
    df = pd.read_csv(csv_path)
    track_id = obstacle_id - 10000
    df_obs = df[df["track_id"] == track_id].copy()
    if df_obs.empty:
        raise ValueError(f"No data for obstacle_id {obstacle_id} (track_id {track_id})")

    # 4) 按 frame 排序
    df_obs.sort_values("frame_id", inplace=True)
    df_obs.reset_index(drop=True, inplace=True)

    # 5) 构造 trajectory 字典（所有列转 ndarray）
    trajectory = {col: df_obs[col].to_numpy() for col in df_obs.columns}

    # 6) 找到离 start_frame 最近的一行，取其 v_lon
    frames = trajectory["frame_id"]
    idx = np.abs(frames - sf).argmin()
    start_v_lon = float(trajectory["v_lon"][idx])
    first_x     = trajectory["x"][idx]
    first_y     = trajectory["y"][idx]
    return {
        "trajectory": trajectory,
        "start_v_lon": start_v_lon,
        "start_x":     first_x,
        "start_y":     first_y,
    }

# 根据dynamic obstacle的轨迹，遍历其经过的lanelet ID，用于构建ref path
def disambiguate_lanelets(lanelet_lists: List[List[int]]):
    N = len(lanelet_lists)
    # 统计全局频次备用（交集空时用）
    all_ids = [lid for sub in lanelet_lists for lid in sub]
    global_freq = Counter(all_ids)

    # 1) 先初始化 result，只有单候选帧有值，多候选帧设为 None
    result = []
    for c in lanelet_lists:
        if len(c) == 1:
            result.append(c[0])
        else:
            result.append(None)

    # 2) 找到所有连续的 None 区间
    t = 0
    while t < N:
        if result[t] is None:
            start = t
            while t < N and result[t] is None:
                t += 1
            end = t  # [start, end) 区间全是 None，需要处理

            # 3) 计算交集
            inter = set(lanelet_lists[start])
            for k in range(start+1, end):
                inter &= set(lanelet_lists[k])

            if inter:
                # 有交集，就用交集中的任意一个（或根据频次再选）
                # 这里选全局频次最高的
                pick = max(inter, key=lambda x: global_freq[x])
            else:
                # 交集空：退回到该区间内出现次数最多的
                sub_ids = [lid for k in range(start, end) for lid in lanelet_lists[k]]
                pick = Counter(sub_ids).most_common(1)[0][0]

            # 把整个区间都赋值为 pick
            for k in range(start, end):
                result[k] = pick
        else:
            t += 1

    return result
def unique_in_order(seq: List[int]) -> List[int]:

    if not seq:
        return []
    out = [seq[0]]
    for x in seq[1:]:
        if x != out[-1]:
            out.append(x)
    return out

# 基于dynamicic obstacle的行驶轨迹构建其曲线坐标系的clcs对象，用于PDM轨迹规划
def build_reference_path_and_clcs(
    ln: LaneletNetwork,
    ref_path_id: list[int],
    limit_projection_domain: float = 40.0,
    eps: float = 0.1,
    eps2: float = 1e-4,
):

    segments = []
    for lid in ref_path_id:
        lanelet = ln.find_lanelet_by_id(lid)
        if lanelet is None:
            raise KeyError(f"lanelet {lid} 不存在于网络中")
        pts = lanelet.center_vertices  # shape=(Ni,2)
        segments.append(pts)
    # 为了避免拐点重复，把后面每段的第一个点去掉
    reference_path = segments[0]
    for seg in segments[1:]:
        # 如果上段末尾与本段开头很接近，就跳过第一个点
        if np.allclose(reference_path[-1], seg[0], atol=1e-6):
            seg = seg[1:]
        reference_path = np.vstack([reference_path, seg])
    
    # 构造 CLCS
    CLCS = pycrccosy.CurvilinearCoordinateSystem(
        reference_path,
        limit_projection_domain,
        eps,
        eps2
    )
    CLCS.compute_and_set_curvature()
    return reference_path, CLCS

def cartesian_to_frenet(
    CLCS: pycrccosy.CurvilinearCoordinateSystem,
    xy_points: np.ndarray
) -> np.ndarray:
    sl = np.zeros((len(xy_points), 2))
    for i, (x,y) in enumerate(xy_points):
        s_i, l_i = CLCS.convert_to_curvilinear_coords(float(x), float(y))
        sl[i,0] = s_i
        sl[i,1] = l_i
    return sl
