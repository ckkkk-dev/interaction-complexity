from enum import Enum
import numpy as np

class ObstacleType(Enum):
    BUS = 1
    CAR = 2
    BICYCLE = 3
    PEDESTRIAN = 4
    TRUCK = 5
    MOTORCYCLE = 6

# 以及事先计算好的虚拟质量字典
OBSTACLE_VIRTUAL_MASS = {
    ObstacleType.BUS:       5000.0 * 0.3,
    ObstacleType.CAR:       1500.0 * 1,
    ObstacleType.BICYCLE:     80.0 * 4.0,
    ObstacleType.PEDESTRIAN:  75.0 * 5.0,
    ObstacleType.TRUCK:       5000.0 * 0.3,
    ObstacleType.MOTORCYCLE:  80.0 * 4.0,

}

# 1. 纵向法规阻力 R_i(v_x) 和 驱动力 G_x
def R_longitudinal(vx, v_lim, r_coeff=20.0):
    """限速惩罚：超过限速时按二次惩罚，否则 0"""
    excess = np.maximum(vx - v_lim, 0.0)
    return float(r_coeff) * excess**2          # 系数可按场景微调

def G_x(m_i, g, n_g, v_des, v_lim):
    """驱动力常量项（纵向）"""
    return m_i * g * n_g * (v_des / v_lim)   # 标量

# 2. 车道弹簧力 F_l1 / F_l2
def lane_spring(l, m_i, l_w, B_l, l0):
    """
    车道弹簧约束力（始终作用）：
      - l:  当前横向偏移，l>l0 为向左偏离，l<l0 为向右偏离
      - l0: 中心线（平衡位置）
      - B_l: 弹簧刚度系数
    返回 (F_left, F_right):
      * F_left > 0  → 车需要往右纠正（说明实际力来自左弹簧）
      * F_right > 0 → 车需要往左纠正（说明实际力来自右弹簧）
    """
    # 平衡偏移点
    delta = l - l0   # >0 向左偏； <0 向右偏
    # 等效弹簧刚度
    k = 1.5 * m_i * B_l

    # 恢复力总量（正时往右推，负时往左推）
    F = -k * delta

    # 分拆成左右两侧弹簧力分量
    # 如果 F>0，就当作“左侧弹簧”在推 → F_left=F；否则当作右弹簧在推 → F_right=-F
    F_left  = F if F > 0 else 0.0
    F_right = -F if F < 0 else 0.0

    return F_left, F_right

# 3. 周车→自车风险斥力 F_{j→i}
def axis_ttc_force(dr, dv, m_neigh,  # 轴向相对位移·速度·质量
                   k=1.0, T0=2.0, r_safe=1.0, eps=1e-6):
    """
    单轴（1D）TTC 斥力标量：
      dr : 相对位移 (m)  (前正 / 上正)
      dv : 相对速度 (m/s) (邻车 - 自车, 前正 / 上正)
      m_neigh : 邻车质量 (kg)      —— 可用虚拟质量权重
    返回 F_axis (向远离对方的正标量；若不逼近→0)
    """
    # ─── 1. “逼近”判定 ───
    approaching = (dr > 0 and dv < 0) or (dr < 0 and dv > 0)
    if not approaching:
        return 0.0

    # ─── 2. Time-to-Collision (TTC) ───
    vr  = abs(dv)                # 闭合速度（正）
    ttc = abs(dr) / (vr + eps)   # 避免除 0

    # ─── 3. 斥力幅值（随 TTC 指数衰减）───
    #     这里附上一点质量权重，可选
    F = m_neigh * k * np.exp(-(ttc - r_safe) / T0)
    return F        # 方向后续再处理
# ------------------------------------------------------

def social_force_split_axes(ego, neigh,
                            k=1.0, T0=2.0, r_safe=1.0, eps=1e-6,
                            front_weight=1.0, back_weight=1.0,
                            lateral_weight=1.0):
    """
    • ego / neigh : dict {'s','l','vx','vy','m'}
    • 返回 F_x, F_y （在曲线坐标系的正 s、正 l 方向）
    """
    # 相对量（邻车 - 自车）
    dx = neigh['s']  - ego['s']
    dy = neigh['l']  - ego['l']
    dvx = neigh['vx'] - ego['vx']
    dvy = neigh['vy'] - ego['vy']

    # 纵向力
    F_s = axis_ttc_force(dx, dvx, neigh['m'],
                         k=k, T0=T0, r_safe=r_safe, eps=eps)
    F_s *= front_weight if dx >= 0 else back_weight
    # 方向：若邻车在前( dx>0) 且逼近 → 推自己向 -s；反之向 +s
    if dx > 0:
        F_s = -F_s

    # 横向力
    F_l = axis_ttc_force(dy, dvy, neigh['m'],
                         k=k, T0=T0, r_safe=r_safe, eps=eps)
    F_l *= lateral_weight
    # 方向：邻车在左( dy>0) 且逼近 → 推自己向 -l；反之向 +l
    if dy > 0:
        F_l = -F_l

    return F_s, F_l


DEFAULT_SOCIAL_CONFIG = {
    "filter_mode": "reachable",
    "coupling": "relative_velocity",
    "k": 1.0,
    "T0": 2.0,
    "r_safe": 1.0,
    "front_weight": 1.0,
    "back_weight": 1.0,
    "lateral_weight": 1.0,
    "v_floor_s": 0.0,
    "v_floor_l": 0.0,
    "slack_back": 10.0,
    "front_s_window": 45.0,
    "back_s_window": 10.0,
    "lat_window": 8.0,
    "goal_s_window": 25.0,
    "goal_l_window": 12.0,
    "path_conflict_enabled": False,
    "path_tube_width": 2.0,
    "conflict_s_margin": 4.0,
    "sigma_l": 1.2,
    "tau_safe": 1.0,
    "beta_time": 0.5,
    "lambda_path_conflict": 1.0,
    "w_overlap": 2.0,
    "ped_bike_weight": 2.0,
    "vehicle_weight": 1.0,
    "crossing_speed_scale": 1.0,
    "sigma_s": 4.0,
    "path_conflict_mode": "event_lump",
    "rho_conflict_drive": 0.0,
    "action_field_component_scaled": False,
    "action_field_alpha": 1.0,
    "action_field_scale_physical": 1.0,
    "action_field_scale_pc": 1.0,
    "action_field_clip": 5.0,
}



def normalize_social_config(social_config=None):
    out = dict(DEFAULT_SOCIAL_CONFIG)
    if social_config:
        for key, value in social_config.items():
            if key in out:
                out[key] = value
    return out


def social_cost_contribution(ego, neigh, n_soc, social_config=None):
    cfg = normalize_social_config(social_config)
    F_s, F_l = social_force_split_axes(
        ego,
        neigh,
        k=float(cfg["k"]),
        T0=float(cfg["T0"]),
        r_safe=float(cfg["r_safe"]),
        front_weight=float(cfg["front_weight"]),
        back_weight=float(cfg["back_weight"]),
        lateral_weight=float(cfg["lateral_weight"]),
    )
    if cfg["coupling"] == "ego_speed":
        v_s = max(abs(float(ego["vx"])), float(cfg["v_floor_s"]))
        v_l = max(abs(float(ego["vy"])), float(cfg["v_floor_l"]))
        cost = abs(F_s) * v_s + abs(F_l) * v_l
    elif cfg["coupling"] == "relative_velocity":
        ds_rel = abs(float(neigh["vx"]) - float(ego["vx"]))
        dl_rel = abs(float(neigh["vy"]) - float(ego["vy"]))
        cost = abs(F_s) * ds_rel + abs(F_l) * dl_rel
    else:
        raise ValueError(f"Unknown social coupling mode: {cfg['coupling']}")
    return float(cost) * float(n_soc), float(F_s), float(F_l)


def _path_conflict_type_weight(type_name, social_config):
    name = str(type_name).upper()
    is_small = int("PEDESTRIAN" in name or "BICYCLE" in name)
    if is_small:
        return float(social_config["ped_bike_weight"]), is_small
    return float(social_config["vehicle_weight"]), is_small


def build_path_conflict_events(config, N_step, social_config=None):
    cfg = normalize_social_config(social_config)
    if not bool(cfg.get("path_conflict_enabled", False)):
        return []
    dt = float(config.planning.dt)
    clcs = config.planning.CLCS
    events = []
    for obs in config.scenario.dynamic_obstacles:
        rows = []
        last_s = None
        last_l = None
        for k in range(int(N_step) + 1):
            occ = obs.occupancy_at_time(k)
            if occ is None:
                rows.append(None)
                last_s = None
                last_l = None
                continue
            try:
                x, y = occ.shape.center
                s_val, l_val = clcs.convert_to_curvilinear_coords(float(x), float(y))
            except Exception:
                rows.append(None)
                last_s = None
                last_l = None
                continue
            if last_s is None or last_l is None:
                v_l = 0.0
            else:
                v_l = (float(l_val) - last_l) / dt
            last_s = float(s_val)
            last_l = float(l_val)
            rows.append({"k": k, "s": float(s_val), "l": float(l_val), "v_l": float(v_l)})

        in_event = False
        start = 0
        for idx in range(len(rows) + 1):
            row = rows[idx] if idx < len(rows) else None
            inside = row is not None and abs(float(row["l"])) <= float(cfg["path_tube_width"])
            if inside and not in_event:
                start = idx
                in_event = True
            if in_event and (not inside or idx == len(rows)):
                segment = [r for r in rows[start:idx] if r is not None]
                in_event = False
                if not segment:
                    continue
                s_vals = np.asarray([r["s"] for r in segment], dtype=float)
                l_vals = np.asarray([r["l"] for r in segment], dtype=float)
                vl_vals = np.asarray([r["v_l"] for r in segment], dtype=float)
                type_name = getattr(obs.obstacle_type, "name", str(obs.obstacle_type))
                type_weight, is_small = _path_conflict_type_weight(type_name, cfg)
                l_min_abs = float(np.min(np.abs(l_vals)))
                occupancy_strength = float(np.exp(-(l_min_abs**2) / (2.0 * float(cfg["sigma_l"]) ** 2)))
                crossing_speed = float(np.nanmedian(np.abs(vl_vals)))
                crossing_weight = 1.0 + float(cfg["crossing_speed_scale"]) * min(crossing_speed, 5.0) / 5.0
                events.append(
                    {
                        "event_id": len(events),
                        "obstacle_id": int(obs.obstacle_id),
                        "obstacle_type": str(type_name),
                        "is_small_actor": int(is_small),
                        "t_enter": float(segment[0]["k"]) * dt,
                        "t_exit": float(segment[-1]["k"] + 1) * dt,
                        "s_min": float(np.min(s_vals) - float(cfg["conflict_s_margin"])),
                        "s_max": float(np.max(s_vals) + float(cfg["conflict_s_margin"])),
                        "l_min_abs": l_min_abs,
                        "crossing_speed": crossing_speed,
                        "occupancy_strength": occupancy_strength,
                        "type_weight": type_weight,
                        "crossing_weight": crossing_weight,
                    }
                )
    return events


def path_conflict_cost_array(traj, path_conflict_events, dt, n_soc, social_config=None):
    cfg = normalize_social_config(social_config)
    social = np.zeros(len(traj["s"]), dtype=float)
    if not path_conflict_events:
        return social
    s_arr = np.asarray(traj["s"], dtype=float)
    mode = str(cfg.get("path_conflict_mode", "event_lump"))
    if mode == "per_step":
        times = np.arange(len(s_arr), dtype=float) * float(dt)
        for event in path_conflict_events:
            dist_before = float(event["s_min"]) - s_arr
            dist_after = s_arr - float(event["s_max"])
            dist_to_interval = np.maximum(np.maximum(dist_before, dist_after), 0.0)
            space_gate = np.exp(-(dist_to_interval**2) / (2.0 * max(float(cfg["sigma_s"]), 1e-6) ** 2))

            time_gap = np.zeros(len(s_arr), dtype=float)
            before = times < float(event["t_enter"])
            after = times > float(event["t_exit"])
            time_gap[before] = float(event["t_enter"]) - times[before]
            time_gap[after] = times[after] - float(event["t_exit"])
            temporal_gate = np.logaddexp(
                0.0,
                (float(cfg["tau_safe"]) - time_gap) / max(float(cfg["beta_time"]), 1e-6),
            )

            in_event_s = (s_arr >= float(event["s_min"])) & (s_arr <= float(event["s_max"]))
            in_event_t = (times >= float(event["t_enter"])) & (times <= float(event["t_exit"]))
            temporal_gate = temporal_gate + float(cfg["w_overlap"]) * (in_event_s & in_event_t).astype(float)

            social += (
                float(cfg["lambda_path_conflict"])
                * float(n_soc)
                * float(event["occupancy_strength"])
                * float(event["type_weight"])
                * float(event["crossing_weight"])
                * space_gate
                * temporal_gate
            )
        return social
    if mode != "event_lump":
        raise ValueError(f"Unknown path_conflict_mode={mode}")
    for event in path_conflict_events:
        in_zone = (s_arr >= float(event["s_min"])) & (s_arr <= float(event["s_max"]))
        idx = np.flatnonzero(in_zone)
        if idx.size == 0:
            continue
        enter_idx = int(idx[0])
        exit_idx = int(idx[-1])
        t_ego_enter = float(enter_idx) * float(dt)
        t_ego_exit = float(exit_idx + 1) * float(dt)
        overlap = max(0.0, min(t_ego_exit, float(event["t_exit"])) - max(t_ego_enter, float(event["t_enter"])))
        clearance = t_ego_enter - float(event["t_exit"])
        clearance_loss = float(np.logaddexp(0.0, (float(cfg["tau_safe"]) - clearance) / max(float(cfg["beta_time"]), 1e-6)))
        event_cost = (
            float(cfg["lambda_path_conflict"])
            * float(n_soc)
            * float(event["occupancy_strength"])
            * float(event["type_weight"])
            * float(event["crossing_weight"])
            * (float(cfg["w_overlap"]) * overlap + clearance_loss)
        )
        social[enter_idx] += event_cost / max(float(dt), 1e-6)
    return social


def path_conflict_potential_at_points(points, step_idx, path_conflict_events, dt, n_soc, social_config=None):
    """Pointwise per-step path-conflict potential for DP lattice nodes.

    Returns a cost rate with the same unit as path_conflict_cost_array values;
    callers multiply by dt when adding it to incremental action.
    """
    cfg = normalize_social_config(social_config)
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return np.zeros(0, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    social = np.zeros(len(pts), dtype=float)
    if not path_conflict_events:
        return social
    s_arr = pts[:, 0]
    time = float(step_idx) * float(dt)
    for event in path_conflict_events:
        dist_before = float(event["s_min"]) - s_arr
        dist_after = s_arr - float(event["s_max"])
        dist_to_interval = np.maximum(np.maximum(dist_before, dist_after), 0.0)
        space_gate = np.exp(-(dist_to_interval**2) / (2.0 * max(float(cfg["sigma_s"]), 1e-6) ** 2))

        if time < float(event["t_enter"]):
            time_gap = float(event["t_enter"]) - time
        elif time > float(event["t_exit"]):
            time_gap = time - float(event["t_exit"])
        else:
            time_gap = 0.0
        temporal_gate = float(np.logaddexp(0.0, (float(cfg["tau_safe"]) - time_gap) / max(float(cfg["beta_time"]), 1e-6)))

        in_event_s = (s_arr >= float(event["s_min"])) & (s_arr <= float(event["s_max"]))
        in_event_t = float(event["t_enter"]) <= time <= float(event["t_exit"])
        gate = temporal_gate + float(cfg["w_overlap"]) * (in_event_s & in_event_t).astype(float)
        social += (
            float(cfg["lambda_path_conflict"])
            * float(n_soc)
            * float(event["occupancy_strength"])
            * float(event["type_weight"])
            * float(event["crossing_weight"])
            * space_gate
            * gate
        )
    return social


def filter_neighbors_for_social(
    neighbors_by_t,
    reach_interface,
    p_initial,
    s_goal=None,
    l_goal=None,
    social_config=None,
):
    cfg = normalize_social_config(social_config)
    reachable_oids = obstacles_within_reachable(
        neighbors_by_t,
        reach_interface,
        slack_back=float(cfg["slack_back"]),
    )
    if cfg["filter_mode"] in {"reachable", "legacy"}:
        keep_oids = set(reachable_oids)
    elif cfg["filter_mode"] == "union":
        keep_oids = set(reachable_oids)
        p_initial = np.asarray(p_initial, dtype=float)
        s0 = float(p_initial[0])
        l0 = float(p_initial[1])
        for neighs in neighbors_by_t:
            for nb in neighs:
                oid = nb.get("oid")
                ds0 = float(nb["s"]) - s0
                dl0 = float(nb["l"]) - l0
                near_ego_corridor = (
                    -float(cfg["back_s_window"]) <= ds0 <= float(cfg["front_s_window"])
                    and abs(dl0) <= float(cfg["lat_window"])
                )
                near_goal = False
                if s_goal is not None and l_goal is not None:
                    near_goal = (
                        abs(float(nb["s"]) - float(s_goal)) <= float(cfg["goal_s_window"])
                        and abs(float(nb["l"]) - float(l_goal)) <= float(cfg["goal_l_window"])
                    )
                if near_ego_corridor or near_goal:
                    keep_oids.add(oid)
    else:
        raise ValueError(f"Unknown social filter mode: {cfg['filter_mode']}")

    filtered = [[nb for nb in neighs if nb.get("oid") in keep_oids] for neighs in neighbors_by_t]
    return filtered, keep_oids, set(reachable_oids)

# ======================================================
# 构造 neighbors_by_t
# ======================================================

def nearest_segment_and_projection(p, polyline):
    """
    给定点 p=[x,y]，在折线 polyline (M×2) 上找到最近投影点
    返回： seg_idx, proj_pt, t_hat, n_hat
        seg_idx   最近线段索引 i（段 [i,i+1]）
        proj_pt   投影点坐标 [x*,y*]
        t_hat     该段单位切向量 (s 方向，指向 poly[i+1])
        n_hat     左侧单位法向量 (l 方向，车道左为 +)
    """
    # 逐段投影，找最小距离
    min_d2 = 1e9
    best   = None
    Px,Py  = p
    for i in range(len(polyline)-1):
        A = polyline[i]
        B = polyline[i+1]
        AB = B-A
        t  = np.clip(np.dot([Px,Py]-A, AB) / (np.dot(AB,AB)+1e-12), 0.0, 1.0)
        proj = A + t*AB
        d2 = np.sum((proj - p)**2)
        if d2 < min_d2:
            min_d2 = d2
            t_hat  = AB / (np.linalg.norm(AB)+1e-12)
            n_hat  = np.array([-t_hat[1], t_hat[0]])    # 左手垂直
            best   = (i, proj, t_hat, n_hat)
    return best      # seg_idx, proj_pt, t_hat, n_hat

def cartesian_velocity_to_curvilinear(pos_xy, vel_xy, path_xy):
    """
    给自车或周车的 (x,y) 与 (vx,vy)，输出 (v_s, v_l)。
    """
    _, proj, t_hat, n_hat = nearest_segment_and_projection(pos_xy, path_xy)
    vx, vy = vel_xy
    v_vec  = np.array([vx, vy])
    v_s = np.dot(v_vec, t_hat)   # 沿切线分量
    v_l = np.dot(v_vec, n_hat)   # 沿法线分量
    return v_s, v_l

def cartesian_to_curvi_safe(pos_xy, clcs):
    """
    安全调用 convert_to_curvilinear_coords，
    返回 (s,l) 或 (None,None)。
    """
    try:
        # 注意：clcs.convert_to_curvilinear_coords 接受两个 floats
        sx, ly = clcs.convert_to_curvilinear_coords(float(pos_xy[0]), 
                                                    float(pos_xy[1]))
        return sx, ly
    except:
        return None, None

def generate_neighbors_by_t(config, T_h = 3.0):
    path_xy = np.asarray(config.planning.reference_path)
    clcs    = config.planning.CLCS
    dt = config.planning.dt
    Nstep = int(round(T_h / dt)) + 1
    neighbors_by_t = [[] for _ in range(Nstep)]
    pos_history   = {}

    for obs in config.scenario.dynamic_obstacles:
        typ    = ObstacleType[obs.obstacle_type.name]
        virt_m = OBSTACLE_VIRTUAL_MASS[typ]
        oid    = obs.obstacle_id
        pos_history[oid] = None

        for k in range(Nstep):
            oc = obs.occupancy_at_time(k)
            if oc is None:
                pos_history[oid] = None
                continue

            # 当前笛卡尔位置
            pos_xy = np.array(oc.shape.center, dtype=float)

            # 差分速度
            prev = pos_history[oid]
            if prev is None:
                vel_vec = np.zeros(2, dtype=float)
            else:
                vel_vec = (pos_xy - prev) / dt
            pos_history[oid] = pos_xy

            # 曲线坐标速度分量
            v_s, v_l = cartesian_velocity_to_curvilinear(pos_xy, vel_vec, path_xy)

            # 安全转换到曲线坐标
            s_val, l_val = cartesian_to_curvi_safe(pos_xy, clcs)
            if s_val is None:
                # 超出 CLCS 支持范围，跳过
                continue

            neighbors_by_t[k].append({
                'oid': obs.obstacle_id,
                's':  s_val,
                'l':  l_val,
                'vx': v_s,
                'vy': v_l,
                'm':  virt_m,
            })
    return neighbors_by_t

def compute_S_total_for_traj(
    traj,
    neighbors_by_t,
    s_goal: float,
    l_goal: float,
    m_i: float,
    l_w: float,
    B_l: float,
    l_init: float,
    n_soc: float,
    w_goal_pos: float,
    dt: float, g, n_g, v_des, v_lim,
    r_coeff: float = 20.0,
    return_incremental: bool = False,
    social_config=None,
    path_conflict_events=None,
):

    s_arr, l_arr   = traj['s'], traj['l']
    vx_arr, vy_arr = traj['v_x'], traj['v_y']
    N       = len(traj['s'])

    # 存各步的 Lagrangian 和 penalty_goal_t
    L_arr        = np.zeros(N)
    penalty_t    = np.zeros(N)

    Gx = G_x(m_i, g, n_g, v_des, v_lim)

    # —— 1) 先算出每步的 L_arr[t] —— #
    #    U_long, U_lat, U_soc
    U_long = np.zeros(N)
    U_lat  = np.zeros(N)
    U_soc  = np.zeros(N)

    cfg = normalize_social_config(social_config)
    use_path_conflict = bool(cfg.get("path_conflict_enabled", False))
    path_social = path_conflict_cost_array(traj, path_conflict_events or [], dt, n_soc, cfg) if use_path_conflict else np.zeros(N)
    rho_conflict_drive = float(cfg.get("rho_conflict_drive", 0.0)) if use_path_conflict else 0.0
    if rho_conflict_drive > 0.0 and np.any(path_social > 0.0):
        positive_social = path_social[path_social > 0.0]
        pc_scale = float(np.nanpercentile(positive_social, 75)) if positive_social.size else 1.0
        pc_gate = np.clip(path_social / max(pc_scale, 1e-6), 0.0, 1.0)
    else:
        pc_gate = np.zeros(N, dtype=float)

    for t in range(1, N):
        # 纵向势能项
        R_i        = R_longitudinal(vx_arr[t], v_lim, r_coeff=r_coeff)
        Gx_t = Gx * (1.0 - rho_conflict_drive * pc_gate[t])
        U_long[t]  = (R_i - Gx_t) * vx_arr[t]

        # 横向势能项
        F_left, F_right = lane_spring(
            l   = l_arr[t],
            m_i = m_i,
            l_w = l_w,
            B_l = B_l,
            l0  = l_init
        )
        U_lat[t] = (F_left - F_right) * vy_arr[t]

        # 社交势能项
        sum_soc = 0.0
        ego = {'s':s_arr[t],'l':l_arr[t],'vx':vx_arr[t],'vy':vy_arr[t],'m':m_i}
        if use_path_conflict:
            sum_soc = float(path_social[t])
        else:
            neighs_t = neighbors_by_t[t] if t < len(neighbors_by_t) else []
            for nb in neighs_t:
                soc_cost, _, _ = social_cost_contribution(ego, nb, n_soc, social_config)
                sum_soc += soc_cost
        U_soc[t] = sum_soc

        # 拉格朗日量 L = T - (U_long + U_lat - U_soc)
        Tkin      = 0.5 * m_i * (vx_arr[t]**2 + vy_arr[t]**2)
        L_arr[t]  = Tkin - (U_long[t] + U_lat[t] - U_soc[t])

    # —— 2) 再算每步的到达目标惩罚 penalty_t[t] —— #
    #    我们这里用瞬时到达目标距离的平方乘 dt 再累加
    dist2 = (s_arr - s_goal)**2 + (l_arr - l_goal)**2
    penalty_t = w_goal_pos * dist2      # 这是一时刻的“罚项幅值”

    # —— 3) 累计到每一步的 S_total —— #
    #    S_total(t) = ∑_{k=1..t} L_arr[k]*dt  + ∑_{k=1..t} penalty_t[k]*dt
    cum_L     = np.cumsum(L_arr) * dt
    cum_pen   = np.cumsum(penalty_t) * dt
    dS       = (L_arr + penalty_t) * dt

    if bool(cfg.get("action_field_component_scaled", False)):
        scale_physical = max(float(cfg.get("action_field_scale_physical", 1.0)), 1e-6)
        scale_pc = max(float(cfg.get("action_field_scale_pc", 1.0)), 1e-6)
        alpha = float(cfg.get("action_field_alpha", 1.0))
        clip = float(cfg.get("action_field_clip", 5.0))
        # Match the v5 offline convention: physical action excludes path-conflict,
        # while path-conflict is normalized on its own train-split scale.
        physical_dS = (L_arr - U_soc + penalty_t) * dt
        pc_dS = U_soc * dt if use_path_conflict else np.zeros_like(physical_dS)
        z_physical = np.clip(physical_dS / scale_physical, -clip, clip)
        z_pc = np.clip(pc_dS / scale_pc, -clip, clip)
        dS = z_physical + alpha * z_pc

    S_total  = np.cumsum(dS)

    if return_incremental:
        return S_total, dS
    return S_total


def compute_dS_for_traj(*args, **kwargs):
    """Return cumulative action and per-step action increment for one trajectory."""
    kwargs["return_incremental"] = True
    return compute_S_total_for_traj(*args, **kwargs)

def compute_action_offset(
    mu_ideal,      # 理想轨迹在“考虑障碍物”场中的累积作用量
    opt_path_actual,
):
    # 1) 实际最优轨迹的 S_total
    S_best = np.array([opt_path_actual[i][2] for i in range(len(opt_path_actual))])  # shape (N,)
    # 2) 两者差值
    offset = mu_ideal - S_best
    return {
        'mu_ideal': mu_ideal,   # 理想轨迹在实际场中的预测累积作用量
        'S_best':   S_best,     # 实际最优轨迹的累积作用量
        'offset':   offset      # 两者之差
    }


def compute_action_offset_incremental(
    dS_ideal=None,
    opt_path_actual=None,
    mu_ideal=None,
):
    if dS_ideal is None:
        dS_ideal = mu_ideal
    dS_ideal = np.asarray(dS_ideal, dtype=float)
    dS_best = np.array([opt_path_actual[i][2] for i in range(len(opt_path_actual))], dtype=float)
    S_ideal = np.cumsum(dS_ideal)
    S_best = np.cumsum(dS_best)
    offset = S_ideal - S_best
    return {
        "mu_ideal": S_ideal,
        "dS_ideal": dS_ideal,
        "S_best": S_best,
        "dS_best": dS_best,
        "offset": offset,
    }

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF

def gaussian_process_regression(S_all, all_speed_profile):
    n_paths, N = S_all.shape
    # ============================================================
    # 1. 为每个 step 拟合一个 GP
    # ============================================================
    gp_models = []        # 存每一步的 GP
    for t in range(N):
        # -------- 1) 组织观测点 (X) 与观测值 (y) ------------------
        X_t = np.zeros((n_paths, 2))     # [s, l]
        y_t = S_all[:, t]                # S_total 的那一列
        for i, traj in enumerate(all_speed_profile):
            X_t[i, 0] = traj['s'][t]
            X_t[i, 1] = traj['l'][t]

        # -------- 2) 构造 & 训练 GP -------------------------------
        kernel = C(1.0, (1e-2, 1e3)) * RBF(length_scale=[5.0, 1.0],
                                        length_scale_bounds=(1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel,
                                    n_restarts_optimizer=5,
                                    alpha=1e-6,
                                    normalize_y=True,
                                    random_state=0)
        gp.fit(X_t, y_t)
        gp_models.append(gp)
    return gp_models


def gaussian_process_regression_incremental(dS_all, all_speed_profile):
    return gaussian_process_regression(dS_all, all_speed_profile)


from numpy.random import default_rng
from typing import List, Tuple

# ------------------------------------------------------------
# 帮助函数：在第 t 帧的 corridor 内随机均匀采样 n 个 (s,l) 点
# ------------------------------------------------------------
def sample_inside_corridor(
        corridor, step: int, n: int = 400, rng=None
) -> np.ndarray:
    """
    参数
    ----
    corridor : CommonRoad Corridor 对象（这里用 corridor[0]）
    step     : 时间步索引
    n        : 采样点数
    返回
    ----
    pts (n,2) ndarray  均匀落在所有 reach-nodes 长方形内部
    """
    rng = default_rng() if rng is None else rng
    nodes = corridor.reach_nodes_at_step(step)
    # —— 预先计算每个小矩形面积 & 累积分布 —— #
    areas = np.array([(n.p_lon_max - n.p_lon_min) *
                      (n.p_lat_max - n.p_lat_min) for n in nodes])
    prob  = areas / areas.sum()
    cdf   = np.cumsum(prob)
    # —— 采样 —— #
    pts = np.zeros((n, 2))
    for k in range(n):
        r = rng.random()
        idx = np.searchsorted(cdf, r)
        node = nodes[idx]
        s = rng.uniform(node.p_lon_min, node.p_lon_max)
        l = rng.uniform(node.p_lat_min, node.p_lat_max)
        pts[k] = [s, l]
    return pts
# ------------------------------------------------------------
# 主函数：给 gp_models & corridors 计算最低作用量平滑轨迹
# ------------------------------------------------------------
def lowest_action_path(
        gp_models: List,             # len N，高斯过程列表
        corridors,                   # corridor[0] 有 reach_nodes_at_step()
        K: int = 40,                 # 每帧保留的候选数
        lam: float = 0.05,           # 平滑权重 λ
        n_samples: int = 400,        # 每帧初始随机采样数
        seed: int = 0
) -> List[Tuple[float, float]]:
    rng = default_rng(seed)
    N = len(gp_models)

    # ---------- 1) 生成候选点 + 节点代价 -----------------------
    cands: List[List[Tuple[float, float, float]]] = []
    for t in range(N):
        pts  = sample_inside_corridor(corridors[0], t, n_samples, rng)
        mu_t = gp_models[t].predict(pts)
        idx  = np.argsort(mu_t)[:K]                    # 取前 K
        cands.append([(pts[i, 0], pts[i, 1], mu_t[i]) for i in idx])

    # ---------- 2) 动态规划 -----------------------------------
    prev_cost = np.array([c[2] for c in cands[0]])
    prev_ptr  = []           # prev_ptr[t][j] = best i at t-1
    for t in range(1, N):
        M = len(cands[t])
        cost_t = np.full(M, np.inf)
        ptr_t  = np.full(M, -1, dtype=int)
        for j, (s, l, mu) in enumerate(cands[t]):
            # 与上一帧所有 i 做转移
            for i, (sp, lp, _) in enumerate(cands[t-1]):
                trans = lam * ((s - sp) ** 2 + (l - lp) ** 2)
                val   = prev_cost[i] + mu + trans
                if val < cost_t[j]:
                    cost_t[j] = val
                    ptr_t[j]  = i
        prev_cost = cost_t
        prev_ptr.append(ptr_t)

    # ---------- 3) 回溯最优路径 -------------------------------
    t = N - 1
    j = prev_cost.argmin()
    path_with_cost: List[Tuple[float,float,float]] = []
    while t >= 0:
        s, l, mu = cands[t][j]
        path_with_cost.append((s, l, mu))
        if t > 0:
            j = prev_ptr[t-1][j]
        t -= 1

    return list(reversed(path_with_cost))


def _predict_gp(gp, pts, return_std=False):
    if return_std:
        mu, sigma = gp.predict(pts, return_std=True)
        return mu, sigma
    return gp.predict(pts), np.zeros(len(pts), dtype=float)


def _physical_transition(
    s,
    l,
    sp,
    lp,
    dt,
    lambda_l=0.8,
    v_s_max=13.9,
    v_l_max=2.67,
    backtrack_penalty=1e6,
    velocity_penalty=1e5,
):
    ds = float(s - sp)
    dl = float(l - lp)
    v_s = ds / dt
    v_l = dl / dt
    if ds < -1e-6 or v_s > v_s_max or abs(v_l) > v_l_max:
        return np.inf
    cost = float(lambda_l) * dl**2
    return cost


def _path_physical_diagnostics(path, dt, v_s_max=13.9, v_l_max=2.67):
    arr = np.asarray(path, dtype=float)
    if len(arr) < 2:
        return {
            "max_v_s": 0.0,
            "max_abs_v_l": 0.0,
            "backward_step_count": 0,
            "velocity_violation_count": 0,
        }
    ds = np.diff(arr[:, 0])
    dl = np.diff(arr[:, 1])
    v_s = ds / dt
    v_l = dl / dt
    velocity_violations = (v_s > v_s_max) | (np.abs(v_l) > v_l_max)
    return {
        "max_v_s": float(np.max(v_s)),
        "max_abs_v_l": float(np.max(np.abs(v_l))),
        "backward_step_count": int(np.sum(ds < -1e-6)),
        "velocity_violation_count": int(np.sum(velocity_violations)),
    }


def _profile_with_velocities(s_arr, l_arr, dt, tag="support"):
    s_arr = np.asarray(s_arr, dtype=float)
    l_arr = np.asarray(l_arr, dtype=float)
    return {
        "s": s_arr,
        "l": l_arr,
        "v_x": np.gradient(s_arr, dt) if len(s_arr) > 1 else np.zeros_like(s_arr),
        "v_y": np.gradient(l_arr, dt) if len(l_arr) > 1 else np.zeros_like(l_arr),
        "_support_tag": str(tag),
    }


def _make_delayed_profile(profile, event, delay_s, dt, hold_margin=3.0):
    s_orig = np.asarray(profile["s"], dtype=float)
    l_orig = np.asarray(profile["l"], dtype=float)
    n = len(s_orig)
    if n < 3:
        return None
    hold_s = float(event["s_min"]) - float(hold_margin)
    idxs = np.flatnonzero(s_orig >= hold_s)
    if idxs.size == 0:
        return None
    hold_idx = int(idxs[0])
    delay_steps = int(round(float(delay_s) / max(float(dt), 1e-6)))
    if delay_steps <= 0 or hold_idx >= n - 2:
        return None
    s_new = np.empty_like(s_orig)
    l_new = np.empty_like(l_orig)
    s_new[:hold_idx] = s_orig[:hold_idx]
    l_new[:hold_idx] = l_orig[:hold_idx]
    end_hold = min(n, hold_idx + delay_steps)
    s_new[hold_idx:end_hold] = s_orig[hold_idx]
    l_new[hold_idx:end_hold] = l_orig[hold_idx]
    for t in range(end_hold, n):
        src = max(hold_idx, t - delay_steps)
        s_new[t] = s_orig[src]
        l_new[t] = l_orig[src]
    s_new = np.maximum.accumulate(s_new)
    return _profile_with_velocities(s_new, l_new, dt, tag=f"yield_delay_{float(delay_s):.1f}")


def build_yielding_support_profiles(all_speed_profile, path_conflict_events, dt, S_all=None, delays=None, hold_margin=3.0, max_base=8):
    """Build compact support set containing original low-action profiles and delayed yielding variants."""
    if delays is None:
        delays = [0.5, 1.0, 1.5, 2.0, 3.0]
    support = []
    if len(all_speed_profile) == 0:
        return support
    # Keep all original sampled profiles as support to preserve temporal continuity;
    # only synthetic yielding variants are limited to a compact low-action subset.
    for idx, profile in enumerate(all_speed_profile):
        base = dict(profile)
        base["_support_tag"] = f"candidate_original_{idx}"
        support.append(base)
    if S_all is not None and len(S_all) == len(all_speed_profile):
        order = list(np.argsort(np.asarray(S_all, dtype=float)[:, -1])[:max_base])
    else:
        order = list(range(min(max_base, len(all_speed_profile))))
    if not path_conflict_events:
        return support
    ranked_events = sorted(
        path_conflict_events,
        key=lambda ev: (
            -int(ev.get("is_small_actor", 0)),
            -float(ev.get("occupancy_strength", 0.0)) * float(ev.get("type_weight", 1.0)) * float(ev.get("crossing_weight", 1.0)),
            float(ev.get("t_enter", 0.0)),
        ),
    )[:3]
    seen_yield = set()
    for event in ranked_events:
        eid = int(event.get("event_id", 0))
        for idx in order:
            for delay in delays:
                prof = _make_delayed_profile(all_speed_profile[int(idx)], event, delay, dt, hold_margin=hold_margin)
                if prof is not None:
                    key = (eid, int(idx), round(float(delay), 2), round(float(prof["s"][min(len(prof["s"])-1, int(round(float(event.get("t_enter", 0.0)) / max(float(dt), 1e-6))))]), 3))
                    if key in seen_yield:
                        continue
                    seen_yield.add(key)
                    prof["_support_tag"] = f"yield_event_{eid}_base_{int(idx)}_delay_{float(delay):.1f}"
                    support.append(prof)
    return support


def _parse_delay_values(delays):
    if delays is None:
        return [0.5, 1.0, 1.5, 2.0, 3.0]
    if isinstance(delays, str):
        return [float(x) for x in delays.split(",") if str(x).strip()]
    return [float(x) for x in delays]


def lowest_action_path_time_velocity(
        gp_models: List,
        corridors,
        p_initial,
        dt: float,
        support_profiles=None,
        K: int = 120,
        lambda_l: float = 0.8,
        lambda_acc: float = 0.5,
        n_samples: int = 2000,
        seed: int = 2025,
        v_s_max: float = 13.9,
        v_l_max: float = 2.67,
        beta_uncertainty: float = 0.0,
        path_conflict_events=None,
        social_config=None,
        n_soc: float = 1.0,
        eta_pc_residual: float = 1.0,
):
    """Time-expanded beam DP with velocity/acceleration-aware transitions."""
    rng = default_rng(seed)
    N = len(gp_models)
    p_initial = np.asarray(p_initial, dtype=float)
    path_conflict_events = path_conflict_events or []
    cfg = normalize_social_config(social_config)

    def build_nodes(t):
        if t == 0:
            pts = np.array([[float(p_initial[0]), float(p_initial[1])]], dtype=float)
            tags = ["initial"]
        else:
            pts = sample_inside_corridor(corridors[0], t, n_samples, rng)
            mu_t, sigma_t = _predict_gp(gp_models[t], pts, return_std=beta_uncertainty > 0)
            pc_t = path_conflict_potential_at_points(pts, t, path_conflict_events, dt, n_soc, cfg)
            rank_value = mu_t + float(eta_pc_residual) * pc_t * float(dt) + float(beta_uncertainty) * sigma_t
            n_low = max(1, int(K * 0.6))
            low_idx = np.argsort(rank_value)[:n_low]
            remaining = np.setdiff1d(np.arange(len(pts)), low_idx, assume_unique=False)
            n_cover = max(0, K - len(low_idx))
            if n_cover > 0 and len(remaining) > 0:
                cover_idx = rng.choice(remaining, size=min(n_cover, len(remaining)), replace=False)
                idx = np.concatenate([low_idx, cover_idx])
            else:
                idx = low_idx
            pts = pts[idx]
            tags = ["sample"] * len(pts)
            if support_profiles is not None:
                support_rows = []
                support_tags = []
                for profile in support_profiles:
                    if t < len(profile.get("s", [])) and t < len(profile.get("l", [])):
                        support_rows.append([float(profile["s"][t]), float(profile["l"][t])])
                        support_tags.append(str(profile.get("_support_tag", "support")))
                if support_rows:
                    pts = np.vstack([pts, np.asarray(support_rows, dtype=float)])
                    tags.extend(support_tags)
        mu, sigma = _predict_gp(gp_models[t], pts, return_std=beta_uncertainty > 0)
        pc = path_conflict_potential_at_points(pts, t, path_conflict_events, dt, n_soc, cfg)
        nodes = []
        seen = set()
        for i in range(len(pts)):
            key = (round(float(pts[i, 0]), 3), round(float(pts[i, 1]), 3), tags[i])
            if key in seen:
                continue
            seen.add(key)
            pc_inc = float(eta_pc_residual) * float(pc[i]) * float(dt)
            field_cost = float(mu[i]) + pc_inc + float(beta_uncertainty) * float(sigma[i])
            nodes.append({
                "s": float(pts[i, 0]),
                "l": float(pts[i, 1]),
                "dS_gp": float(mu[i]),
                "pc_inc": pc_inc,
                "field_cost": field_cost,
                "sigma": float(sigma[i]),
                "tag": tags[i],
            })
        return nodes

    layers = []
    nodes0 = build_nodes(0)
    layers.append([{**nodes0[0], "total_cost": float(nodes0[0]["field_cost"]), "transition_cost": 0.0, "v_s": 0.0, "v_l": 0.0, "a_s": 0.0, "prev": -1}])

    for t in range(1, N):
        nodes = build_nodes(t)
        next_states = []
        for node in nodes:
            for i, prev in enumerate(layers[-1]):
                ds = float(node["s"] - prev["s"])
                dl = float(node["l"] - prev["l"])
                v_s = ds / float(dt)
                v_l = dl / float(dt)
                if ds < -1e-6 or v_s > v_s_max or abs(v_l) > v_l_max:
                    continue
                a_s = (v_s - float(prev.get("v_s", 0.0))) / float(dt)
                transition = float(lambda_l) * (dl ** 2) + float(lambda_acc) * (a_s ** 2) * float(dt)
                total = float(prev["total_cost"]) + float(node["field_cost"]) + transition
                next_states.append({**node, "total_cost": total, "transition_cost": transition, "v_s": float(v_s), "v_l": float(v_l), "a_s": float(a_s), "prev": int(i)})
        if not next_states:
            raise RuntimeError(f"No feasible time-velocity DP transition at step {t}; try increasing K/n_samples/beam or relaxing velocity bounds.")
        next_states.sort(key=lambda st: st["total_cost"])
        sample_states = [st for st in next_states if st.get("tag") == "sample"]
        support_states = [st for st in next_states if st.get("tag") != "sample"]
        # Preserve support-profile continuity; otherwise the beam can prune away
        # the only states that represent delayed/yielding trajectories.
        max_support_keep = max(int(K), 160)
        kept = sample_states[: int(K)] + support_states[: max_support_keep]
        kept.sort(key=lambda st: st["total_cost"])
        layers.append(kept)

    j = int(np.argmin([st["total_cost"] for st in layers[-1]]))
    best_score = float(layers[-1][j]["total_cost"])
    path, tags, v_s_profile, v_l_profile, a_s_profile, transition_costs, pc_increments = [], [], [], [], [], [], []
    for t in range(N - 1, -1, -1):
        st = layers[t][j]
        path.append((float(st["s"]), float(st["l"]), float(st["field_cost"])))
        tags.append(str(st["tag"]))
        v_s_profile.append(float(st["v_s"]))
        v_l_profile.append(float(st["v_l"]))
        a_s_profile.append(float(st["a_s"]))
        transition_costs.append(float(st["transition_cost"]))
        pc_increments.append(float(st["pc_inc"]))
        j = int(st["prev"]) if t > 0 else -1
    path = list(reversed(path)); tags = list(reversed(tags)); v_s_profile = list(reversed(v_s_profile)); v_l_profile = list(reversed(v_l_profile)); a_s_profile = list(reversed(a_s_profile)); transition_costs = list(reversed(transition_costs)); pc_increments = list(reversed(pc_increments))

    diagnostics = _path_physical_diagnostics([(p[0], p[1]) for p in path], dt, v_s_max=v_s_max, v_l_max=v_l_max)
    yielding_count = int(sum(tag.startswith("yield_") for tag in tags))
    diagnostics.update({
        "dp_score": best_score,
        "S_best_final": float(np.sum([p[2] for p in path])),
        "transition_cost_sum": float(np.sum(transition_costs)),
        "path_conflict_residual_sum": float(np.sum(pc_increments)),
        "seed": int(seed),
        "K": int(K),
        "beam_k": int(K),
        "n_samples": int(n_samples),
        "transition_mode": "time_velocity",
        "lambda_l": float(lambda_l),
        "lambda_acc": float(lambda_acc),
        "beta_uncertainty": float(beta_uncertainty),
        "eta_pc_residual": float(eta_pc_residual),
        "yielding_node_usage_count": yielding_count,
        "yielding_node_usage_ratio": float(yielding_count / max(len(tags), 1)),
        "dp_support_tags": tags,
        "dp_velocity_profile": v_s_profile,
        "dp_lateral_velocity_profile": v_l_profile,
        "dp_acceleration_profile": a_s_profile,
        "dp_path_conflict_residual_profile": pc_increments,
    })
    return path, diagnostics


def lowest_action_path_incremental(
        gp_models: List,
        corridors,
        p_initial,
        dt: float,
        support_profiles=None,
        K: int = 80,
        lambda_l: float = 0.8,
        n_samples: int = 2000,
        seed: int = 2025,
        v_s_max: float = 13.9,
        v_l_max: float = 2.67,
        beta_uncertainty: float = 0.0,
):
    rng = default_rng(seed)
    N = len(gp_models)
    p_initial = np.asarray(p_initial, dtype=float)

    cands: List[List[Tuple[float, float, float, float]]] = []
    mu0, sigma0 = _predict_gp(gp_models[0], np.array([[p_initial[0], p_initial[1]]]), return_std=beta_uncertainty > 0)
    cands.append([(float(p_initial[0]), float(p_initial[1]), float(mu0[0]), float(sigma0[0]))])

    for t in range(1, N):
        pts = sample_inside_corridor(corridors[0], t, n_samples, rng)
        mu_t, sigma_t = _predict_gp(gp_models[t], pts, return_std=beta_uncertainty > 0)
        rank_value = mu_t + float(beta_uncertainty) * sigma_t
        n_low = max(1, int(K * 0.6))
        low_idx = np.argsort(rank_value)[:n_low]
        remaining = np.setdiff1d(np.arange(len(pts)), low_idx, assume_unique=False)
        n_cover = max(0, K - len(low_idx))
        if n_cover > 0 and len(remaining) > 0:
            cover_idx = rng.choice(remaining, size=min(n_cover, len(remaining)), replace=False)
            idx = np.concatenate([low_idx, cover_idx])
        else:
            idx = low_idx
        cand_t = [(float(pts[i, 0]), float(pts[i, 1]), float(mu_t[i]), float(sigma_t[i])) for i in idx]
        if support_profiles is not None:
            support_pts = np.array(
                [[float(profile["s"][t]), float(profile["l"][t])] for profile in support_profiles],
                dtype=float,
            )
            support_mu, support_sigma = _predict_gp(
                gp_models[t],
                support_pts,
                return_std=beta_uncertainty > 0,
            )
            cand_t.extend(
                [
                    (float(support_pts[i, 0]), float(support_pts[i, 1]), float(support_mu[i]), float(support_sigma[i]))
                    for i in range(len(support_pts))
                ]
            )
        cands.append(cand_t)

    prev_cost = np.array([c[2] + float(beta_uncertainty) * c[3] for c in cands[0]], dtype=float)
    prev_ptr = []
    for t in range(1, N):
        cost_t = np.full(len(cands[t]), np.inf)
        ptr_t = np.full(len(cands[t]), -1, dtype=int)
        for j, (s, l, dS, sigma) in enumerate(cands[t]):
            node_cost = dS + float(beta_uncertainty) * sigma
            for i, (sp, lp, _, _) in enumerate(cands[t - 1]):
                trans = _physical_transition(
                    s,
                    l,
                    sp,
                    lp,
                    dt,
                    lambda_l=lambda_l,
                    v_s_max=v_s_max,
                    v_l_max=v_l_max,
                )
                val = prev_cost[i] + node_cost + trans
                if val < cost_t[j]:
                    cost_t[j] = val
                    ptr_t[j] = i
        if not np.any(np.isfinite(cost_t)):
            raise RuntimeError(
                f"No feasible incremental DP transition at step {t}; "
                f"try increasing K/n_samples or relaxing velocity bounds."
            )
        prev_cost = cost_t
        prev_ptr.append(ptr_t)

    t = N - 1
    j = int(prev_cost.argmin())
    best_score = float(prev_cost[j])
    path = []
    while t >= 0:
        s, l, dS, sigma = cands[t][j]
        path.append((s, l, dS))
        if t > 0:
            j = int(prev_ptr[t - 1][j])
        t -= 1
    path = list(reversed(path))
    diagnostics = _path_physical_diagnostics([(p[0], p[1]) for p in path], dt, v_s_max=v_s_max, v_l_max=v_l_max)
    diagnostics.update(
        {
            "dp_score": best_score,
            "S_best_final": float(np.sum([p[2] for p in path])),
            "seed": int(seed),
            "K": int(K),
            "n_samples": int(n_samples),
            "transition_mode": "incremental_physical",
            "lambda_l": float(lambda_l),
            "beta_uncertainty": float(beta_uncertainty),
        }
    )
    return path, diagnostics

def eval_ideal_on_actual_field(opt_path_ideal,
                               gp_models):

    s_ideal = opt_path_ideal['s']
    l_ideal = opt_path_ideal['l']
    N       = len(s_ideal)

    # 对每一帧，把 (s,l) 喂给 gp_models[t]
    mu_ideal = np.array([
        gp_models[t].predict(np.array([[s_ideal[t], l_ideal[t]]]))[0]
        for t in range(N)
    ])
    return mu_ideal


def eval_ideal_on_actual_field_incremental(opt_path_ideal, gp_models):
    return eval_ideal_on_actual_field(opt_path_ideal, gp_models)


def get_goal_xy(goal_pos):
    """
    Robustly extract a representative (x, y) for goal_pos which may be:
    - Rectangle / Circle / Polygon-like shapes with .center
    - Shapes with .vertices
    - ShapeGroup with .shapes (list of shapes)
    """
    # 1) most common: Rectangle/Circle/Polygon wrapper has center
    if hasattr(goal_pos, "center"):
        c = goal_pos.center
        return float(c[0]), float(c[1])

    # 2) Polygon/Rectangle may have vertices -> use centroid of vertices
    verts = getattr(goal_pos, "vertices", None)
    if verts is not None:
        verts = np.asarray(verts, dtype=float)
        return float(verts[:, 0].mean()), float(verts[:, 1].mean())

    # 3) ShapeGroup: take average of sub-shape centers (or vertices-centroid)
    shapes = getattr(goal_pos, "shapes", None)
    if shapes is not None and len(shapes) > 0:
        centers = []
        for shp in shapes:
            if hasattr(shp, "center"):
                c = shp.center
                centers.append([float(c[0]), float(c[1])])
            else:
                v = getattr(shp, "vertices", None)
                if v is not None:
                    v = np.asarray(v, dtype=float)
                    centers.append([float(v[:, 0].mean()), float(v[:, 1].mean())])

        if len(centers) > 0:
            centers = np.asarray(centers, dtype=float)
            return float(centers[:, 0].mean()), float(centers[:, 1].mean())

    raise RuntimeError(f"Cannot extract goal (x,y) from goal_pos type={type(goal_pos)}")

def has_nan_or_inf(traj):
    """
    判断一个 trajectory dict 中是否存在 NaN / Inf
    """
    for k, v in traj.items():
        if isinstance(v, np.ndarray):
            if not np.all(np.isfinite(v)):
                return True
    return False

def filter_nan_speed_profiles(all_speed_profile, verbose=True):
    """
    删除 all_speed_profile 中包含 NaN / Inf 的轨迹
    """
    cleaned_profiles = []
    dropped_indices = []

    for i, traj in enumerate(all_speed_profile):
        if has_nan_or_inf(traj):
            dropped_indices.append(i)
            if verbose:
                print(f"[DROP] trajectory #{i} contains NaN / Inf")
        else:
            cleaned_profiles.append(traj)

    if verbose:
        print(
            f"[INFO] kept {len(cleaned_profiles)} / {len(all_speed_profile)} trajectories"
        )

    if len(cleaned_profiles) == 0:
        raise RuntimeError("❌ all_speed_profile 中所有轨迹都包含 NaN")

    return cleaned_profiles


from LeastAction_utils import nodes_sample, generate_trajs, allocate_terminal_speeds,generate_speed_profile, obstacles_within_reachable

def main(
    config,
    reach_interface,
    corridors,
    m_i,
    g,
    n_g,
    l_w,
    v_lim,
    v_des,
    v_ref,
    B_l,
    n_soc,
    w_goal_pos,
    N_step,
    r_coeff=20.0,
    action_gp_target="cumulative",
    dp_mode="legacy_current",
    dp_k=40,
    dp_samples=2000,
    dp_seed=2025,
    dp_lambda=0.8,
    dp_lambda_acc=0.5,
    eta_pc_residual=0.0,
    hold_margin=3.0,
    yield_delays=None,
    return_diagnostics=False,
    social_config=None,
):
    final, s_final, l_final = nodes_sample(corridors=corridors, n_samples=50, terminal_step=N_step)
    s_finals, l_finals = list(s_final), list(l_final)
    p0 = config.planning.p_initial
    v0 = [config.planning.v_lon_initial,0]

    # parameters
    dt = config.planning.dt # 采样步长
    ##### SIND ######
    xg, yg = config.planning_problem.goal.state_list[0].position.center #
    #### SIND ######

    #### CommonRoad ######
    # goal_pos = config.planning_problem.goal.state_list[0].position
    # xg, yg = get_goal_xy(goal_pos)
    #### CommonRoad ######

    s_goal, l_goal = config.planning.CLCS.convert_to_curvilinear_coords(xg, yg)
    # —— 几何轨迹采样 —— #
    all_trajs = generate_trajs(corridors, s_finals, l_finals, p0[0], p0[1], N_step, dt)
    # ===== 终端速度分配 =====
    
    ##### SIND ######
    center_cart = config.planning_problem.goal.state_list[0].position.center
    ##### SIND ######

    #### CommonRoad ######
    # center_cart = np.array(get_goal_xy(goal_pos), dtype=float)
    #### CommonRoad ######
    center_cuvl = config.planning.CLCS.convert_to_curvilinear_coords(center_cart[0], center_cart[1])
    v_finals = allocate_terminal_speeds(
        s_finals,
        p0[0],
        s_ref=center_cuvl[0],
        v_ref=v_ref,
        v0=config.planning.v_lon_initial
    )

    # 速度多项式规划
    all_speed_profile = []
    for i, item in enumerate(all_trajs):
        s_arr = item['s']
        l_arr = item['l']
        profile = generate_speed_profile(
            s_arr, l_arr, v0 = v0[0], a0 = 0,
            vf = v_finals[i],
            dt = dt
        )
        all_speed_profile.append(profile)

    all_speed_profile = filter_nan_speed_profiles(
        all_speed_profile,
        verbose=True
    )

    # 只保留与 ego 车有交互关系的 dynamic obstacle（按 oid 过滤）
    # corridors 无法考虑向后的传播，设置了固定值slack_back，考虑ego car后10m的车辆
    neighbors_by_t_raw = generate_neighbors_by_t(config, T_h=N_step * dt)
    neighbors_by_t, KEEP_OIDS, REACHABLE_OIDS = filter_neighbors_for_social(
        neighbors_by_t_raw,
        reach_interface,
        p0,
        s_goal=s_goal,
        l_goal=l_goal,
        social_config=social_config,
    )
    path_conflict_events = build_path_conflict_events(config, N_step, social_config)

    S_all = []
    dS_all = []
    for traj in all_speed_profile:
        S_total_i, dS_i = compute_dS_for_traj(
            traj,
            neighbors_by_t,
            s_goal, l_goal,
            m_i, l_w, B_l, p0[1],
            n_soc, w_goal_pos,
            dt, g, n_g, v_des, v_lim,
            r_coeff=r_coeff,
            social_config=social_config,
            path_conflict_events=path_conflict_events,
        )
        S_all.append(S_total_i)
        dS_all.append(dS_i)
    S_all = np.array(S_all)
    dS_all = np.array(dS_all)

    # 根据随机采样结果在每个time step进行高斯过程回归
    diagnostics = {
        "action_gp_target": action_gp_target,
        "dp_mode": dp_mode,
        "dS_all_shape": tuple(dS_all.shape),
        "S_all_shape": tuple(S_all.shape),
        "cumsum_dS_matches_S_total": bool(np.allclose(np.cumsum(dS_all, axis=1), S_all)),
        "candidate_min_S": float(np.min(S_all[:, -1])),
        "candidate_median_S": float(np.median(S_all[:, -1])),
        "candidate_mad_S": float(np.median(np.abs(S_all[:, -1] - np.median(S_all[:, -1]))) + 1e-6),
        "social_config": normalize_social_config(social_config),
        "neighbor_keep_count": int(len(KEEP_OIDS)),
        "neighbor_reachable_count": int(len(REACHABLE_OIDS)),
        "neighbor_pair_count": int(sum(len(neighs) for neighs in neighbors_by_t)),
        "path_conflict_event_count": int(len(path_conflict_events)),
        "path_conflict_small_actor_event_count": int(sum(int(ev.get("is_small_actor", 0)) for ev in path_conflict_events)),
    }
    support_profiles_for_dp = all_speed_profile
    if dp_mode == "time_velocity":
        support_profiles_for_dp = build_yielding_support_profiles(
            all_speed_profile,
            path_conflict_events,
            dt,
            S_all=S_all,
            delays=_parse_delay_values(yield_delays),
            hold_margin=hold_margin,
            max_base=8,
        )
        diagnostics["support_profile_count"] = int(len(support_profiles_for_dp))
        diagnostics["yielding_support_profile_count"] = int(sum(str(p.get("_support_tag", "")).startswith("yield_") for p in support_profiles_for_dp))
    if action_gp_target == "incremental":
        gp_models = gaussian_process_regression_incremental(dS_all, all_speed_profile)
    elif action_gp_target == "cumulative":
        gp_models = gaussian_process_regression(S_all, all_speed_profile)
    else:
        raise ValueError(f"Unknown action_gp_target={action_gp_target}")

    if dp_mode == "incremental_physical":
        opt_path, dp_diag = lowest_action_path_incremental(
            gp_models=gp_models,
            corridors=corridors,
            p_initial=p0,
            dt=dt,
            support_profiles=support_profiles_for_dp,
            K=dp_k,
            lambda_l=dp_lambda,
            n_samples=dp_samples,
            seed=dp_seed,
            v_s_max=v_lim,
        )
        diagnostics.update(dp_diag)
        diagnostics["dp_vs_candidate_min_mad"] = float(
            (diagnostics["candidate_min_S"] - diagnostics["S_best_final"]) / diagnostics["candidate_mad_S"]
        )
    elif dp_mode == "time_velocity":
        opt_path, dp_diag = lowest_action_path_time_velocity(
            gp_models=gp_models,
            corridors=corridors,
            p_initial=p0,
            dt=dt,
            support_profiles=support_profiles_for_dp,
            K=dp_k,
            lambda_l=dp_lambda,
            lambda_acc=dp_lambda_acc,
            n_samples=dp_samples,
            seed=dp_seed,
            v_s_max=v_lim,
            path_conflict_events=path_conflict_events,
            social_config=social_config,
            n_soc=n_soc,
            eta_pc_residual=eta_pc_residual,
        )
        diagnostics.update(dp_diag)
        diagnostics["dp_vs_candidate_min_mad"] = float(
            (diagnostics["candidate_min_S"] - diagnostics["S_best_final"]) / diagnostics["candidate_mad_S"]
        )
    elif dp_mode in {"legacy_current", "current"}:
        opt_path = lowest_action_path(
            gp_models=gp_models,
            corridors=corridors,
            K=dp_k,
            lam=dp_lambda,
            n_samples=dp_samples,
            seed=dp_seed,
        )
        diagnostics.update(
            {
                "S_best_final": float(opt_path[-1][2]),
                "seed": int(dp_seed),
                "K": int(dp_k),
                "n_samples": int(dp_samples),
                "transition_mode": "legacy_distance",
                "lambda_l": float(dp_lambda),
            }
        )
    else:
        raise ValueError(f"Unknown dp_mode={dp_mode}")

    if return_diagnostics:
        return gp_models, all_speed_profile, S_all, opt_path, final, diagnostics
    return gp_models, all_speed_profile, S_all, opt_path, final
