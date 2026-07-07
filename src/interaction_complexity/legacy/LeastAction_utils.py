import numpy as np
from scipy.stats import qmc

def _sorted_reach_nodes(nodes):
    return sorted(
        list(nodes),
        key=lambda n: (
            float(n.p_lon_min),
            float(n.p_lon_max),
            float(n.p_lat_min),
            float(n.p_lat_max),
        ),
    )

def make_preview_dict(p0, v0, a0, T_preview=3.0, dt=0.1):
    """
    生成常加速度预瞄轨迹，并打包成
      {'t', 's', 'l', 'v_x', 'v_y'}
    """
    # 时间网格
    N = int(np.round(T_preview/dt)) + 1
    t = np.linspace(0.0, T_preview, N)
    
    # 预瞄位置 s, l
    s = p0[0] + v0[0]*t + 0.5*a0[0]*t**2
    l = p0[1] + v0[1]*t + 0.5*a0[1]*t**2
    
    # 纵横向速度 v_x, v_y
    # v(t) = v0 + a0 * t
    v_x = v0[0] + a0[0]*t
    v_y = v0[1] + a0[1]*t
    
    return {
        't':   t,
        's':   s,
        'l':   l,
        'v_x': v_x,
        'v_y': v_y,
    }

def _infer_last_corridor_step(corridors, max_probe=1000):
    last_step = None
    for k in range(max_probe + 1):
        nodes = _sorted_reach_nodes(corridors[0].reach_nodes_at_step(k))
        if not nodes:
            break
        last_step = k
    if last_step is None:
        raise ValueError("No reach nodes found in corridors.")
    return last_step


def nodes_sample(corridors, n_samples=50, terminal_step=None):

    if terminal_step is None:
        terminal_step = _infer_last_corridor_step(corridors)

    terminal_step = int(terminal_step)
    if terminal_step < 0:
        raise ValueError(f"terminal_step must be non-negative, got {terminal_step}")

    # 1) 使用当前规划 horizon 对应的最后一帧 corridor
    nodes = _sorted_reach_nodes(corridors[0].reach_nodes_at_step(terminal_step))
    if not nodes:
        raise ValueError(f"No reach nodes found at terminal_step={terminal_step}")
    # 2) 计算总包围框
    s_min = min(n.p_lon_min for n in nodes)
    s_max = max(n.p_lon_max for n in nodes)
    l_min = min(n.p_lat_min for n in nodes)
    l_max = max(n.p_lat_max for n in nodes)

    # 3) LHS 在 [0,1]^2
    sampler = qmc.LatinHypercube(d=2, seed=0)
    unit_pts = sampler.random(n=n_samples*2)  # 先多采一些，防止拒绝率

    # 4) 缩放到大长方体
    l_bounds = [s_min, l_min]
    u_bounds = [s_max, l_max]
    scaled = qmc.scale(unit_pts, l_bounds, u_bounds)  # → shape (M,2)
    s_pts = scaled[:, 0]
    l_pts = scaled[:, 1]

    # 5) 拒绝采样：只保留落入任一小矩形内的点
    final = []
    for s, l in zip(s_pts, l_pts):
        for n in nodes:
            s0, l0, s1, l1 = n.p_lon_min, n.p_lat_min, n.p_lon_max, n.p_lat_max
            if s0 <= s <= s1 and l0 <= l <= l1:
                final.append((s, l))
                break
        if len(final) >= n_samples:
            break

    final = np.array(final[:n_samples])
    s_final = final[:,0]
    l_final = final[:,1]
    return final, s_final, l_final


from scipy.optimize import minimize, Bounds
from typing import List, Tuple, Dict


def corridor_bounds(corridors, step:int, s_val:float) -> Tuple[float,float]:

    for n in _sorted_reach_nodes(corridors[0].reach_nodes_at_step(step)):
        if n.p_lon_min <= s_val <= n.p_lon_max:
            return n.p_lat_min, n.p_lat_max
    return None, None


def l_poly(s0, a:np.ndarray, s:np.ndarray) -> np.ndarray:
    """a = [a0..a5]"""
    u = s - s0
    return (((((a[5]*u + a[4])*u + a[3])*u + a[2])*u + a[1])*u + a[0])

def generate_trajs(corridors, s_finals, l_finals, 
                   s0, l0, 
                   N_step, dt):
    """
    对给定的终点列表循环生成优化轨迹。
    Args:
        s_finals, l_finals: 等长的 list/array，所有候选终点。
        s0, l0: 起点 Frenet 坐标。
        N_step: 步数。
        dt: 时间步长。
        initial_guess: 无参函数，返回初始多项式系数 a0。
        l_poly: 函数 l = l_poly(a, s_array)。
        corridor_bounds: 函数 (l_lo, l_hi) = corridor_bounds(step, s_val)。
    Returns:
        all_trajs: list of dicts，每个 dict 包含 keys 't','s','l'。
    """
    all_trajs = []
    # 纵向采样网格 (统一，不依赖于每条轨迹的 s_f)
    t_grid = np.arange(0.0, N_step*dt + 1e-6, dt)
    
    for s_f, l_f in zip(s_finals, l_finals):

        # 1) 针对本次目标定义初猜
        def initial_guess_for(s_f, l_f):
            a = np.zeros(6)
            a[0] = l0
            a[1] = (l_f - l0) / (s_f - s0 + 1e-9)
            return a

        def objective(a_var: np.ndarray) -> float:
            a = a_var.copy()
            a[0] = l0
            # s_grid 按本次 s_final 线性均匀
            s_grid = np.linspace(s0, s_f, N_step+1)
            l_grid = l_poly(s0, a, s_grid)
            penalty = 0.0
            w_out, w_slope, w_end = 5e-1, 2e-1, 5.0

            # corridor 越界罚
            for k, (s_k, l_k) in enumerate(zip(s_grid, l_grid)):
                lo_hi = corridor_bounds(corridors, k, s_k)
                if lo_hi[0] is None:
                    continue
                l_lo, l_hi = lo_hi
                if l_k < l_lo:
                    penalty += w_out * (l_lo - l_k)**2
                elif l_k > l_hi:
                    penalty += w_out * (l_k - l_hi)**2

            # 横向平滑
            dl_ds = np.gradient(l_grid, s_grid)
            penalty += w_slope * np.sum(dl_ds**2)

            # 末端收敛
            penalty += w_end * (l_grid[-1] - l_f)**2

            return penalty

        # 初始化与边界
        a0 = initial_guess_for(s_f, l_f)
        bounds = Bounds(lb=[l0, -5, -5, -5, -5, -5],
                        ub=[l0,  5,  5,  5,  5,  5])

        # 优化
        res = minimize(objective, x0=a0, method='L-BFGS-B',
                       bounds=bounds,
                       options=dict(maxiter=200, ftol=1e-6, disp=False))
        a_opt = res.x
        a_opt[0] = l0

        # 生成最终离散轨迹
        s_grid = np.linspace(s0, s_f, len(t_grid))
        l_grid = l_poly(s0, a_opt, s_grid)

        traj = {'t': t_grid, 's': s_grid, 'l': l_grid}
        all_trajs.append(traj)

    return all_trajs

def allocate_terminal_speeds(
    s_finals,
    s0,
    s_ref,
    v_ref,
    v0,
    a_min_brake = -4.0,   # m/s² (负值)
    v_min_allowed = 0.0,
    v_max_allowed = 10.0,
    k_sqrt = True,
    stop_margin = 1.05    # 预留 5% 余量
):
    """
    根据剩余位移为每个终点分配可行 v_f:
      1) 极小位移 (<= s_stop) → v_f = 0
      2) 中位移 → sqrt(s/s_ref) 或线性缩放
      3) 裁剪到 [v_min_allowed, v_max_allowed]
    """
    s_arr = np.asarray(s_finals, dtype=float) - float(s0)   # 剩余位移 Δs
    s_arr = np.maximum(s_arr, 0.0)

    # —— 刹车所需最小距离 —— #
    s_stop  = (v0**2) / (-2.0 * a_min_brake)    # a_min_brake<0
    s_stop *= stop_margin                       # 加一点安全余量

    # —— (1) 初步目标速度 —— #
    if k_sqrt:
        v_target = v_ref * np.sqrt(s_arr / max(s_ref - s0, 1e-3))
    else:
        v_target = v_ref * (s_arr / max(s_ref - s0, 1e-3))

    # —— (2) 极小位移直接刹停 —— #
    v_target = np.where(s_arr <= s_stop, 0.0, v_target)

    # —— (3) 物理极限速度下界 —— #
    v_phys_min = np.sqrt(np.maximum(0.0, v0**2 + 2.0 * a_min_brake * s_arr))
    v_adj      = np.maximum(v_target, v_phys_min)

    # —— (4) 最终裁剪 —— #
    v_final = np.clip(v_adj, v_min_allowed, v_max_allowed)

    return v_final

# ------------------------------------------------------------
#  基本工具
# ------------------------------------------------------------
def compute_arc_lengths(s_arr, l_arr):
    ds = np.hypot(np.diff(s_arr), np.diff(l_arr))
    return ds, ds.sum()

def solve_cubic_coeffs(S_tot, v0, a0, vf, T):
    A = np.array([[     T**2,       T**3],
                  [ T**3/3.0,   T**4/4.0]])
    B = np.array([vf - v0 - a0*T,
                  S_tot - v0*T - 0.5*a0*T**2])
    return np.linalg.solve(A, B)  # b, c

# ------------------------------------------------------------
#  若出现负弧长速度 → 上移 + 缩放
# ------------------------------------------------------------
def enforce_non_negative(v_poly, t_grid, v0, vf, S_tot):
    v_vals = v_poly(t_grid)
    v_min  = v_vals.min()
    if v_min >= 0:
        return v_poly                        # 已满足

    dv = -v_min                              # 上移 Δv
    def v_shift(t, dv=dv):
        return v_poly(t) + dv

    scale = vf / v_shift(t_grid[-1])         # 拉回末速度

    def v_new(t, scale=scale, dv=dv):
        return scale * (v_poly(t) + dv)

    # 微调 scale 保面积
    area  = np.trapz(v_new(t_grid), t_grid)
    scale *= S_tot / max(area, 1e-6)

    return lambda t, scale=scale, dv=dv: scale * (v_poly(t) + dv)

# ------------------------------------------------------------
#  生成 Frenet 纵/横向速度剖面
# ------------------------------------------------------------
def generate_speed_profile(s_arr, l_arr,
                           v0, a0, vf, dt):
    N          = len(s_arr) - 1
    ds, S_tot  = compute_arc_lengths(s_arr, l_arr)
    T          = N * dt

    # ——— 1) 求弧长速度三次多项式 ——— #
    b, c       = solve_cubic_coeffs(S_tot, v0, a0, vf, T)
    v_poly_raw = lambda t, v0=v0, a0=a0, b=b, c=c: v0 + a0*t + b*t**2 + c*t**3

    t_grid     = np.linspace(0.0, T, N+1)
    v_poly     = enforce_non_negative(v_poly_raw, t_grid, v0, vf, S_tot)
    v_arc      = v_poly(t_grid)                      # (N+1,)

    # ——— 2) Frenet 速度分量 ——— #
    # k(s)=dl/ds, 与 s_arr 对齐
    k_arr      = np.gradient(l_arr, s_arr)
    denom      = np.sqrt(1.0 + k_arr**2)
    v_s        = v_arc / denom                      # 纵向速度 ≥0
    v_lat      = v_arc * k_arr / denom              # 横向速度

    # ——— 3) 用 v_s 积分重建 s_est(t)（校验） ——— #
    s_est      = np.empty_like(v_s)
    s_est[0]   = s_arr[0]
    for k in range(N):
        s_est[k+1] = s_est[k] + 0.5*(v_s[k]+v_s[k+1]) * dt
    l_est = np.interp(s_est, s_arr, l_arr)

    return {
        "t"      : t_grid,
        "v_arc"  : v_arc,
        "v_x"    : v_s,
        "v_y"    : v_lat,
        "s"      : s_est,
        "l"      : l_est
    }

def obstacles_within_reachable(neighbors_by_t, reach_interface, slack_back):
    in_bounds_oids = set()

    # 对每个时间步
    for t, obs_list in enumerate(neighbors_by_t):
        # 1) 从可达集 nodes 获取这一帧所有 node 的边界
        nodes = reach_interface.reachable_set_at_step(t)
        if not nodes:
            continue

        # 2) 计算这一帧整体的 s,l 最小最大边界
        s_mins = [n.p_lon_min for n in nodes]
        s_maxs = [n.p_lon_max for n in nodes]
        l_mins = [n.p_lat_min for n in nodes]
        l_maxs = [n.p_lat_max for n in nodes]
        s_min_global, s_max_global = min(s_mins), max(s_maxs)
        l_min_global, l_max_global = min(l_mins), max(l_maxs)

        # 3) 对该时刻的所有动态障碍物，检查是否落在边界内
        for ob in obs_list:
            s_o, l_o = ob['s'], ob['l']
            if (s_min_global-10 <= s_o <= s_max_global
                    and l_min_global <= l_o <= l_max_global):
                in_bounds_oids.add(ob['oid'])

    return in_bounds_oids
