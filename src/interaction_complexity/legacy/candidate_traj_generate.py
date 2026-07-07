# 使用PDM生成dynamic obstacle的候选轨迹
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple

# ────────────────────────────────────────────────
# 0. 参数封装
# ────────────────────────────────────────────────
@dataclass
class GenParam:
    dt: float       = 0.1      # [s]
    T:  float       = 3.0      # [s]  规划时域
    v_limit: float  = 15.0     # [m/s] 限速
    a_max:  float   = 4.0      # IDM
    b_comf: float   = 3.0
    T_gap:  float   = 1.5
    s0_gap: float   = 2.0

P = GenParam() 

# ────────────────────────────────────────────────
# 1. 纵向 IDM rollout
# ────────────────────────────────────────────────
def idm_longitudinal(
        s0: float, v0: float,
        v_des: float,
        param: GenParam,
        lead: Dict[str, np.ndarray]=None
) -> Tuple[np.ndarray, np.ndarray]:
    """返回 s_arr, v_arr (长度 N+1)"""
    N = int(param.T/param.dt); dt = param.dt
    s_arr = np.zeros(N+1); v_arr = np.zeros(N+1)
    s_arr[0], v_arr[0] = s0, v0

    if lead is None:
        s_lead = np.full(N+1, np.inf); v_lead = np.zeros(N+1)
    else:
        s_lead = lead['s']; v_lead = lead['v_x']

    a,b,Tgap,s0gap = param.a_max, param.b_comf, param.T_gap, param.s0_gap
    for k in range(N):
        gap   = max(s_lead[k] - s_arr[k] - s0gap, 0.1)
        dv    = v_arr[k] - v_lead[k]
        s_star= s0gap + v_arr[k]*Tgap + v_arr[k]*dv/(2*np.sqrt(a*b))
        acc   = a * (1 - (v_arr[k]/v_des)**4 - (s_star/gap)**2)
        v_nxt = max(v_arr[k] + acc*dt, 0.0)
        s_nxt = s_arr[k] + 0.5*(v_arr[k]+v_nxt)*dt
        v_arr[k+1] = v_nxt;  s_arr[k+1] = s_nxt
    return s_arr, v_arr

# ────────────────────────────────────────────────
# 2. 横向五次多项式 offset
# ────────────────────────────────────────────────
def quintic_offset(l0: float, l_f: float, param: GenParam):
    T = param.T; t = np.linspace(0, T, int(T/param.dt)+1)
    a0=l0; a1=a2=0
    a3 = 10*(l_f-l0)/T**3
    a4 = -15*(l_f-l0)/T**4
    a5 = 6*(l_f-l0)/T**5
    l  = a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5
    dl = a1 + 2*a2*t + 3*a3*t**2 + 4*a4*t**3 + 5*a5*t**4
    return l, dl

# ────────────────────────────────────────────────
# 3. 主函数：生成 25 条候选
# ────────────────────────────────────────────────
def generate_pdm_candidates(
        ref_clcs,                       # Curvilinear CS (提供 s,l↔x,y)
        s0: float, l0: float, v0: float,
        param: GenParam = P,
        lead: Dict[str,np.ndarray]=None
) -> List[Dict[str,np.ndarray]]:
    N = int(param.T/param.dt)
    t_full = np.linspace(0, param.T, N+1)

    v_targets   = np.array([0.6, 0.8, 1.0, 1.2]) * param.v_limit  # np.array([0.15]) * param.v_limit  #0.4,0.6,0.8,
    lat_offsets = np.array([ -3.0, -2.0 , 0.0 , +2.0, 3.0])       # np.array([ 0.0])       # m-3.0, +3.0, -1.0 +1.0

    trajs = []
    for v_des in v_targets:
        s_arr, v_x = idm_longitudinal(s0, v0, v_des, param, lead)
        for off in lat_offsets:
            # 横向五次
            l_arr, v_y = quintic_offset(l0, off, param)

            # Frenet → Cartesian，遇到超出投影域就截断
            xs, ys = [], []
            cut_idx = N+1
            for k, (s_k, l_k) in enumerate(zip(s_arr, l_arr)):
                try:
                    x,y = ref_clcs.convert_to_cartesian_coords(float(s_k), float(l_k))
                except ValueError:
                    cut_idx = k
                    break
                else:
                    xs.append(x); ys.append(y)

            if cut_idx < 2:
                # 如果一两个点都算不出，就跳过这个候选
                continue

            # 截断其余数组
            t_grid = t_full[:cut_idx]
            s_cut  = s_arr[:cut_idx]
            l_cut  = l_arr[:cut_idx]
            v_x_cut= v_x[:cut_idx]
            v_y_cut= v_y[:cut_idx]
            x_cut  = np.array(xs)
            y_cut  = np.array(ys)

            trajs.append({
                't':   t_grid,
                's':   s_cut,
                'l':   l_cut,
                'v_x': v_x_cut,
                'v_y': v_y_cut,
                'x':   x_cut,
                'y':   y_cut,
                'tag': f'v={v_des:.1f}, off={off:+.1f}'
            })

    return trajs