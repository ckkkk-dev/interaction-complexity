import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def _infer_last_corridor_step(corridors, max_probe=1000):
    last_step = None
    for k in range(max_probe + 1):
        nodes = corridors[0].reach_nodes_at_step(k)
        if not nodes:
            break
        last_step = k
    if last_step is None:
        raise ValueError("No reach nodes found in corridors.")
    return last_step


def visualization_sample_nodes(corridors, final, terminal_step=None):
    # —— 准备数据 —— #
    if terminal_step is None:
        terminal_step = _infer_last_corridor_step(corridors)
    # 当前规划 horizon 对应的最后一帧 ReachNode
    nodes = corridors[0].reach_nodes_at_step(int(terminal_step))

    # LHS 抽样结果
    # final.shape = (n_samples, 2)
    # s_final = final[:, 0]; l_final = final[:, 1]
    s_final = final[:, 0]
    l_final = final[:, 1]

    # —— 可视化 —— #
    fig, ax = plt.subplots(figsize=(8, 6))

    # 1) 绘制每个 corridor 矩形
    for n in nodes:
        s0, l0 = n.p_lon_min, n.p_lat_min
        s1, l1 = n.p_lon_max, n.p_lat_max
        width  = s1 - s0
        height = l1 - l0
        rect = patches.Rectangle(
            (s0, l0), width, height,
            edgecolor='blue', facecolor='none',
            linewidth=1.5, linestyle='--'
        )
        ax.add_patch(rect)

    # 2) 绘制采样点
    ax.scatter(s_final, l_final,
            color='red', s=30,
            label='LHS Sampled Endpoints')

    # 3) 标注和美化
    ax.set_xlabel("s (m)")
    ax.set_ylabel("l (m)")
    ax.set_title("Final-Step Corridors & LHS Samples")
    ax.legend(loc='upper right')
    ax.grid(linestyle='--', alpha=0.5)

    # 4) 固定坐标范围（可根据需要微调）
    ax.set_xlim(min(n.p_lon_min for n in nodes) - 1,
                max(n.p_lon_max for n in nodes) + 1)
    ax.set_ylim(min(n.p_lat_min for n in nodes) - 1,
                max(n.p_lat_max for n in nodes) + 1)

    plt.tight_layout()
    plt.show()


def visualization_traj(corridors, final, all_speed_profile, terminal_step=None):
    # —— 准备数据 —— #
    if terminal_step is None:
        terminal_step = _infer_last_corridor_step(corridors)
    nodes = corridors[0].reach_nodes_at_step(int(terminal_step))

    # LHS 抽样结果
    s_finals = final[:, 0]
    l_finals = final[:, 1]


    # —— 可视化 —— #
    fig, ax = plt.subplots(figsize=(10, 6))

    # 1) 绘制每个 corridor 矩形
    for n in nodes:
        s0_corr, l0_corr = n.p_lon_min, n.p_lat_min
        s1_corr, l1_corr = n.p_lon_max, n.p_lat_max
        rect = patches.Rectangle(
            (s0_corr, l0_corr),
            s1_corr - s0_corr,
            l1_corr - l0_corr,
            edgecolor='blue', facecolor='none',
            linewidth=1.2, linestyle='--'
        )
        ax.add_patch(rect)

    # 2) 绘制所有轨迹
    for traj in all_speed_profile:
        s_traj = traj['s']
        l_traj = traj['l']
        ax.plot(s_traj, l_traj,
                color='green', alpha=0.3, linewidth=1)
        
    # 3) 绘制采样终点，并标注索引
    for idx, (s_end, l_end) in enumerate(zip(s_finals, l_finals)):
        ax.scatter(s_end, l_end,
                color='red', s=40, zorder=5)
        ax.text(s_end, l_end,
                f"{idx}",
                color='black', fontsize=9,
                ha='left', va='bottom',
                zorder=6)

    # 4) 美化
    ax.set_xlabel("s (m)")
    ax.set_ylabel("l (m)")
    ax.set_title("Trajectories Through Final-Step Corridors")
    ax.legend(loc='upper right')
    ax.grid(linestyle='--', alpha=0.4)

    # 5) 固定坐标范围
    s_min = min(n.p_lon_min for n in nodes) - 1
    s_max = max(n.p_lon_max for n in nodes) + 1
    l_min = min(n.p_lat_min for n in nodes) - 1
    l_max = max(n.p_lat_max for n in nodes) + 1
    ax.set_xlim(s_min, s_max)
    ax.set_ylim(l_min, l_max)
    plt.axis('equal')          # ← 这一行保证 x、y 轴等比例

    plt.tight_layout()
    plt.show()

import os
def plot_gp_mean_in_corridor(gp_models,
                             step_idx: int,
                             corridors,
                             preview_traj,
                             best_traj,
                             save_dir: str,
                             s_lim=(5, 40), l_lim=(-5, 5),
                             n_grid=80,
                             cmap='viridis'):

    gp = gp_models[step_idx]
    # 1) 生成 (s, l) 网格
    s_vals = np.linspace(s_lim[0], s_lim[1], n_grid)
    l_vals = np.linspace(l_lim[0], l_lim[1], n_grid)
    Sg, Lg = np.meshgrid(s_vals, l_vals)
    pts = np.column_stack([Sg.ravel(), Lg.ravel()])

    # 2) 预测 GP 均值
    mu = gp.predict(pts).reshape(Sg.shape)

    # 3) 构造遮罩：只保留落到任意一个 corridor node 的矩形内的点
    mask = np.zeros_like(mu, dtype=bool)
    nodes = corridors[0].reach_nodes_at_step(step_idx)
    for node in nodes:
        # 每个 node 对应一个矩形 [s_lo, s_hi] x [l_lo, l_hi]
        s_lo, l_lo = node.p_lon_min, node.p_lat_min
        s_hi, l_hi = node.p_lon_max, node.p_lat_max
        # 在网格上打 mask
        mask |= ((Sg >= s_lo) & (Sg <= s_hi)
                 & (Lg >= l_lo) & (Lg <= l_hi))

    # 4) 只在 mask 区域显示值，其他置为 nan
    mu_masked = np.where(mask, mu, np.nan)

    # 5) 绘图
    fig, ax = plt.subplots(figsize=(20, 15))
    cf = ax.contourf(Sg, Lg, mu_masked, levels=30, cmap=cmap,
                     vmin=np.nanpercentile(mu_masked, 5),
                     vmax=np.nanpercentile(mu_masked, 95))
    ax.set_title(f'Step {step_idx}: GP mean of $S_{{total}}$ (masked)')
    ax.set_xlabel('s (m)')
    ax.set_ylabel('l (m)')
    ax.set_aspect('equal', 'box')
    fig.colorbar(cf, ax=ax, label='$S_{total}$ mean')

    # 6) 叠加走廊边界
    for node in nodes:
        s_lo, l_lo = node.p_lon_min, node.p_lat_min
        width  = node.p_lon_max - s_lo
        height = node.p_lat_max - l_lo
        rect = patches.Rectangle((s_lo, l_lo), width, height,
                                 fill=False, edgecolor='black',
                                 linewidth=1.0, linestyle='--')
        ax.add_patch(rect)

    # 7) 找出最小值点并标记
    #    注意排除 nan

    # idx = np.nanargmin(mu_masked)
    # i_min, j_min = np.unravel_index(idx, mu_masked.shape)
    # s_min = Sg[i_min, j_min]
    # l_min = Lg[i_min, j_min]
    # val_min = mu_masked[i_min, j_min]

    ax.scatter(best_traj['s'][step_idx], best_traj['l'][step_idx],
               marker='*', color='red', s=50,)

    # # --- 6) 预瞄轨迹点 ---
    s_p, l_p = preview_traj['s'][step_idx], preview_traj['l'][step_idx]
    ax.scatter([s_p], [l_p],
               marker='o', color='yellow', edgecolor='k', s=80,)

    # ax.legend(loc='upper right')

    plt.tight_layout()
    fname = os.path.join(save_dir, f"step_{step_idx:03d}.png")
    fig.savefig(fname, dpi=200)
    plt.close(fig)
