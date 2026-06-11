"""
可视化模块

生成混沌置乱循环分析的各类图表：
1. 平均阶 vs N 曲线（对数/线性纵轴）
2. 循环长度分布直方图
3. 阶的分布箱线图
4. 循环圈数 / 最大循环 / 不动点数 vs N
5. 雪崩效应对比图

所有图表保存为 PNG 格式，DPI=150。
"""

import os
import matplotlib
matplotlib.use('Agg')  # 无 GUI 后端，适用于服务器/后台运行
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ============================================================
# 全局样式
# ============================================================
# 为每种混沌映射指定一致的颜色和标记
COLORS = {
    'Logistic':   '#2196F3',  # 蓝色
    'Tent':       '#FF9800',  # 橙色
    'Sine':       '#4CAF50',  # 绿色
    'Chebyshev':  '#E91E63',  # 粉色/玫红
}
MARKERS = {
    'Logistic':   'o',
    'Tent':       's',
    'Sine':       '^',
    'Chebyshev':  'D',
}

# 设置 matplotlib 全局参数
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 15,
    'axes.labelsize': 13,
    'legend.fontsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'axes.unicode_minus': False,
})


def _ensure_dir(path: str):
    """确保文件的目录存在。"""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


# ============================================================
# 图 1: 平均阶 vs N（对数纵轴）
# ============================================================
def plot_average_order_vs_N(results: dict, save_path: str = None):
    """绘制「平均阶 — N」曲线，纵轴使用对数尺度。

    参数:
        results: {map_name: [dict_per_N, ...]}
                 每个 dict_per_N 需含 'N', 'avg_order', 'std_order'
        save_path: 保存路径（可选）
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    for name, data in results.items():
        Ns = np.array([d['N'] for d in data])
        orders = np.array([d['avg_order'] for d in data])
        stds = np.array([d['std_order'] for d in data])

        color = COLORS.get(name, '#333333')
        marker = MARKERS.get(name, 'o')

        ax.errorbar(Ns, orders, yerr=stds,
                    marker=marker, color=color, label=name,
                    linewidth=2.2, markersize=9, capsize=5,
                    alpha=0.88, markeredgecolor='white', markeredgewidth=0.5)

    ax.set_xlabel('N（置乱表大小）')
    ax.set_ylabel('平均阶（log scale）')
    ax.set_title('不同混沌映射的平均置换阶 vs 置乱表大小 N')
    ax.set_yscale('log')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(left=0)

    fig.tight_layout()
    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path)
    plt.close(fig)


# ============================================================
# 图 2: 平均阶 vs N（线性纵轴）
# ============================================================
def plot_average_order_vs_N_linear(results: dict, save_path: str = None):
    """绘制「平均阶 — N」曲线，纵轴线性尺度。"""
    fig, ax = plt.subplots(figsize=(12, 7))

    for name, data in results.items():
        Ns = np.array([d['N'] for d in data])
        orders = np.array([d['avg_order'] for d in data])

        color = COLORS.get(name, '#333333')
        marker = MARKERS.get(name, 'o')

        ax.plot(Ns, orders, marker=marker, color=color, label=name,
                linewidth=2.2, markersize=9, alpha=0.88)

    ax.set_xlabel('N（置乱表大小）')
    ax.set_ylabel('平均阶')
    ax.set_title('不同混沌映射的平均置换阶 vs N（线性尺度）')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    fig.tight_layout()
    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path)
    plt.close(fig)


# ============================================================
# 图 3: 循环长度分布
# ============================================================
def plot_cycle_length_distribution(all_stats_by_map: dict, N: int,
                                   save_path: str = None):
    """绘制指定 N 下的循环长度分布（每种映射一个子图）。

    参数:
        all_stats_by_map: {map_name: aggregate_statistics 结果}
        N: 当前 N 值（仅用于标题）
        save_path: 保存路径
    """
    n_maps = len(all_stats_by_map)
    if n_maps == 0:
        return

    fig, axes = plt.subplots(1, n_maps, figsize=(5.5 * n_maps, 4.5),
                             squeeze=False)
    axes = axes[0]  # 1D array

    for ax, (name, agg) in zip(axes, all_stats_by_map.items()):
        dist = agg['length_distribution']
        if not dist:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            continue

        lengths = list(dist.keys())
        # 归一化：每个样本的平均出现次数
        counts = [c / agg['num_samples'] for c in dist.values()]

        color = COLORS.get(name, '#333333')
        ax.bar(lengths, counts, color=color, alpha=0.75,
               edgecolor='white', linewidth=0.3)

        ax.set_xlabel('循环长度')
        ax.set_ylabel('每样本平均次数')
        ax.set_title(f'{name}（N={N}）')
        ax.set_xlim(0, N + 1)
        ax.grid(True, alpha=0.2, axis='y')

    fig.suptitle(f'循环长度分布（N = {N}）', fontsize=15, y=1.01)
    fig.tight_layout()
    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path)
    plt.close(fig)


# ============================================================
# 图 4: 阶的箱线图
# ============================================================
def plot_order_boxplot(results: dict, N_value: int,
                       save_path: str = None):
    """对某个固定 N 画出所有映射的阶的箱线图。

    参数:
        results: {map_name: [dict_per_N, ...]}，每个 dict 需含 'N' 和 'all_orders'
        N_value: 要绘制的 N
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    data_list = []
    labels = []
    color_list = []

    for name, map_data in results.items():
        for d in map_data:
            if d['N'] == N_value and 'all_orders' in d:
                data_list.append(d['all_orders'])
                labels.append(name)
                color_list.append(COLORS.get(name, '#333333'))
                break

    if not data_list:
        plt.close(fig)
        return

    bp = ax.boxplot(data_list, labels=labels, patch_artist=True,
                    showfliers=True, widths=0.5)

    for patch, color in zip(bp['boxes'], color_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)

    ax.set_ylabel('阶（log scale）')
    ax.set_title(f'置换阶的分布（N = {N_value}）')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path)
    plt.close(fig)


# ============================================================
# 图 5: 循环圈数 vs N
# ============================================================
def plot_num_cycles_vs_N(results: dict, save_path: str = None):
    """绘制平均循环圈数 vs N。"""
    fig, ax = plt.subplots(figsize=(12, 7))

    for name, data in results.items():
        Ns = np.array([d['N'] for d in data])
        cycles = np.array([d['avg_num_cycles'] for d in data])

        color = COLORS.get(name, '#333333')
        marker = MARKERS.get(name, 'o')

        ax.plot(Ns, cycles, marker=marker, color=color, label=name,
                linewidth=2.2, markersize=9, alpha=0.88)

    # 参考线：随机置换的期望 ≈ ln(N) + γ (Harmonic number approx)
    N_ref = np.linspace(min(Ns), max(Ns), 100)
    expected = np.log(N_ref) + np.euler_gamma
    ax.plot(N_ref, expected, 'k--', alpha=0.35, linewidth=1.2,
            label=r'随机期望 $\approx \ln N + \gamma$')

    ax.set_xlabel('N（置乱表大小）')
    ax.set_ylabel('平均循环圈数')
    ax.set_title('平均循环圈数 vs N')
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path)
    plt.close(fig)


# ============================================================
# 图 6: 最大循环长度 vs N
# ============================================================
def plot_max_cycle_vs_N(results: dict, save_path: str = None):
    """绘制平均最大循环长度 vs N。"""
    fig, ax = plt.subplots(figsize=(12, 7))

    for name, data in results.items():
        Ns = np.array([d['N'] for d in data])
        max_cyc = np.array([d['avg_max_cycle'] for d in data])

        color = COLORS.get(name, '#333333')
        marker = MARKERS.get(name, 'o')

        ax.plot(Ns, max_cyc, marker=marker, color=color, label=name,
                linewidth=2.2, markersize=9, alpha=0.88)

    # 上界参考线 y = N
    ax.plot(Ns, Ns, 'k--', alpha=0.25, linewidth=1,
            label='y = N（理论上界）')

    # 随机期望：对于随机置换，最长循环的期望 ≈ 0.62433 * N (Golomb-Dickman 常数)
    golomb_dickman = 0.6243299885
    ax.plot(Ns, golomb_dickman * Ns, 'gray', alpha=0.35, linewidth=1,
            linestyle=':', label=f'随机期望 ≈ {golomb_dickman:.3f} N')

    ax.set_xlabel('N（置乱表大小）')
    ax.set_ylabel('平均最大循环长度')
    ax.set_title('平均最大循环长度 vs N')
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path)
    plt.close(fig)


# ============================================================
# 图 7: 不动点数 vs N
# ============================================================
def plot_fixed_points_vs_N(results: dict, save_path: str = None):
    """绘制平均不动点（1-cycles）数量 vs N。"""
    fig, ax = plt.subplots(figsize=(12, 7))

    for name, data in results.items():
        Ns = np.array([d['N'] for d in data])
        fp = np.array([d['avg_fixed_points'] for d in data])

        color = COLORS.get(name, '#333333')
        marker = MARKERS.get(name, 'o')

        ax.plot(Ns, fp, marker=marker, color=color, label=name,
                linewidth=2.2, markersize=9, alpha=0.88)

    # 随机置换的期望不动点数 = 1
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5,
               label='随机期望 = 1')

    ax.set_xlabel('N（置乱表大小）')
    ax.set_ylabel('平均不动点数')
    ax.set_title('平均不动点数（1-cycles）vs N')
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path)
    plt.close(fig)


# ============================================================
# 图 8: 雪崩效应对比
# ============================================================
def plot_avalanche_comparison(avalanche: dict, save_path: str = None):
    """绘制各映射的雪崩效应（汉明距离）对比图。"""
    fig, ax = plt.subplots(figsize=(9, 6))

    names = list(avalanche.keys())
    means = [avalanche[n]['mean_distance'] for n in names]
    stds = [avalanche[n]['std_distance'] for n in names]
    colors_bar = [COLORS.get(n, '#333333') for n in names]

    x_pos = np.arange(len(names))
    bars = ax.bar(x_pos, means, yerr=stds, color=colors_bar,
                  alpha=0.75, capsize=8, edgecolor='white', linewidth=0.5)

    # 理想参考线
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5,
               label='理想值 1.0（完全雪崩）')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(names)
    ax.set_ylabel('归一化汉明距离')
    ax.set_title('雪崩效应分析（种子扰动 ε = 1e-8）')
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.2, axis='y')
    ax.set_ylim(0, 1.15)

    for bar, mean_val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{mean_val:.4f}', ha='center', va='bottom', fontsize=11)

    fig.tight_layout()
    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path)
    plt.close(fig)


# ============================================================
# 综合仪表板
# ============================================================
def generate_all_plots(results: dict, N_fixed: int,
                       all_stats_by_map: dict,
                       avalanche: dict,
                       save_dir: str = 'output/figures'):
    """一键生成所有分析图表。

    参数:
        results: 主实验结果 {map_name: [per_N_data]}
        N_fixed: 用于单点分析的 N 值
        all_stats_by_map: {map_name: {N: aggregate_stats}}
        avalanche: 雪崩分析结果 {map_name: dict}
        save_dir: 图表保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    generated = []

    def _save(name, func, *args):
        path = os.path.join(save_dir, name)
        func(*args, save_path=path)
        generated.append(path)

    # 1. 平均阶 vs N（对数）
    _save('01_avg_order_vs_N_log.png', plot_average_order_vs_N, results)

    # 2. 平均阶 vs N（线性）
    _save('02_avg_order_vs_N_linear.png', plot_average_order_vs_N_linear, results)

    # 3. 循环长度分布（固定 N）
    stats_at_N = {name: s[N_fixed] for name, s in all_stats_by_map.items()
                  if N_fixed in s}
    _save('03_cycle_length_distribution.png',
          plot_cycle_length_distribution, stats_at_N, N_fixed)

    # 4. 阶的箱线图
    _save('04_order_boxplot.png', plot_order_boxplot, results, N_fixed)

    # 5. 循环圈数 vs N
    _save('05_num_cycles_vs_N.png', plot_num_cycles_vs_N, results)

    # 6. 最大循环长度 vs N
    _save('06_max_cycle_vs_N.png', plot_max_cycle_vs_N, results)

    # 7. 不动点 vs N
    _save('07_fixed_points_vs_N.png', plot_fixed_points_vs_N, results)

    # 8. 雪崩效应
    if avalanche:
        _save('08_avalanche_effect.png', plot_avalanche_comparison, avalanche)

    print(f"\n  [OK] 共生成 {len(generated)} 张图表:")
    for p in generated:
        print(f"     {p}")
