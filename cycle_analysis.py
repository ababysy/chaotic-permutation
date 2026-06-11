"""
循环结构分析模块

分析置乱表的循环（cycle）结构，包括：
- 找出所有循环圈
- 统计每种长度的循环圈数量
- 计算置换的阶（order）：所有循环圈长度的最小公倍数（LCM）
- 聚合多个置乱表的统计数据

数学背景：
  一个 N 元置换 σ 可唯一分解为若干不相交的循环圈。
  置换的阶 ord(σ) = LCM(各循环圈长度)，满足 σ^ord(σ) = id。
  这是衡量"需要应用置乱多少次才能回到原始排列"的指标。
"""

import math
from collections import Counter
import numpy as np


# ============================================================
# 循环查找
# ============================================================
def find_cycles(perm: list) -> list:
    """在置乱表中寻找所有不相交的循环圈。

    算法：遍历每个未访问的元素，沿置乱链追踪直到回到起点。
    时间复杂度 O(N)，空间复杂度 O(N)。

    参数:
        perm: 置乱表，perm[i] = 元素 i 的目标位置

    返回:
        list[list[int]]: 循环圈列表，每个内层列表是一个循环圈（顶点序列）

    示例:
        >>> find_cycles([1, 2, 0])  # 一个 3-循环
        [[0, 1, 2]]
        >>> find_cycles([1, 0, 2])  # 一个 2-循环 + 一个不动点
        [[0, 1], [2]]
    """
    n = len(perm)
    visited = [False] * n
    cycles = []

    for i in range(n):
        if not visited[i]:
            cycle = []
            j = i
            while not visited[j]:
                visited[j] = True
                cycle.append(j)
                j = perm[j]
            cycles.append(cycle)

    return cycles


# ============================================================
# 阶的计算
# ============================================================
def gcd(a: int, b: int) -> int:
    """最大公约数。"""
    return math.gcd(a, b)


def lcm(a: int, b: int) -> int:
    """最小公倍数。"""
    return a * b // gcd(a, b)


def compute_order(cycles: list) -> int:
    """计算置换的阶（order）。

    阶 = 所有循环圈长度的最小公倍数 (LCM)。
    这是使 σ^k = id 的最小正整数 k。

    对于大 N，阶可能是天文数字（如 N=1024 时可达 10^36 量级）。
    Python 的任意精度整数可正确处理。

    参数:
        cycles: find_cycles() 返回的循环圈列表

    返回:
        int: 置换的阶

    示例:
        >>> compute_order([[0, 1, 2]])  # 单一 3-循环
        3
        >>> compute_order([[0, 1], [2, 3, 4, 5]])  # 2-循环 + 4-循环
        4
    """
    order = 1
    for cycle in cycles:
        L = len(cycle)
        # order = lcm(order, L)
        # 用 math.gcd 直接计算，避免函数调用开销
        order = order * L // math.gcd(order, L)
    return order


# ============================================================
# 单表统计
# ============================================================
def cycle_statistics(perm: list) -> dict:
    """计算单个置乱表的完整循环统计信息。

    参数:
        perm: 置乱表

    返回:
        dict: 包含以下键的字典
            - cycles: 循环圈列表
            - cycle_lengths: 各循环圈长度
            - num_cycles: 循环圈总数
            - order: 置换的阶 (LCM)
            - length_distribution: {长度: 出现次数}（已排序）
            - max_cycle: 最长循环圈的长度
            - min_cycle: 最短循环圈的长度
            - mean_cycle: 平均循环圈长度
            - fixed_points: 不动点（1-循环）的数量
    """
    cycles = find_cycles(perm)
    lengths = [len(c) for c in cycles]
    dist = Counter(lengths)

    return {
        'cycles': cycles,
        'cycle_lengths': lengths,
        'num_cycles': len(cycles),
        'order': compute_order(cycles),
        'length_distribution': dict(sorted(dist.items())),
        'max_cycle': max(lengths) if lengths else 0,
        'min_cycle': min(lengths) if lengths else 0,
        'mean_cycle': float(np.mean(lengths)) if lengths else 0.0,
        'fixed_points': dist.get(1, 0),  # 1-cycles = 不动点
    }


# ============================================================
# 聚合统计
# ============================================================
def aggregate_statistics(perms: list, verbose: bool = False) -> dict:
    """聚合多个置乱表（同 N，不同种子）的循环统计信息。

    用于评估在固定 N 下，不同种子生成的置乱表的整体循环特性。

    参数:
        perms: 置乱表列表（所有置乱表长度相同）
        verbose: 是否打印进度

    返回:
        dict: 聚合统计信息，包含
            - N: 置乱表大小
            - num_samples: 样本数
            - avg_order / std_order / median_order / max_order / min_order
            - avg_num_cycles / std_num_cycles
            - avg_max_cycle
            - avg_fixed_points
            - length_distribution: 所有样本的循环长度聚合分布
            - all_orders: 所有阶的列表（用于箱线图等）
            - all_stats: 每个置乱表的完整统计（用于深入分析）
    """
    all_stats = []
    for i, perm in enumerate(perms):
        if verbose and (i + 1) % 50 == 0:
            print(f"    分析进度: {i + 1}/{len(perms)}")
        all_stats.append(cycle_statistics(perm))

    N = len(perms[0])
    orders = [s['order'] for s in all_stats]
    num_cycles = [s['num_cycles'] for s in all_stats]
    max_cycles = [s['max_cycle'] for s in all_stats]
    fixed_points = [s['fixed_points'] for s in all_stats]

    # 聚合循环长度分布（跨所有样本）
    combined_lengths = []
    for s in all_stats:
        combined_lengths.extend(s['cycle_lengths'])
    length_dist = Counter(combined_lengths)

    return {
        'N': N,
        'num_samples': len(perms),
        'avg_order': float(np.mean(orders)),
        'std_order': float(np.std(orders)),
        'median_order': float(np.median(orders)),
        'max_order': int(max(orders)),
        'min_order': int(min(orders)),
        'avg_num_cycles': float(np.mean(num_cycles)),
        'std_num_cycles': float(np.std(num_cycles)),
        'avg_max_cycle': float(np.mean(max_cycles)),
        'avg_fixed_points': float(np.mean(fixed_points)),
        'length_distribution': dict(sorted(length_dist.items())),
        'all_orders': orders,
        'all_stats': all_stats,
    }


def print_statistics(agg: dict, map_name: str):
    """格式化打印聚合统计信息。"""
    print(f"\n{'=' * 62}")
    print(f"  {map_name} 映射 — 循环分析结果 (N = {agg['N']})")
    print(f"{'=' * 62}")
    print(f"  样本数:              {agg['num_samples']}")
    print(f"  平均阶 (avg order):  {agg['avg_order']:.4e}")
    print(f"  阶的标准差:          {agg['std_order']:.4e}")
    print(f"  阶的中位数:          {agg['median_order']:.4e}")
    print(f"  最大阶:              {agg['max_order']:.4e}")
    print(f"  最小阶:              {agg['min_order']}")
    print(f"  平均循环圈数:        {agg['avg_num_cycles']:.2f}")
    print(f"  循环圈数标准差:      {agg['std_num_cycles']:.2f}")
    print(f"  平均最大圈长度:      {agg['avg_max_cycle']:.2f}")
    print(f"  平均不动点数:        {agg['avg_fixed_points']:.2f}")
    print(f"  循环长度分布 (长度: 次数):")
    dist_items = list(agg['length_distribution'].items())
    # 如果分布太长，分段展示
    if len(dist_items) <= 20:
        for length, count in dist_items:
            bar = '█' * min(count, 50)
            print(f"    {length:4d}: {count:4d}  {bar}")
    else:
        print(f"    （共 {len(dist_items)} 种不同长度，仅显示前 20 种）")
        for length, count in dist_items[:20]:
            bar = '█' * min(count, 50)
            print(f"    {length:4d}: {count:4d}  {bar}")
        print(f"    ...")
    print(f"{'=' * 62}")
