"""
置乱表生成模块

使用混沌映射生成置乱表的核心算法：
1. 从种子 x0 出发，迭代 M 轮以消除暂态效应
2. 继续迭代 N 轮，得到 N 个混沌轨道值
3. 对 N 个值按升序排序，以每个值在排序后的位置作为其置乱目标位置
4. 返回置乱表 perm[0..N-1]，其中 perm[i] = 元素 i 的新位置

还提供了种子生成、多置乱表批量生成、以及雪崩效应分析工具。
"""

import numpy as np
from chaotic_maps import ChaoticMap


def generate_permutation(chaotic_map: ChaoticMap, x0: float,
                         M: int = 1000, N: int = 256) -> list:
    """使用混沌映射生成单个置乱表。

    算法流程：
    1. 验证种子 x0 在映射定义域内
    2. 执行 M 次迭代，丢弃暂态
    3. 再执行 N 次迭代，记录 N 个混沌值
    4. 将 N 个 (值, 原始索引) 对按值排序
    5. 排序后的位置即为置乱目标: perm[原始索引] = 排序位置

    参数:
        chaotic_map: 混沌映射实例
        x0: 初始种子值（必须在映射定义域内）
        M: 跳过的迭代轮数（消除暂态），默认 1000
        N: 置乱表大小，默认 256

    返回:
        list[int]: 长度为 N 的置乱表，perm[i] 表示元素 i 的新位置

    示例:
        >>> from chaotic_maps import LogisticMap
        >>> cmap = LogisticMap(3.99)
        >>> perm = generate_permutation(cmap, 0.5, M=1000, N=8)
        >>> len(perm)
        8
    """
    chaotic_map.validate_seed(x0)

    x = x0

    # 第 1 阶段：跳过 M 轮（暂态消除）
    for _ in range(M):
        x = chaotic_map.iterate(x)

    # 第 2 阶段：生成 N 个混沌轨道值
    values = []
    for _ in range(N):
        x = chaotic_map.iterate(x)
        values.append(x)

    # 第 3 阶段：按值排序，获取置乱索引
    # 创建 (值, 原始位置) 对
    indexed = [(values[i], i) for i in range(N)]
    # Python 的 sort 是稳定排序；值相等（极端罕见）时保持原序
    indexed.sort(key=lambda pair: pair[0])

    # 构造置乱表: perm[原位置] = 排序后位置
    perm = [0] * N
    for new_pos, (_, orig_idx) in enumerate(indexed):
        perm[orig_idx] = new_pos

    return perm


def generate_permutations(chaotic_map: ChaoticMap, seeds: list,
                          M: int = 1000, N: int = 256) -> list:
    """使用多个种子批量生成置乱表。

    参数:
        chaotic_map: 混沌映射实例
        seeds: 种子列表
        M: 跳过的迭代轮数
        N: 置乱表大小

    返回:
        list[list[int]]: 置乱表列表，顺序与 seeds 对应
    """
    return [generate_permutation(chaotic_map, s, M, N) for s in seeds]


def generate_random_seeds(n: int, domain: tuple = (0, 1),
                          rng: np.random.Generator = None) -> list:
    """在指定定义域内生成 n 个均匀分布的随机种子。

    参数:
        n: 种子数量
        domain: 定义域 (lo, hi)
        rng: numpy 随机数生成器（可选，不传则新建）

    返回:
        list[float]: 种子列表
    """
    if rng is None:
        rng = np.random.default_rng()
    lo, hi = domain
    # 添加小的 epsilon 避免恰好落在边界上
    eps = 1e-15
    lo_safe, hi_safe = lo + eps, hi - eps
    return list(rng.uniform(lo_safe, hi_safe, n))


# ============================================================
# 雪崩效应分析
# ============================================================
def compute_hamming_distance(perm1: list, perm2: list) -> float:
    """计算两个置乱表之间的归一化汉明距离。

    值为 1 表示两个置乱表完全不同（理想雪崩），
    值为 0 表示完全相同。

    参数:
        perm1, perm2: 等长的置乱表

    返回:
        float: 归一化汉明距离 ∈ [0, 1]
    """
    n = len(perm1)
    if n != len(perm2):
        raise ValueError("置乱表长度必须相同")
    diff = sum(1 for i in range(n) if perm1[i] != perm2[i])
    return diff / n


def avalanche_analysis(chaotic_map: ChaoticMap, n_seeds: int = 100,
                       N: int = 256, epsilon: float = 1e-8,
                       M: int = 1000, rng_seed: int = 42) -> dict:
    """雪崩效应（Avalanche Effect）分析。

    对每个种子 s，生成 s 和 s+ε 两个种子对应的置乱表，
    计算两者的汉明距离。
    理想情况下，微小的 ε 应导致完全不同的置乱表（距离 ≈ 1）。

    这是衡量"密钥敏感性"的重要指标。

    参数:
        chaotic_map: 混沌映射实例
        n_seeds: 测试的种子对数
        N: 置乱表大小
        epsilon: 种子的微小扰动
        M: 暂态迭代数
        rng_seed: 随机种子

    返回:
        dict: 包含 mean_distance, std_distance, min_distance, max_distance
    """
    rng = np.random.default_rng(rng_seed)
    lo, hi = chaotic_map.domain
    eps_safe = max(epsilon, 1e-15)

    seeds = generate_random_seeds(n_seeds, chaotic_map.domain, rng)

    distances = []
    for s in seeds:
        # 计算扰动后的种子，确保不越界
        s2 = s + eps_safe
        if s2 >= hi:
            s2 = s - eps_safe  # 向下扰动

        perm1 = generate_permutation(chaotic_map, s, M, N)
        perm2 = generate_permutation(chaotic_map, s2, M, N)
        distances.append(compute_hamming_distance(perm1, perm2))

    return {
        'mean_distance': float(np.mean(distances)),
        'std_distance': float(np.std(distances)),
        'min_distance': float(np.min(distances)),
        'max_distance': float(np.max(distances)),
    }
