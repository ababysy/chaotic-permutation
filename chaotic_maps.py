"""
混沌映射实现模块

支持的混沌映射：
- Logistic 映射:   x_{n+1} = μ * x_n * (1 - x_n)
- Tent 映射:       x_{n+1} = μ * min(x_n, 1 - x_n)
- Sine 映射:       x_{n+1} = μ * sin(π * x_n)
- Chebyshev 映射:  x_{n+1} = cos(k * arccos(x_n))

每种映射都经过参数调优，确保处于强混沌状态。
"""

import math
from abc import ABC, abstractmethod


class ChaoticMap(ABC):
    """混沌映射的抽象基类。

    所有混沌映射都需要实现 iterate() 方法，
    并指定定义域 domain，用于验证种子的合法性。
    """

    def __init__(self, name: str, param_name: str, param_value: float,
                 domain: tuple):
        """
        参数:
            name: 映射名称（如 "Logistic"）
            param_name: 参数名（如 "μ"）
            param_value: 参数值（如 3.99）
            domain: 有效输入区间 (lo, hi)
        """
        self.name = name
        self.param_name = param_name
        self.param_value = param_value
        self.domain = domain

    @abstractmethod
    def iterate(self, x: float) -> float:
        """执行一次迭代映射。"""
        pass

    def validate_seed(self, x0: float):
        """验证种子值是否在定义域内。"""
        lo, hi = self.domain
        if not (lo < x0 < hi):
            raise ValueError(
                f"[{self.name}] 种子必须在 ({lo}, {hi}) 内，得到 {x0}"
            )

    def __repr__(self) -> str:
        return f"{self.name}({self.param_name}={self.param_value})"


# ============================================================
# Logistic 映射
# ============================================================
class LogisticMap(ChaoticMap):
    """Logistic 映射: x_{n+1} = μ * x_n * (1 - x_n)

    当 3.57 < μ ≤ 4 时系统进入混沌状态。
    默认 μ = 3.99，处于高度混沌区域。
    定义域: (0, 1)

    注意：
    - μ 越接近 4，混沌特性越强
    - μ = 4 时具有满映射性质，但存在周期窗口
    - 不变分布为 Beta(1/2, 1/2)，在两端密度较高
    """

    def __init__(self, mu: float = 3.99):
        super().__init__("Logistic", "μ", mu, (0, 1))
        self.mu = mu

    def iterate(self, x: float) -> float:
        return self.mu * x * (1.0 - x)


# ============================================================
# Tent 映射（帐篷映射）
# ============================================================
class TentMap(ChaoticMap):
    """Tent 映射: x_{n+1} = μ * min(x_n, 1 - x_n)

    当 1 < μ ≤ 2 时呈现混沌特性。
    默认 μ = 1.99，接近满帐篷（μ=2）。
    定义域: (0, 1)

    特点：
    - 分段线性，计算简单
    - Lyapunov 指数 = ln(μ)，μ=2 时为 ln(2)
    - 具有均匀的不变分布（当 μ=2 时严格均匀）
    """

    def __init__(self, mu: float = 1.99):
        super().__init__("Tent", "μ", mu, (0, 1))
        self.mu = mu

    def iterate(self, x: float) -> float:
        if x < 0.5:
            return self.mu * x
        else:
            return self.mu * (1.0 - x)


# ============================================================
# Sine 映射
# ============================================================
class SineMap(ChaoticMap):
    """Sine 映射: x_{n+1} = μ * sin(π * x_n)

    当 0.87 < μ ≤ 1 时呈现混沌。
    默认 μ = 0.99。
    定义域: (0, 1)

    特点：
    - 与 Logistic 映射拓扑共轭
    - 具有类似的分岔结构
    - 轨道行为更加光滑（C∞ 而非 C⁰）
    """

    def __init__(self, mu: float = 0.99):
        super().__init__("Sine", "μ", mu, (0, 1))
        self.mu = mu

    def iterate(self, x: float) -> float:
        return self.mu * math.sin(math.pi * x)


# ============================================================
# Chebyshev 映射
# ============================================================
class ChebyshevMap(ChaoticMap):
    """Chebyshev 映射: x_{n+1} = cos(k * arccos(x_n))

    当 k ≥ 2 时呈现混沌。
    默认 k = 4。
    定义域: (-1, 1)

    特点：
    - 具有显式的代数结构（与 Chebyshev 多项式相关）
    - 不变分布为 1/(π√(1-x²))（当 k ≥ 2 时）
    - Lyapunov 指数 = ln(k)
    - 具有良好的密码学特性

    注意：由于浮点误差，对 x 进行 clamp 防止 arccos 参数越界。
    """

    def __init__(self, k: int = 4):
        super().__init__("Chebyshev", "k", float(k), (-1, 1))
        self.k = k

    def iterate(self, x: float) -> float:
        # 防止浮点误差导致 |x| > 1（使 arccos 参数合法）
        x = max(-1.0, min(1.0, x))
        return math.cos(self.k * math.acos(x))


# ============================================================
# 工厂函数
# ============================================================
def get_default_maps() -> list:
    """返回默认配置的四种混沌映射实例列表。"""
    return [
        LogisticMap(mu=3.99),
        TentMap(mu=1.99),
        SineMap(mu=0.99),
        ChebyshevMap(k=4),
    ]


def get_map_by_name(name: str):
    """根据名称获取映射实例。"""
    mapping = {
        'logistic': LogisticMap,
        'tent': TentMap,
        'sine': SineMap,
        'chebyshev': ChebyshevMap,
    }
    cls = mapping.get(name.lower())
    if cls is None:
        raise ValueError(f"未知映射: {name}，可选: {list(mapping.keys())}")
    return cls()
