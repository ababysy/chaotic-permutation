# 混沌置乱的循环阶分析

> Cycle Order Analysis of Chaotic Permutations

基于混沌映射生成置乱表，系统分析其循环结构的密码学特性。

## 项目简介

本项目研究使用混沌映射（Chaotic Maps）构造置乱表（Permutation Table）的方法，
重点分析置乱表的**循环结构**和**置换阶**（Order），评估不同混沌映射在密码学应用中的安全性。

### 核心概念

- **置乱表**：将 N 个元素重新排列的映射表，由混沌序列的排序位置确定
- **循环圈**：置乱表中元素形成的闭合置换链
- **置换阶（Order）**：所有循环圈长度的最小公倍数（LCM），即需要重复应用多少次置乱才能恢复原始排列
- **雪崩效应**：种子（密钥）的微小变化导致置乱表完全不同的程度

## 混沌映射

| 映射 | 公式 | 参数 | 定义域 | Lyapunov 指数 |
|------|------|------|--------|---------------|
| Logistic | $x_{n+1} = \mu x_n(1-x_n)$ | μ=3.99 | (0,1) | ln(3.99) |
| Tent | $x_{n+1} = \mu \min(x_n, 1-x_n)$ | μ=1.99 | (0,1) | ln(1.99) |
| Sine | $x_{n+1} = \mu \sin(\pi x_n)$ | μ=0.99 | (0,1) | ~ln(πμ) |
| Chebyshev | $x_{n+1} = \cos(k \cdot \arccos(x_n))$ | k=4 | (-1,1) | ln(4) |

## 实验结果摘要

### 平均阶 vs N（N=1024，200个种子）

| 映射 | 平均阶 | 中位数阶 | 平均循环圈数 | 平均不动点 |
|------|--------|---------|-------------|-----------|
| Logistic | 5.13×10¹¹ | 2.29×10⁷ | 7.2 | 0.92 |
| Tent | 1.65×10¹² | 7.60×10⁷ | 7.5 | 0.95 |
| Sine | 2.41×10¹¹ | 6.44×10⁷ | 7.5 | 0.91 |
| Chebyshev | 1.31×10¹² | 7.87×10⁷ | 7.6 | 1.00 |
| **随机期望** | ~10³⁶ | — | ~ln(N)+γ ≈ 7.5 | 1.0 |

### 雪崩效应（N=256, ε=1e-8）

所有映射的归一化汉明距离均 **≈ 0.996**，接近理想值 1.0，
表明密钥敏感性极强。

## 项目结构

```
chaotic-permutation/
├── chaotic_maps.py      # 混沌映射实现（Logistic/Tent/Sine/Chebyshev）
├── permutation.py       # 置乱表生成 + 雪崩效应分析
├── cycle_analysis.py    # 循环查找 + 阶的计算 + 统计分析
├── visualization.py     # 图表生成（8张分析图）
├── main.py              # 主程序入口（命令行可配置）
├── requirements.txt     # 依赖（numpy, matplotlib）
├── output/              # 实验结果输出
│   ├── figures/         # 可视化图表
│   ├── results.json     # 统计数据
│   ├── all_orders.json  # 所有阶的值
│   └── analysis_report.md  # 完整实验报告
└── README.md
```

## 快速开始

### 环境要求
- Python 3.8+
- numpy >= 1.21.0
- matplotlib >= 3.5.0

### 安装与运行

```bash
# 安装依赖
pip install -r requirements.txt

# 使用默认配置运行（N=32..1024, 200个种子）
python main.py

# 自定义参数
python main.py --N-values 64,128,256,512 --num-seeds 500

# 仅测试部分映射
python main.py --maps chebyshev,logistic

# 指定输出目录
python main.py --output-dir my_results

# 查看所有选项
python main.py --help
```

### 代码示例

```python
from chaotic_maps import LogisticMap, ChebyshevMap
from permutation import generate_permutation
from cycle_analysis import cycle_statistics

# 创建混沌映射
cmap = LogisticMap(mu=3.99)

# 生成置乱表 (N=256, M=1000, seed=0.5)
perm = generate_permutation(cmap, x0=0.5, M=1000, N=256)

# 分析循环结构
stats = cycle_statistics(perm)
print(f"置换阶: {stats['order']:.2e}")
print(f"循环圈数: {stats['num_cycles']}")
print(f"循环长度: {stats['cycle_lengths']}")
```

## 安全性分析结论

1. **置换阶**: N=1024 时阶达 10¹¹~10¹² 量级，暴力恢复不可行
2. **循环结构**: 短循环比例低，接近随机置换分布，无明显弱点
3. **雪崩效应**: 汉明距离 ~0.996，密钥敏感性优秀
4. **Chebyshev 映射**综合表现最佳（均匀分布 + 大阶 + 高 Lyapunov 指数）
5. **Tent 映射**计算效率最高（仅需乘法和比较，无需三角函数）

## 参考资料

- Strogatz, S. H. (2018). *Nonlinear Dynamics and Chaos*. CRC Press.
- Kocarev, L., & Lian, S. (2011). *Chaos-Based Cryptography*. Springer.
- Landau 函数 g(N): 最大可能的置换阶 (OEIS A000793)

## License

MIT
