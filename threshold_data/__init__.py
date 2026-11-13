"""资格阈值附近的数字化扶持效果识别。

模块组成：
- io/domain       资料读取、身份与引用检查（标准库）
- policy/prep     资格规则、冻结分口径、修订审计、首次有效申请
- support         离散评分与并列识别（不加噪声）
- diagnostics     堆积、基线连续、稀缺、一阶段、伪阈值、donut、窗口
- estimation      局部加权回归、聚类稳健推断、ITT/FS/LATE
- randomization   离散路径的 Fisher 随机化推断
- simulation      独立 DGP 的偏差与覆盖率验证
- pipeline        编排、留痕与报告
"""

__version__ = '0.2.0'
