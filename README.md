# 资格阈值附近的数字化扶持效果识别

数字化扶持按企业上期评分划定资格，实际领取还要经过申请和审核。本项目
在保留原有资料身份检查的基础上，实现阈值附近的**模糊断点（fuzzy RDD）
效果识别**：先做诊断，再给效果解释，并在评分离散时切换到 Fisher 随机化
推断。全程仅使用 Python 3.11 标准库，无需联网或额外依赖。

## 识别纪律

- **资格口径**：历史资格只使用公布前冻结的连续评分，后补修订分仅审计、
  永不覆盖；支持每批次独立的阈值、资格方向与等号规则。
- **样本口径**：主分析取每家企业首次有效申请，其他时期进辅助样本；
  结果有固定观察窗与资料可见时间。
- **先诊断后解释**：评分堆积（Fisher 精确检验）、基线连续性、样本
  稀缺、第一阶段跳跃、伪阈值、donut、不同局部窗口逐一检查。
- **离散评分不加噪声**：不同取值过少或并列严重时，改用批次内分层
  Fisher 随机化推断；窗口只按政策前基线平衡预选，不看结果挑窗口。
- **多批次保留结构**：批次固定效应 + 批次特定斜率 + 逐批次跳跃，不把
  异质处理强度拼成一条曲线。
- **三个跳跃分开报告**：资格对结果的 ITT、资格对领取的第一阶段、
  假设成立时两者比值的局部 LATE；第一阶段弱或方向反常就停止报告精确
  比值，不沿用普通标准误（按 firm 聚类 + wild cluster bootstrap）。
- **不挑显著带宽**：窗口×权重×donut 全网格全部估计并留痕。
- **证据边界**：诊断通过不等于证明了无操纵、排除限制或单调性；结论只
  覆盖阈值附近合规企业，不能单独回答“扩围”的整体效果。

## 模块

| 模块 | 职责 |
| --- | --- |
| `io` / `domain` | 资料读取、SHA256 身份与引用检查 |
| `policy` / `prep` | 资格规则、阈值距离、冻结分审计、首次有效申请、观察窗 |
| `support` | 离散评分与并列识别、窗口与 donut 口径 |
| `diagnostics` | 堆积、基线、稀缺、一阶段、伪阈值、敏感性网格 |
| `estimation` | 局部加权回归、聚类稳健推断、ITT/FS/LATE 与弱一阶段闸门 |
| `randomization` | 离散路径的 Fisher 随机化推断与检验反演区间 |
| `simulation` | 四类独立 DGP 的偏差、覆盖率与报警率验证 |
| `pipeline` | 编排、纳入清单/逐点权重/口径留痕与 Markdown 报告 |
| `statsutil` | 纯标准库线性代数、t/F 分布、Fisher 检验、wild bootstrap |

方法细节见 [docs/methodology.md](docs/methodology.md)，资料字段见
[docs/data-contract.md](docs/data-contract.md)。

## 运行

建议使用隔离模式，防止资料目录中被植入的同名模块被导入：

```sh
# 单元测试（60+ 用例）
python -I run_tests.py

# 语法编译检查
python -I -m compileall -q threshold_data tests run_tests.py run_analysis.py

# 资料身份检查（兼容旧入口）
python -I run_analysis.py check data/normal.json
python -I run_analysis.py data/normal.json

# 完整诊断与效果识别，产物写入 out/<包名>/
python -I run_analysis.py analyze data/normal.json --out out
python -I run_analysis.py analyze data/boundary.json --out out

# 独立模拟验证偏差与区间覆盖
python -I run_analysis.py simulate --reps 200
```

`analyze` 产物：`included_firms.jsonl`（纳入企业与逐点权重）、
`exclusions.json`（未纳入记录及原因）、`revisions_audit.json`（修订分
审计）、`diagnostics.json`、`estimates.json`、`sensitivity.csv`（全口径
网格）、`report.md` / `report.json`。

## 两个资料包的预期结论

- `normal`：整数评分判为离散支撑，走 Fisher 随机化路径；修订分 a0 会
  改变资格但冻结分仍判不合格；第一阶段在小样本下不显著，停止报告
  LATE，只保留 ITT 与领取率证据。
- `boundary`：存在资格规则但**领取率完全不跳跃**，第一阶段标记为
  弱/反常，明确不报 LATE——用于核对“有资格跳跃却无领取跳跃”的情形。

容器内沿用原有镜像：

```sh
docker build -t threshold-data .
docker run --rm threshold-data
```
