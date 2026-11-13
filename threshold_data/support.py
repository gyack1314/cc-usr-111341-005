"""评分支撑集诊断：识别离散评分与大量并列。

本模块只做识别与分流，绝不给评分加噪声、不做任何“抖动变连续”的处理。
离散支撑下连续 RDD 的局部线性回归没有可识别的密度连续前提，分析转入
randomization 模块的 Fisher 随机化推断；样本不足时明确停止并说明原因。
"""
from collections import Counter

DEFAULT_UNIQUE_LIMIT = 10
DEFAULT_TIE_FRACTION = 0.2
MIN_POINTS_PER_SIDE = 2


def support_summary(records, unique_limit=DEFAULT_UNIQUE_LIMIT,
                    tie_fraction=DEFAULT_TIE_FRACTION):
    """统计阈值距离的支撑集形态。"""
    n = len(records)
    distances = [r['distance'] for r in records]
    counts = Counter(distances)
    unique_values = sorted(counts)
    tie_share = 1.0 - len(unique_values) / n if n else float('nan')
    boundary_share = counts.get(0, 0) / n if n else float('nan')
    left_unique = [v for v in unique_values if v < 0]
    right_unique = [v for v in unique_values if v >= 0]
    discrete = (len(unique_values) <= unique_limit
                or tie_share >= tie_fraction)
    reasons = []
    if len(unique_values) <= unique_limit:
        reasons.append(f'阈值距离仅 {len(unique_values)} 个不同取值'
                       f'（<= {unique_limit}），评分明显离散')
    if tie_share >= tie_fraction:
        reasons.append(f'并列占比 {tie_share:.0%} 达到 {tie_fraction:.0%} 门槛')
    enough_per_side = (len(left_unique) >= MIN_POINTS_PER_SIDE
                       and len(right_unique) >= MIN_POINTS_PER_SIDE)
    return {
        'n': n,
        'unique_distances': unique_values,
        'unique_count': len(unique_values),
        'tie_fraction': tie_share,
        'boundary_cell_share': boundary_share,
        'boundary_cell_n': counts.get(0, 0),
        'left_unique_n': len(left_unique),
        'right_unique_n': len(right_unique),
        'discrete_support': discrete,
        'enough_points_per_side': enough_per_side,
        'reasons': reasons,
        'point_counts': dict(sorted(counts.items())),
    }


def select_window(records, half_width, donut_cells=0):
    """按阈值距离选局部窗口；donut 两侧各删最近的 donut_cells 个格点。

    离散支撑下，左侧最近格点为 -1,-2,...，资格侧最近格点为 0,1,...
    （距离 0 是 inclusive 规则下的首个资格格点）。donut 删除边界格点是
    针对“分数恰好堆到资格侧”的操纵敏感性检验，必须显式报告删了哪些
    格点，而不是静默改变样本。距离单位格点用各自侧的最近次序判断。
    """
    if donut_cells == 0:
        return [r for r in records if abs(r['distance']) <= half_width]
    selected = []
    for record in records:
        distance = record['distance']
        if abs(distance) > half_width:
            continue
        # 两侧各自从阈值向外数最近格点（次序从 1 起）：
        # 非资格侧 -1,-2,...；资格侧 inclusive 时 0,1,2,...
        nearest_order = (-distance if distance < 0 else distance + 1)
        if nearest_order <= donut_cells:
            continue
        selected.append(record)
    return selected


def donut_removed_values(support_summary_info, donut_cells):
    """报告 donut 实际删除的格点距离值，供口径留痕。"""
    if donut_cells == 0:
        return []
    values = support_summary_info['unique_distances']
    left = sorted((v for v in values if v < 0), reverse=True)[:donut_cells]
    right = sorted(v for v in values if v >= 0)[:donut_cells]
    return sorted(left + right)


def default_window_grid(support, max_half_width=6):
    """离散支撑下给出预先固定的整数窗口网格，禁止事后按显著性挑选。"""
    values = support['unique_distances']
    negatives = [abs(v) for v in values if v < 0]
    positives = [v for v in values if v > 0]
    limit = min(max_half_width,
                max(negatives, default=0),
                max(positives, default=0))
    return [h for h in range(1, int(limit) + 1)]
