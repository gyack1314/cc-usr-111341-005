"""阈值诊断：先诊断、后解释。

任何效果解释之前必须先回答：
1. 阈值两侧有没有评分堆积（密度不连续）；
2. 基线变量在阈值处是否连续；
3. 局部样本是否稀缺到无法支撑推断；
4. 领取率在阈值处到底跳不跳（第一阶段）；
5. 伪阈值、donut 与不同窗口会不会改变判断。

离散格点使用 Fisher 精确检验等有依据的推断，不做加噪声伪装连续。
诊断通过也只能说明“未发现违背”，不能声称精确操纵、排除限制或单调性
已被证明——这些假设本身不可由现有数据检验。
"""
from collections import Counter
from . import statsutil as su
from . import estimation as est
from .support import select_window, donut_removed_values


def cell_counts(records):
    """每个阈值距离格点的样本数、领取数与资格侧标记。"""
    grouped = {}
    for r in records:
        cell = grouped.setdefault(r['distance'], {
            'distance': r['distance'], 'eligible': r['eligible'],
            'n': 0, 'received': 0, 'outcome_sum': 0.0,
            'baseline_sum': 0.0})
        cell['n'] += 1
        cell['received'] += 1 if r['received'] else 0
        cell['outcome_sum'] += r['outcome']
        cell['baseline_sum'] += r['baseline']
    for cell in grouped.values():
        cell['outcome_mean'] = cell['outcome_sum'] / cell['n']
        cell['baseline_mean'] = cell['baseline_sum'] / cell['n']
        cell['takeup_rate'] = cell['received'] / cell['n']
    return [grouped[k] for k in sorted(grouped)]


def bunching_test(records, half_width):
    """比较窗口内两侧格点总数，Fisher 精确检验密度是否不连续。"""
    sub = select_window(records, half_width)
    left_n = sum(1 for r in sub if not r['eligible'])
    right_n = len(sub) - left_n
    # 用全样本两侧总量构造 2x2：窗口内 vs 窗口外，检验阈值附近是否异常。
    out_left = sum(1 for r in records if not r['eligible']) - left_n
    out_right = sum(1 for r in records if r['eligible']) - right_n
    table = [[right_n, left_n], [out_right, out_left]]
    p_value = su.fisher_exact_two_sided(table)
    boundary = [c for c in cell_counts(sub) if c['distance'] == 0]
    return {
        'half_width': half_width,
        'left_n': left_n,
        'right_n': right_n,
        'boundary_cell_n': boundary[0]['n'] if boundary else 0,
        'fisher_p': p_value,
        'table': table,
        'flag': p_value < 0.05,
    }


def baseline_continuity_test(records, half_width, weighting='uniform'):
    """把政策前 baseline 当结果跑同一局部模型，检验阈值处是否连续。"""
    sub = select_window(records, half_width)
    if len({r['eligible'] for r in sub}) < 2:
        return {'status': 'not_identifiable', 'half_width': half_width}
    fit, _ = est._fit_outcome(sub, half_width, weighting, 'baseline',
                              bootstrap=False)
    return {
        'half_width': half_width,
        'baseline_jump': fit['estimate'],
        'se': fit['se'],
        'p_value': fit['p_value'],
        'ci95': fit['ci95'],
        'flag': fit['p_value'] < 0.05,
        'status': 'ok',
    }


def scarcity_check(records, half_width, min_per_side=5):
    """样本稀缺：两侧计数、独立企业数、每格点平均样本。"""
    sub = select_window(records, half_width)
    left = [r for r in sub if not r['eligible']]
    right = [r for r in sub if r['eligible']]
    distinct = len({r['distance'] for r in sub})
    firms = len({r['firm'] for r in sub})
    sparse = (len(left) < min_per_side or len(right) < min_per_side
              or firms < min_per_side)
    return {
        'half_width': half_width,
        'n': len(sub),
        'left_n': len(left),
        'right_n': len(right),
        'firms': firms,
        'distinct_cells': distinct,
        'mean_per_cell': len(sub) / distinct if distinct else float('nan'),
        'sparse': sparse,
        'min_per_side_threshold': min_per_side,
    }


def first_stage_diagnostic(records, half_width, weighting='triangular'):
    """领取率是否在阈值处跳跃；弱或反向时明确标记。"""
    sub = select_window(records, half_width)
    fit = est.first_stage_f(sub, half_width, weighting, bootstrap=False)
    return {
        'half_width': half_width,
        'jump': fit['estimate'],
        'se': fit['se'],
        'f_stat': fit['f_stat'],
        'f_p_value': fit['f_p_value'],
        'takeup_left': fit['takeup_left'],
        'takeup_right': fit['takeup_right'],
        'weak': fit['f_stat'] < est.WEAK_FS_F,
        'wrong_signed': fit['estimate'] <= 0,
    }


def placebo_diagnostics(records, shifts, half_width, weighting='triangular'):
    """伪阈值检验：平移资格规则后不应出现系统性跳跃。"""
    results = []
    for shift in shifts:
        fit = est.placebo_cutoff(records, shift, weighting, half_width)
        results.append(fit)
    flagged = [r for r in results if r.get('status') == 'ok'
               and r['p_value'] < 0.05]
    return {'shifts': results,
            'significant_placebo_count': len(flagged),
            'note': '多个伪阈值显著意味着跳跃未必来自资格规则'}


def sensitivity_grid(records, window_grid, weightings=('uniform', 'triangular'),
                     donut_options=(0, 1)):
    """预先固定的窗口×权重×donut 网格，全部估计并全部报告。

    不返回“最优带宽”，也不按显著性筛选；donut 删除的格点逐一留痕。
    """
    rows = []
    support_cells = None
    for half_width in window_grid:
        for weighting in weightings:
            for donut in donut_options:
                sub = select_window(records, half_width, donut)
                removed = donut_removed_values(
                    {'unique_distances': sorted({r['distance']
                                                 for r in records})},
                    donut)
                if len({r['eligible'] for r in sub}) < 2 or len(sub) < 4:
                    rows.append({'half_width': half_width,
                                 'weighting': weighting,
                                 'donut_cells': donut,
                                 'donut_removed': removed,
                                 'status': 'not_identifiable', 'n': len(sub)})
                    continue
                try:
                    result = est.estimate_window(sub, half_width, weighting)
                    result['status'] = 'ok'
                    result['donut_cells'] = donut
                    result['donut_removed'] = removed
                    rows.append(result)
                except su.InsufficientData as exc:
                    rows.append({'half_width': half_width,
                                 'weighting': weighting,
                                 'donut_cells': donut,
                                 'donut_removed': removed,
                                 'status': 'failed', 'reason': str(exc),
                                 'n': len(sub)})
    return rows


def diagnostic_bundle(records, window_grid,
                      weightings=('uniform', 'triangular')):
    """汇总所有诊断，供 pipeline 统一调用。"""
    max_window = max(window_grid)
    return {
        'cells': cell_counts(records),
        'bunching': [bunching_test(records, h) for h in window_grid],
        'baseline': [baseline_continuity_test(records, h)
                     for h in window_grid],
        'scarcity': [scarcity_check(records, h) for h in window_grid],
        'first_stage': [first_stage_diagnostic(records, h, w)
                        for h in window_grid for w in weightings],
        'placebo': placebo_diagnostics(
            records, shifts=(-2, 2), half_width=max_window),
        'window_grid': list(window_grid),
        'weightings': list(weightings),
        'assumption_note': '未发现密度或基线不连续，仅表示现有数据未证伪；'
                           '精确操纵不存在、排除限制与单调性均无法被证明。',
    }
