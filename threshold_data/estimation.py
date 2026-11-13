"""阈值附近的局部回归估计：ITT、第一阶段与 LATE 比值。

设计纪律：
- 多批次保留批次结构：批次固定效应 + 批次特定斜率，绝不把异质批次
  简单拼成一条曲线；
- 窗口与权重方案在分析前固定，全部报告，不允许看完效果再挑最显著带宽；
- 企业多期出现时标准误按 firm 聚类，并使用 wild cluster bootstrap
  处理小样本，禁止直接沿用普通标准误；
- 第一阶段弱或方向反常时不报“稳定精确”的 LATE，只给警示性比值。
"""
import math
from . import statsutil as su

WEAK_FS_F = 10.0
MIN_CLUSTERS_BOOTSTRAP = 6
BOOTSTRAP_REPS = 999


def triangular_weights(distances, half_width):
    # 边界点标准三角核权重为 0，会让小窗口损失整侧观测；加一个不改变
    # 核形状排序的小常数，保证每个入选观测都有正权重。
    epsilon = 1e-6
    return [max(epsilon, 1.0 - abs(d) / half_width) for d in distances]


def build_design(records):
    """构造满秩设计矩阵：截距、E、距离、E×距离、批次 FE、批次斜率。

    极小窗口下距离格点不足，斜率列与截距/E 共线，此时按支撑自适应丢弃
    共线列，退化为水平跳跃模型。返回的列名里 E 的下标固定为 1。
    """
    batches = sorted({r['batch'] for r in records})
    if not batches:
        raise su.InsufficientData('窗口为空')
    rows, names = [], ['intercept', 'E']
    # 降秩依据有效观测数：参数列数不能超过样本数，否则退化为水平跳跃。
    distinct_d = {r['distance'] for r in records}
    n_batches = len(batches)
    slope_columns = 2 + max(0, n_batches - 1)
    level_columns = 2 + max(0, n_batches - 1)
    include_slopes = (len(distinct_d) >= 3
                      and len(records) >= level_columns + slope_columns)
    if include_slopes:
        names += ['d', 'E_d']
    for b in batches[1:]:
        names.append(f'fe_{b}')
        if include_slopes:
            names.append(f'slope_{b}')
    for r in records:
        d = r['distance']
        e = 1.0 if r['eligible'] else 0.0
        row = [1.0, e]
        if include_slopes:
            row += [d, e * d]
        for b in batches[1:]:
            row.append(1.0 if r['batch'] == b else 0.0)
            if include_slopes:
                row.append(d if r['batch'] == b else 0.0)
        rows.append(row)
    return rows, batches, names


def _fit_outcome(records, half_width, weighting, outcome_key,
                 bootstrap=True):
    distances = [r['distance'] for r in records]
    if weighting == 'triangular':
        weights = triangular_weights(distances, half_width)
    elif weighting == 'uniform':
        weights = [1.0] * len(records)
    else:
        raise ValueError(f'未知权重方案 {weighting}')
    x, batches, names = build_design(records)
    e_index = names.index('E')
    assert e_index == 1
    y = [float(r[outcome_key]) for r in records]
    clusters = [r['firm'] for r in records]
    if len(set(y)) == 1:
        # 结果无变异（如该窗口内全部未领取）：跳跃恒为 0，方差不可估计，
        # 返回显式标记而不是抛出奇异矩阵错误。
        g = len(set(clusters))
        return ({'estimate': 0.0, 'se': float('nan'), 't': float('nan'),
                 'p_value': float('nan'),
                 'ci95': (float('nan'), float('nan')),
                 'n': len(records), 'clusters': g,
                 'half_width': half_width, 'weighting': weighting,
                 'batches': batches, 'df': max(g - 1, 1),
                 'degenerate': True,
                 'bootstrap_p': float('nan'),
                 'bootstrap_ci95': (float('nan'), float('nan')),
                 'bootstrap_reps': 0},
                (x, y, clusters, weights, None, None))
    beta, resid, _ = su.ols(x, y, weights)
    vcov, g = su.cluster_robust_vcov(x, resid, clusters, weights)
    se = math.sqrt(vcov[1][1])
    t_stat = beta[1] / se if se > 0 else float('nan')
    df = max(g - 1, 1)
    result = {
        'estimate': beta[1],
        'se': se,
        't': t_stat,
        'p_value': su.t_two_sided_p(t_stat, df),
        'ci95': (beta[1] - su.t_ppf(0.05, df) * se,
                 beta[1] + su.t_ppf(0.05, df) * se),
        'n': len(records),
        'clusters': g,
        'half_width': half_width,
        'weighting': weighting,
        'batches': batches,
        'df': df,
    }
    if bootstrap and g >= MIN_CLUSTERS_BOOTSTRAP:
        t_boot, _, _ = su.wild_cluster_bootstrap(
            x, y, clusters, weights, beta, vcov, 1,
            reps=BOOTSTRAP_REPS)
        result['bootstrap_p'] = su.bootstrap_pvalue(t_boot, t_stat)
        result['bootstrap_ci95'] = su.bootstrap_t_interval(
            t_boot, beta[1], se)
        result['bootstrap_reps'] = len(t_boot)
    else:
        result['bootstrap_p'] = float('nan')
        result['bootstrap_ci95'] = (float('nan'), float('nan'))
        result['bootstrap_reps'] = 0
    return result, (x, y, clusters, weights, beta, vcov)


def first_stage_f(records, half_width, weighting, bootstrap=True):
    """第一阶段：received 对资格的回归，返回跳跃与稳健 F。"""
    fit, (x, y, clusters, weights, beta, vcov) = _fit_outcome(
        records, half_width, weighting, 'received',
        bootstrap=bootstrap)
    if fit.get('degenerate'):
        fit['f_stat'] = 0.0
        fit['f_p_value'] = float('nan')
    else:
        f_stat, f_p = su.wald_pvalue(beta, vcov, [1])
        if f_stat != f_stat:  # NaN：received 与资格完全共线等情形
            # 完美依从（两侧领取率为 0/1）时第一阶段本质上是确定性的，
            # 用大有限哨兵标记强一阶段，F 不可估计在 f_note 显式注明，
            # 避免产生非有限 JSON。
            f_stat = 1e12 if fit['estimate'] != 0 else 0.0
            fit['f_note'] = '方差矩阵奇异，F 不可估计；按跳跃非零判强一阶段'
        fit['f_stat'] = f_stat
        fit['f_p_value'] = f_p
    fit['takeup_left'] = _side_mean(records, eligible=False)
    fit['takeup_right'] = _side_mean(records, eligible=True)
    return fit


def _side_mean(records, eligible):
    side = [r for r in records if r['eligible'] == eligible]
    if not side:
        return float('nan')
    return sum(1.0 for r in side if r['received']) / len(side)


def estimate_window(records, half_width, weighting='triangular',
                    bootstrap=True):
    """一个固定窗口上的完整三跳跃分析。"""
    if len({r['batch'] for r in records}) == 0:
        raise su.InsufficientData('窗口为空')
    itt, cache = _fit_outcome(records, half_width, weighting, 'outcome',
                              bootstrap=bootstrap)
    fs = first_stage_f(records, half_width, weighting,
                       bootstrap=bootstrap)
    weak = fs['f_stat'] < WEAK_FS_F
    wrong_sign = fs['estimate'] <= 0
    late = float('nan')
    late_status = 'not_reported'
    if fs['estimate'] != 0 and not wrong_sign:
        late = itt['estimate'] / fs['estimate']
        late_status = 'weak_first_stage' if weak else 'ok'
    elif wrong_sign:
        late_status = 'wrong_signed_first_stage'
    return {
        'itt': itt,
        'first_stage': fs,
        'late_ratio': late,
        'late_status': late_status,
        'weak_first_stage': weak,
        'wrong_signed_first_stage': wrong_sign,
        'half_width': half_width,
        'weighting': weighting,
        'n': len(records),
    }


def batch_specific_jumps(records, half_width, weighting='triangular'):
    """逐批次估计资格跳跃，用于检验批次异质性，不做跨批强度拼曲线。"""
    output = {}
    for batch in sorted({r['batch'] for r in records}):
        sub = [r for r in records if r['batch'] == batch]
        if len(sub) < 4 or len({r['eligible'] for r in sub}) < 2:
            output[batch] = {'status': 'insufficient', 'n': len(sub)}
            continue
        try:
            output[batch] = estimate_window(sub, half_width, weighting)
        except su.InsufficientData as exc:
            output[batch] = {'status': 'failed', 'reason': str(exc),
                             'n': len(sub)}
    return output


def placebo_cutoff(records, fake_distance, weighting='triangular',
                   half_width=None):
    """伪阈值：把阈值平移到不含真实阈值的位置重新估计 ITT。

    fake_distance 是相对真实阈值的平移量；资格标记按平移后的虚拟规则
    重新生成。伪阈值处不应出现显著跳跃。
    """
    shifted = []
    for r in records:
        row = dict(r)
        row['distance'] = r['distance'] - fake_distance
        row['eligible'] = row['distance'] >= 0
        shifted.append(row)
    if half_width is None:
        half_width = min(abs(v) for v in
                         {r['distance'] for r in shifted} if v != 0) or 1
    sub = [r for r in shifted if abs(r['distance']) <= half_width]
    if len({r['eligible'] for r in sub}) < 2:
        return {'status': 'not_identifiable', 'shift': fake_distance}
    fit, _ = _fit_outcome(sub, half_width, weighting, 'outcome')
    fit['shift'] = fake_distance
    fit['status'] = 'ok'
    return fit
