"""离散评分路径的 Fisher 随机化推断。

当评分支撑离散、并列严重时，连续 RDD 的密度连续前提不成立，本模块改用
局部随机化框架：在窄窗口内把资格视为近似随机分配，用批次内分层置换
构造精确 p 值（组合数过大时用固定种子的 Monte Carlo 近似）。

纪律：
- 不向评分加任何噪声，不把离散数据伪装成连续数据；
- 窗口选择只允许使用政策前基线变量（协变量平衡），禁止看结果挑窗口；
- 模糊断点且样本极小时没有可靠的比值抽样理论，只报 ITT 与第一阶段的
  随机化推断，并明确停止 LATE 解释；
- 置换在批次内分层进行，保留批次结构。
"""
from itertools import combinations
from math import comb
import random
from . import statsutil as su
from .support import select_window

MAX_EXACT_COMBINATIONS = 5000
RI_SEED = 20261114
MC_REPS = 1999


def _ri_windows(records, max_k):
    """取 k=1..max_k 的对称窗口：左侧 -k..-1，资格侧 0..k-1。"""
    for k in range(1, max_k + 1):
        sub = select_window(records, half_width=k)
        eligible = [r for r in sub if r['eligible']]
        control = [r for r in sub if not r['eligible']]
        if eligible and control:
            yield k, sub


def _permutation_p(records, key, reps=MC_REPS):
    """批次内分层置换检验，返回双侧 p 值、观测统计量与置换分布。

    组大小固定时，均值差是各层“处理组和”的线性函数，用预计算系数在
    每次置换里只做 O(层数×抽取数) 运算，避免重建整列标志与全样本均值。
    """
    values = [float(r[key]) for r in records]
    observed_flag = [1 if r['eligible'] else 0 for r in records]
    n1 = sum(observed_flag)
    n = len(records)
    strata = {}
    for i, r in enumerate(records):
        strata.setdefault(r['batch'], []).append(i)

    stratum_info = []
    total_treated = sum(1 for f in observed_flag if f)
    total_control = n - total_treated
    observed_treated_sum = 0.0
    observed_total_sum = 0.0
    for indices in strata.values():
        k = sum(observed_flag[i] for i in indices)
        stratum_sum = sum(values[i] for i in indices)
        treated_sum = sum(values[i] for i in indices if observed_flag[i])
        observed_treated_sum += treated_sum
        observed_total_sum += stratum_sum
        # 全局均值差：S_t/n1 - (S_all-S_t)/n0
        # 对层内处理组和 s_g 的系数为 (1/n1 + 1/n0)，
        # 层总和项 -S_g/n0 与置换无关，提到循环外只算一次。
        stratum_info.append((indices, k, stratum_sum))
    observed_stat = (observed_treated_sum / total_treated
                     - (observed_total_sum - observed_treated_sum)
                     / total_control)
    constant = -sum(s / total_control for _, _, s in stratum_info)

    def stat_from(picks_by_stratum):
        treated_total = 0.0
        for (indices, k, stratum_sum), picks in zip(stratum_info,
                                                     picks_by_stratum):
            treated_total += sum(values[i] for i in picks)
        return (constant
                + treated_total * (1.0 / total_treated
                                   + 1.0 / total_control))

    total_comb = 1
    for indices, k, _ in stratum_info:
        total_comb *= comb(len(indices), k)

    perm_stats = []
    if total_comb <= MAX_EXACT_COMBINATIONS:
        options = [list(combinations(indices, k))
                   for indices, k, _ in stratum_info]
        def recurse(depth, picks):
            if depth == len(options):
                perm_stats.append(stat_from(picks))
                return
            for choice in options[depth]:
                recurse(depth + 1, picks + [choice])
        recurse(0, [])
        hits = sum(1 for t in perm_stats
                   if abs(t) >= abs(observed_stat) - 1e-15)
        p_value = hits / len(perm_stats)
        method = 'exact'
    else:
        rng = random.Random(RI_SEED)
        for _ in range(reps):
            picks = [tuple(rng.sample(indices, k))
                     for indices, k, _ in stratum_info]
            perm_stats.append(stat_from(picks))
        hits = sum(1 for t in perm_stats
                   if abs(t) >= abs(observed_stat) - 1e-15)
        p_value = (1 + hits) / (1 + len(perm_stats))
        method = 'monte_carlo'
    return {'p_value': p_value, 'observed_stat': observed_stat,
            'perm_stats': perm_stats, 'method': method,
            'n': n, 'n_treated': n1}


def invert_ci(perm_stats, observed_stat, alpha=0.05):
    """加性常数效应下检验反演：|T_obs - tau| 的临界值来自置换分布。"""
    critical = su.quantile([abs(t) for t in perm_stats], 1 - alpha)
    return observed_stat - critical, observed_stat + critical


def ri_window_report(records, max_k, alpha=0.05):
    """全部预登记窗口的随机化推断，窗口选择只看基线平衡。"""
    rows = []
    for k, sub in _ri_windows(records, max_k):
        outcome = _permutation_p(sub, 'outcome')
        baseline = _permutation_p(sub, 'baseline')
        receipt = _permutation_p(sub, 'received')
        outcome['ci95'] = invert_ci(outcome['perm_stats'],
                                    outcome['observed_stat'], alpha)
        rows.append({
            'k': k,
            'n': outcome['n'],
            'n_treated': outcome['n_treated'],
            'method': outcome['method'],
            'outcome_diff': outcome['observed_stat'],
            'outcome_p': outcome['p_value'],
            'outcome_ci95': outcome['ci95'],
            'baseline_diff': baseline['observed_stat'],
            'baseline_p': baseline['p_value'],
            'baseline_balanced': baseline['p_value'] >= alpha,
            'takeup_diff': receipt['observed_stat'],
            'takeup_p': receipt['p_value'],
            'takeup_ci95': invert_ci(receipt['perm_stats'],
                                     receipt['observed_stat'], alpha),
            'first_stage_detected': receipt['p_value'] < alpha
                                    and receipt['observed_stat'] > 0,
        })
    return rows


def select_ri_window(rows, alpha=0.05):
    """仅依据基线平衡选窗口：取基线平衡的最窄窗口。

    这是协变量驱动选择，全程不看 outcome 与 received 的显著性。窄窗口
    局部随机化前提最强；若没有任何窗口通过平衡检验，返回 None 由调用方
    决定停止还是降级呈现。
    """
    for row in rows:
        if row['baseline_balanced']:
            return row
    return None


def randomization_conclusion(records, max_k, alpha=0.05):
    rows = ri_window_report(records, max_k, alpha)
    if not rows:
        return {'status': 'stop',
                'reason': '没有同时包含资格侧与非资格侧的局部窗口',
                'windows': []}
    chosen = select_ri_window(rows, alpha)
    if chosen is None:
        return {'status': 'stop',
                'reason': '所有候选窗口基线都不平衡，局部随机化前提存疑',
                'windows': rows}
    # 窄窗口通过但更宽窗口失衡时仍记录，作为敏感性信息。
    balance_breakdown = [row['k'] for row in rows
                         if not row['baseline_balanced']]
    conclusion = {
        'status': 'ok' if chosen['first_stage_detected'] else 'weak_first_stage',
        'primary_window_k': chosen['k'],
        'window_selector': 'narrowest nested window with baseline balance '
                           '(covariate-only, outcome never inspected)',
        'windows_failing_balance': balance_breakdown,
        'primary': chosen,
        'windows': rows,
    }
    if not chosen['first_stage_detected']:
        conclusion['late'] = None
        conclusion['late_note'] = ('第一阶段在随机化检验下不显著，离散小样本'
                                   '没有可靠的比值推断，停止报告 LATE')
    return conclusion
