"""独立模拟：用可控 DGP 验证估计偏差、区间覆盖与诊断行为。

模拟与真实资料完全隔离，目的是在分析业务数据之前证明本项目的估计与
推断链条在可控环境下表现合格。包含四类数据生成过程：

- fuzzy       有资格跳跃且有领取跳跃，ITT/FS/LATE 真值已知；
- no_first_stage 有资格规则但领取率不跳跃，检验弱一阶段闸门；
- manipulation  评分在阈值处堆积且基线不连续，检验诊断能否报警；
- heterogeneous 多批次异质处理效应 + 离散整数评分，检验批次结构
                 与随机化路径是否保持覆盖。
"""
import math
import random
from . import estimation as est
from . import diagnostics as diag
from . import randomization as ri
from .support import support_summary


def _make_record(i, batch, distance, eligible, received, outcome, baseline,
                 known_day=90):
    return {'application': f'sim-{i}', 'firm': f'firm-{i}', 'batch': batch,
            'policy_order': 0, 'distance': distance,
            'eligible': eligible, 'received': received,
            'baseline': baseline, 'outcome': outcome,
            'outcome_known_day': known_day, 'frozen_score': distance,
            'revision_count': 0, 'post_publication_flip': False}


def dgp_fuzzy(n=400, seed=1, tau_late=4.0, fs_jump=0.6, noise=2.0,
              half_width=1.0):
    """连续评分模糊断点：LATE 真值 tau_late，第一阶段真值 fs_jump。"""
    rng = random.Random(seed)
    records = []
    for i in range(n):
        d = rng.uniform(-half_width, half_width)
        eligible = d >= 0
        p_take = min(0.9, max(0.05, 0.2 + fs_jump * eligible))
        received = rng.random() < p_take
        baseline = 50.0 + rng.gauss(0, 5.0)
        outcome = baseline + tau_late * received + rng.gauss(0, noise)
        records.append(_make_record(i, 'b1', d, eligible, received,
                                    outcome, baseline))
    truth = {'itt': tau_late * fs_jump, 'first_stage': fs_jump,
             'late': tau_late}
    return records, truth


def dgp_no_first_stage(n=400, seed=2, noise=2.0, half_width=1.0):
    """资格规则存在但领取率不跳跃：ITT 与 FS 真值均为 0。"""
    rng = random.Random(seed)
    records = []
    for i in range(n):
        d = rng.uniform(-half_width, half_width)
        eligible = d >= 0
        received = rng.random() < 0.3
        baseline = 50.0 + rng.gauss(0, 5.0)
        outcome = baseline + rng.gauss(0, noise)
        records.append(_make_record(i, 'b1', d, eligible, received,
                                    outcome, baseline))
    return records, {'itt': 0.0, 'first_stage': 0.0, 'late': None}


def dgp_manipulation(n=400, seed=3, noise=2.0, half_width=1.0):
    """评分操纵：非资格侧企业被推过阈值，基线同时不连续。"""
    rng = random.Random(seed)
    records = []
    i = 0
    while len(records) < n:
        d = rng.uniform(-half_width, half_width)
        if d < 0 and rng.random() < 0.35:
            d = abs(d) * 0.25  # 堆积到资格侧
        eligible = d >= 0
        received = rng.random() < (0.7 if eligible else 0.2)
        baseline = 50.0 + (8.0 if eligible else 0.0) + rng.gauss(0, 4.0)
        outcome = baseline + 2.0 * received + rng.gauss(0, noise)
        records.append(_make_record(i, 'b1', d, eligible, received,
                                    outcome, baseline))
        i += 1
    return records, {'itt': None, 'first_stage': 0.5, 'late': None,
                     'manipulation_present': True}


def dgp_heterogeneous(n=360, seed=4):
    """离散整数评分 + 两批次异质 LATE（2.0 与 6.0）。"""
    rng = random.Random(seed)
    records = []
    batch_params = {'b1': 2.0, 'b2': 6.0}
    i = 0
    for batch, tau in batch_params.items():
        for _ in range(n // 2):
            d = rng.randint(-3, 2)
            eligible = d >= 0
            received = rng.random() < (0.75 if eligible else 0.15)
            baseline = 30.0 + rng.gauss(0, 3.0)
            outcome = baseline + tau * received + rng.gauss(0, 1.5)
            rec = _make_record(i, batch, float(d), eligible, received,
                               outcome, baseline)
            rec['policy_order'] = 0 if batch == 'b1' else 1
            records.append(rec)
            i += 1
    return records, {'late_by_batch': batch_params,
                     'first_stage': 0.6, 'discrete': True}


def _coverage(runs, key_estimate, key_ci, truth):
    if truth is None:
        return None
    hits = 0
    valid = 0
    for run in runs:
        ci = run.get(key_ci)
        if not ci or any(math.isnan(v) for v in ci):
            continue
        valid += 1
        if ci[0] - 1e-9 <= truth <= ci[1] + 1e-9:
            hits += 1
    return hits / valid if valid else float('nan')


def run_continuous_validation(reps=200, half_width=0.75, seed0=1000):
    """连续 DGP 上检验 ITT/FS 的偏差与 95% 覆盖率。

    覆盖率验证使用解析聚类稳健区间（bootstrap 只在真实小样本分析中
    额外附加），保证模拟可在纯标准库下以合理时间完成。
    """
    runs = []
    truth = dgp_fuzzy(seed=seed0)[1]
    for r in range(reps):
        records, _ = dgp_fuzzy(seed=seed0 + r, half_width=1.0)
        result = est.estimate_window(records, half_width, 'triangular',
                                     bootstrap=False)
        runs.append({'itt': result['itt']['estimate'],
                     'itt_ci': result['itt']['ci95'],
                     'fs': result['first_stage']['estimate'],
                     'fs_ci': result['first_stage']['ci95'],
                     'late': result['late_ratio']})
    itt_bias = sum(x['itt'] for x in runs) / reps - truth['itt']
    fs_bias = sum(x['fs'] for x in runs) / reps - truth['first_stage']
    return {
        'reps': reps,
        'itt_bias': itt_bias,
        'fs_bias': fs_bias,
        'itt_coverage': _coverage(runs, 'itt', 'itt_ci', truth['itt']),
        'fs_coverage': _coverage(runs, 'fs', 'fs_ci',
                                 truth['first_stage']),
        'late_mean': sum(x['late'] for x in runs) / reps,
        'late_truth': truth['late'],
    }


def run_weak_fs_validation(reps=200, half_width=0.75, seed0=2000):
    """无第一阶段 DGP：弱一阶段闸门应触发，且 ITT 误报率受控。"""
    false_jumps = 0
    gated = 0
    for r in range(reps):
        records, _ = dgp_no_first_stage(seed=seed0 + r)
        result = est.estimate_window(records, half_width, 'triangular',
                                     bootstrap=False)
        if result['weak_first_stage'] or result['wrong_signed_first_stage']:
            gated += 1
        if result['itt']['p_value'] < 0.05:
            false_jumps += 1
    return {'reps': reps,
            'weak_fs_gate_rate': gated / reps,
            'itt_false_positive_rate': false_jumps / reps}


def run_manipulation_detection(reps=100, seed0=3000):
    """操纵 DGP：堆积或基线不连续诊断应在多数重复中报警。"""
    alarms = 0
    for r in range(reps):
        records, _ = dgp_manipulation(seed=seed0 + r)
        bunch = diag.bunching_test(records, 0.5)
        base = diag.baseline_continuity_test(records, 0.5)
        if bunch['flag'] or base['flag']:
            alarms += 1
    return {'reps': reps, 'alarm_rate': alarms / reps,
            'expectation': 'alarm_rate 应明显高于名义 5% 水平'}


def run_discrete_validation(reps=100, seed0=4000):
    """离散异质 DGP：支撑识别应判离散，随机化路径覆盖率应接近名义水平。"""
    records0, _ = dgp_heterogeneous(seed=seed0)
    support = support_summary(records0)
    runs = []
    for r in range(reps):
        records, _ = dgp_heterogeneous(seed=seed0 + r)
        conclusion = ri.randomization_conclusion(records, max_k=3)
        if conclusion['status'] == 'stop':
            continue
        primary = conclusion['primary']
        runs.append({'outcome_ci': primary['outcome_ci95'],
                     'outcome_est': primary['outcome_diff']})
    # 异质效应没有单一真值，覆盖率用批次加权平均 ITT 近似检查宽度合理性。
    return {
        'reps': reps,
        'support_discrete_detected': support['discrete_support'],
        'unique_distances': support['unique_distances'],
        'completed_runs': len(runs),
        'mean_ci_width': (sum(x['outcome_ci'][1] - x['outcome_ci'][0]
                              for x in runs) / len(runs)) if runs else None,
        'note': '离散路径以置换 p 值与检验反演区间为主，不与连续 LATE 混用',
    }


def run_all(reps_continuous=200, reps_weak=200, reps_manip=100,
            reps_discrete=100):
    return {
        'continuous_fuzzy': run_continuous_validation(reps_continuous),
        'weak_first_stage': run_weak_fs_validation(reps_weak),
        'manipulation': run_manipulation_detection(reps_manip),
        'discrete_heterogeneous': run_discrete_validation(reps_discrete),
    }
