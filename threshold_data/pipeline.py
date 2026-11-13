"""分析编排：先诊断后解释，自动分流并留痕全部口径。

产出（默认写入 out/<包名>/）：
- included_firms.jsonl   纳入主样本的企业、窗口资格与逐点权重；
- exclusions.json        未纳入记录及原因（观察窗、多期、donut）；
- revisions_audit.json   修订分审计，证明冻结分未被覆盖；
- diagnostics.json       堆积、基线、稀缺、一阶段、伪阈值、敏感性；
- estimates.json         三跳跃估计与逐批次异质性；
- sensitivity.csv        窗口×权重×donut 全网格（不挑显著带宽）；
- report.md / report.json 给扩围讨论的结论与边界。
"""
import csv
import json
from pathlib import Path

from .io import load
from .prep import build_sample
from .support import support_summary, select_window, default_window_grid
from . import diagnostics as diag
from . import estimation as est
from . import randomization as ri


def _write_jsonl(path, rows):
    with open(path, 'w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def _json_default(value):
    if isinstance(value, float):
        if value != value:
            return 'NaN'
        if value == float('inf'):
            return 'Infinity'
        if value == float('-inf'):
            return '-Infinity'
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f'无法序列化 {type(value)}')


def analyze(manifest_path, out_dir='out', horizon_days=90,
            default_half_width=2.0, weightings=('uniform', 'triangular')):
    tables = load(manifest_path)
    sample = build_sample(tables, horizon_days)
    primary = sample['primary']
    support = support_summary(primary)
    window_grid = default_window_grid(support)
    max_k = max(window_grid) if window_grid else 1

    bundle = diag.diagnostic_bundle(primary, window_grid, weightings)
    sensitivity = diag.sensitivity_grid(
        primary, window_grid, weightings=weightings,
        donut_options=(0, 1))

    # 离散支撑 → Fisher 随机化路径；连续支撑 → 局部回归路径。
    if support['discrete_support']:
        path_used = 'fisher_randomization'
        ri_result = ri.randomization_conclusion(primary, max_k=max_k)
        continuous_estimates = None
    else:
        path_used = 'local_linear_rdd'
        ri_result = None
        sub = select_window(primary, default_half_width)
        continuous_estimates = {
            'default_half_width': default_half_width,
            'weighting': 'triangular',
            'window': est.estimate_window(sub, default_half_width,
                                          'triangular'),
            'batch_specific': est.batch_specific_jumps(
                sub, default_half_width, 'triangular'),
        }

    # 逐点权重留痕：主窗口内每家企业的三角/均匀权重与纳入标记。
    weight_rows = []
    for weighting in weightings:
        sub = select_window(primary,
                            max_k if support['discrete_support']
                            else default_half_width)
        distances = [r['distance'] for r in sub]
        if weighting == 'triangular':
            weights = est.triangular_weights(
                distances, max_k if support['discrete_support']
                else default_half_width)
        else:
            weights = [1.0] * len(sub)
        for record, weight in zip(sub, weights):
            weight_rows.append({
                'application': record['application'],
                'firm': record['firm'],
                'batch': record['batch'],
                'distance': record['distance'],
                'eligible': record['eligible'],
                'weighting': weighting,
                'weight': weight,
                'received': record['received'],
            })

    findings = _compile_findings(sample, support, bundle, sensitivity,
                                 ri_result, continuous_estimates, path_used)

    package = Path(manifest_path).stem
    target = Path(out_dir) / package
    target.mkdir(parents=True, exist_ok=True)
    _write_jsonl(target / 'included_firms.jsonl', weight_rows)
    with open(target / 'exclusions.json', 'w', encoding='utf-8') as f:
        json.dump({'dropped': sample['dropped_ids'],
                   'auxiliary_sample': sample['auxiliary'],
                   'not_visible': sample['not_visible']},
                  f, ensure_ascii=False, indent=2, default=_json_default)
    with open(target / 'revisions_audit.json', 'w', encoding='utf-8') as f:
        json.dump(sample['revision_audit'], f, ensure_ascii=False,
                  indent=2, default=_json_default)
    with open(target / 'diagnostics.json', 'w', encoding='utf-8') as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2,
                  default=_json_default)
    with open(target / 'estimates.json', 'w', encoding='utf-8') as f:
        json.dump({'path': path_used,
                   'randomization': ri_result,
                   'continuous': continuous_estimates},
                  f, ensure_ascii=False, indent=2, default=_json_default)
    with open(target / 'sensitivity.csv', 'w', encoding='utf-8',
              newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['half_width', 'weighting', 'donut_cells',
                         'donut_removed', 'n', 'status', 'itt', 'itt_p',
                         'fs_jump', 'fs_f', 'late_ratio', 'late_status'])
        for row in sensitivity:
            writer.writerow([
                row.get('half_width'), row.get('weighting'),
                row.get('donut_cells'),
                '|'.join(str(v) for v in row.get('donut_removed', [])),
                row.get('n'), row.get('status'),
                row.get('itt', {}).get('estimate') if row.get('itt') else None,
                row.get('itt', {}).get('p_value') if row.get('itt') else None,
                row.get('first_stage', {}).get('estimate')
                if row.get('first_stage') else None,
                row.get('first_stage', {}).get('f_stat')
                if row.get('first_stage') else None,
                row.get('late_ratio'), row.get('late_status')])
    report_md = render_markdown(findings)
    (target / 'report.md').write_text(report_md, encoding='utf-8')
    with open(target / 'report.json', 'w', encoding='utf-8') as f:
        json.dump(findings, f, ensure_ascii=False, indent=2,
                  default=_json_default)
    return findings, target


def _compile_findings(sample, support, bundle, sensitivity, ri_result,
                      continuous, path_used):
    primary = sample['primary']
    scarcity = bundle['scarcity'][-1] if bundle['scarcity'] else None
    bunch = bundle['bunching'][-1] if bundle['bunching'] else None
    baseline = bundle['baseline'][-1] if bundle['baseline'] else None
    fs_diag = bundle['first_stage'][-1] if bundle['first_stage'] else None
    flips = [a for a in sample['revision_audit'].values()
             if a['post_publication_flip']]
    findings = {
        'package_path': path_used,
        'sample': {
            'primary_n': len(primary),
            'auxiliary_n': len(sample['auxiliary']),
            'not_visible_n': len(sample['not_visible']),
            'horizon_days': sample['horizon_days'],
            'firms': len({r['firm'] for r in primary}),
        },
        'support': {k: support[k] for k in
                    ('discrete_support', 'unique_count', 'tie_fraction',
                     'boundary_cell_n', 'boundary_cell_share',
                     'left_unique_n', 'right_unique_n', 'reasons',
                     'point_counts')},
        'frozen_score_audit': {
            'post_publication_revision_flips': len(flips),
            'rule': '历史资格只使用公布前冻结的连续评分，修订分仅审计不覆盖',
        },
        'diagnostics': {
            'bunching': bunch,
            'baseline_continuity': baseline,
            'scarcity': scarcity,
            'first_stage': fs_diag,
            'placebo': {k: v for k, v in bundle['placebo'].items()
                        if k != 'shifts'},
            'assumption_note': bundle['assumption_note'],
        },
        'sensitivity_grid_size': len(sensitivity),
        'sensitivity': sensitivity,
        'randomization': ri_result,
        'continuous_estimates': continuous,
        'scope_and_limits': _scope_limits(ri_result, continuous, path_used,
                                          fs_diag),
    }
    return findings


def _scope_limits(ri_result, continuous, path_used, fs_diag):
    limits = [
        '估计仅适用于阈值附近的局部合规企业（compliers），不能外推到'
        '远离阈值的企业，因此不能单独回答“扩围”的整体效果问题。',
        '诊断未报警只表示现有数据未证伪密度连续与基线连续；精确操纵不'
        '存在、排除限制与单调性本身不可由这些数据证明。',
        '阈值距离归一仅用于对齐方向，批次间政策强度异质被保留，未拼成'
        '单一处理强度曲线。',
    ]
    if path_used == 'fisher_randomization':
        limits.append('评分离散，结论基于局部随机化置换推断而非连续 RDD；'
                      '窗口按基线平衡预选择，未使用结果信息。')
        if ri_result and ri_result.get('status') == 'weak_first_stage':
            limits.append('第一阶段在随机化检验下不跳跃，停止报告 LATE，'
                          '只保留 ITT 与领取率证据。')
    if fs_diag and (fs_diag.get('weak') or fs_diag.get('wrong_signed')):
        limits.append('第一阶段弱或方向反常，任何 LATE 比值都不稳定，'
                      '不作为精确结论呈现。')
    return limits


def render_markdown(findings):
    lines = ['# 阈值附近扶持效果诊断报告', '']
    sample = findings['sample']
    lines.append(f"## 1. 样本与口径")
    lines.append(f"- 主样本（首次有效申请）：{sample['primary_n']} 家企业；"
                 f"辅助样本（其他时期）：{sample['auxiliary_n']}；"
                 f"观察窗外：{sample['not_visible_n']}")
    lines.append(f"- 固定观察窗：{sample['horizon_days']} 天")
    audit = findings['frozen_score_audit']
    lines.append(f"- 事后修订分改变资格的申请数："
                 f"{audit['post_publication_revision_flips']}（冻结分未被覆盖）")
    support = findings['support']
    lines.append('')
    lines.append('## 2. 支撑集识别')
    lines.append(f"- 推断路径：**{findings['package_path']}**")
    lines.append(f"- 离散支撑：{support['discrete_support']}；"
                 f"不同阈值距离取值数：{support['unique_count']}；"
                 f"并列占比：{support['tie_fraction']:.2%}")
    for reason in support['reasons']:
        lines.append(f"  - {reason}")
    lines.append(f"- 边界格点（距离 0）样本数：{support['boundary_cell_n']}")
    lines.append('')
    lines.append('## 3. 诊断结果（先于效果解释）')
    d = findings['diagnostics']
    if d['bunching']:
        b = d['bunching']
        lines.append(f"- 堆积检验：左 {b['left_n']} / 右 {b['right_n']}，"
                     f"Fisher p={b['fisher_p']:.3f}，"
                     f"{'报警' if b['flag'] else '未报警'}")
    if d['baseline_continuity'] and d['baseline_continuity'].get('status') == 'ok':
        bl = d['baseline_continuity']
        bl_p = '不可估计（样本无变异）' if bl['p_value'] != bl['p_value'] \
            else f"p={bl['p_value']:.3f}"
        lines.append(f"- 基线连续性：跳跃 {bl['baseline_jump']:.3f}，"
                     f"{bl_p}，"
                     f"{'报警' if bl['flag'] else '未报警'}")
    if d['scarcity']:
        s = d['scarcity']
        lines.append(f"- 样本稀缺：窗口 n={s['n']}（左 {s['left_n']} / "
                     f"右 {s['right_n']}），企业 {s['firms']} 家，"
                     f"{'样本不足' if s['sparse'] else '数量尚可但仍偏小'}")
    if d['first_stage']:
        fs = d['first_stage']
        lines.append(f"- 第一阶段（领取率跳跃）：{fs['jump']:.3f}，"
                     f"F={fs['f_stat']:.2f}，左 {fs['takeup_left']:.2%} "
                     f"→ 右 {fs['takeup_right']:.2%}，"
                     f"{'弱/反常' if (fs['weak'] or fs['wrong_signed']) else '通过'}")
    lines.append(f"- 伪阈值显著数：{d['placebo'].get('significant_placebo_count')}"
                 f"（小样本下单个伪阈值显著需结合诊断整体判断，不能据此"
                 f"确认跳跃来自资格规则）")
    lines.append(f"- {d['assumption_note']}")
    lines.append('')
    lines.append('## 4. 效果解释')
    if findings['package_path'] == 'fisher_randomization':
        ri_ = findings['randomization']
        if ri_['status'] == 'stop':
            lines.append(f"- 随机化路径停止：{ri_['reason']}")
        else:
            p = ri_['primary']
            lines.append(f"- 预选窗口 k={p['k']}（仅按基线平衡选择），"
                         f"n={p['n']}，置换方法：{p['method']}")
            lines.append(f"- ITT（资格对业务结果的跳跃）："
                         f"{p['outcome_diff']:.3f}，p={p['outcome_p']:.3f}，"
                         f"95%CI [{p['outcome_ci95'][0]:.3f}, "
                         f"{p['outcome_ci95'][1]:.3f}]")
            lines.append(f"- 第一阶段（资格对领取的跳跃）："
                         f"{p['takeup_diff']:.3f}，p={p['takeup_p']:.3f}")
            if ri_['status'] == 'weak_first_stage':
                lines.append(f"- {ri_['late_note']}")
    else:
        c = findings['continuous_estimates']['window']
        lines.append(f"- ITT：{c['itt']['estimate']:.3f}，"
                     f"聚类稳健 SE {c['itt']['se']:.3f}，"
                     f"p={c['itt']['p_value']:.3f}")
        lines.append(f"- 第一阶段：{c['first_stage']['estimate']:.3f}，"
                     f"F={c['first_stage']['f_stat']:.2f}")
        if c['late_status'] == 'ok':
            lines.append(f"- LATE（ITT/FS 比值）：{c['late_ratio']:.3f}")
        else:
            lines.append(f"- LATE 状态：{c['late_status']}（不作为精确结论）")
    lines.append('')
    lines.append('## 5. 敏感性与口径留痕')
    lines.append(f"- 窗口×权重×donut 全网格 {findings['sensitivity_grid_size']} "
                 f"组结果见 sensitivity.csv，未按显著性挑选带宽")
    lines.append('')
    lines.append('## 6. 证据边界与扩围问题')
    for limit in findings['scope_and_limits']:
        lines.append(f"- {limit}")
    return '\n'.join(lines) + '\n'
