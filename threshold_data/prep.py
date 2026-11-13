"""分析样本构建：冻结分口径、修订审计、首次有效申请、观察窗。

口径纪律（与 docs/data-contract.md 一致）：
- applications.score 是公布前冻结的连续评分，历史资格只由它计算；
- score_revisions 是后补资料，只用于审计，任何情况下都不覆盖冻结分；
- 主分析只保留每家企业的首次有效申请，其余时期归入辅助样本；
- outcome_known_day 超过固定观察窗的申请视为窗口内不可见，不进主样本。
"""
from .policy import is_eligible, signed_distance

DEFAULT_HORIZON_DAYS = 90


def _revision_audit(applications, revisions, policies):
    by_application = {}
    for revision in revisions:
        by_application.setdefault(revision['application'], []).append(revision)
    audit = {}
    for app in applications:
        policy = policies[app['batch']]
        entries = sorted(by_application.get(app['id'], []),
                         key=lambda r: r['known_day'])
        frozen_eligible = is_eligible(policy, app['score'])
        post_publication = [r for r in entries if r['known_day'] > 0]
        flips = []
        for revision in entries:
            revised_eligible = is_eligible(policy, revision['new_score'])
            flips.append({
                'revision': revision['id'],
                'known_day': revision['known_day'],
                'new_score': revision['new_score'],
                'post_publication': revision['known_day'] > 0,
                'would_change_eligibility':
                    revised_eligible != frozen_eligible,
            })
        audit[app['id']] = {
            'revision_count': len(entries),
            'post_publication_count': len(post_publication),
            'post_publication_flip': any(
                f['post_publication'] and f['would_change_eligibility']
                for f in flips),
            'details': flips,
        }
    return audit


def build_sample(tables, horizon_days=DEFAULT_HORIZON_DAYS):
    """把原始三表整理成主样本、辅助样本与修订审计。

    返回 dict：
      primary      每家企业首次有效申请的分析记录
      auxiliary    同企业其他时期的申请记录
      not_visible  超出固定观察窗的申请
      revision_audit 逐申请的修订审计
      dropped_ids  未进主样本的申请 id 及原因
    """
    policies = {}
    for order, row in enumerate(tables['policies']):
        entry = dict(row)
        entry['policy_order'] = order
        policies[row['id']] = entry
    audit = _revision_audit(tables['applications'],
                            tables['score_revisions'], policies)

    visible, not_visible = [], []
    for app in tables['applications']:
        policy = policies[app['batch']]
        record = {
            'application': app['id'],
            'firm': app['firm'],
            'batch': app['batch'],
            'policy_order': policy['policy_order'],
            'frozen_score': app['score'],
            'distance': signed_distance(policy, app['score']),
            'eligible': is_eligible(policy, app['score']),
            'received': bool(app['received']),
            'baseline': app['baseline'],
            'outcome': app['outcome'],
            'outcome_known_day': app['outcome_known_day'],
            'revision_count': audit[app['id']]['revision_count'],
            'post_publication_flip': audit[app['id']]['post_publication_flip'],
        }
        if app['outcome_known_day'] <= horizon_days:
            visible.append(record)
        else:
            not_visible.append(record)

    # 文件顺序不代表事件顺序；按批次登记顺序、申请 id 确定确定性次序。
    visible.sort(key=lambda r: (r['firm'], r['policy_order'], r['application']))
    primary, auxiliary, seen_firms = [], [], set()
    for record in visible:
        if record['firm'] in seen_firms:
            auxiliary.append(record)
        else:
            seen_firms.add(record['firm'])
            primary.append(record)

    dropped = [{'application': r['application'],
                'reason': f'outcome_known_day {r["outcome_known_day"]} > '
                          f'horizon {horizon_days}'}
               for r in not_visible]
    dropped.extend({'application': r['application'],
                    'reason': 'later-period application of same firm '
                              '(auxiliary sample)'}
                   for r in auxiliary)
    return {
        'primary': primary,
        'auxiliary': auxiliary,
        'not_visible': not_visible,
        'revision_audit': audit,
        'dropped_ids': dropped,
        'horizon_days': horizon_days,
        'policies': policies,
    }
