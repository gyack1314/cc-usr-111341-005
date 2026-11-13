"""政策批次的资格规则与阈值距离。

每个批次自带 cutoff、eligible_above（资格在阈值哪一侧）与 inclusive
（等号是否属于资格侧）。资格判断只允许使用公布前冻结的连续评分。
"""


def signed_distance(policy, score):
    """有符号阈值距离：资格一侧为正，非资格一侧为负，阈值处为 0。"""
    cutoff = policy['cutoff']
    if policy['eligible_above']:
        return score - cutoff
    return cutoff - score


def is_eligible(policy, score):
    """按批次自身的方向与等号规则判断资格。"""
    distance = signed_distance(policy, score)
    if policy['inclusive']:
        return distance >= 0
    return distance > 0


def policy_map(policies):
    mapping = {}
    for order, policy in enumerate(policies):
        row = dict(policy)
        row['policy_order'] = order
        mapping[policy['id']] = row
    return mapping
