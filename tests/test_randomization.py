"""Fisher 随机化推断与离散路径闸门测试。"""
import unittest
from threshold_data import randomization as ri
from threshold_data.simulation import dgp_heterogeneous


def _rows():
    records, _ = dgp_heterogeneous(seed=4000)
    return records


class RandomizationTests(unittest.TestCase):
    def test_exact_permutation_p_values(self):
        # 小到可完全枚举的构造
        records = []
        for i, (d, y) in enumerate([(-1, 10), (-1, 12),
                                    (0, 20), (0, 22)]):
            records.append({'application': f'a{i}', 'firm': f'f{i}',
                            'batch': 'b1', 'distance': float(d),
                            'eligible': d >= 0, 'received': d >= 0,
                            'baseline': float(10 + i % 2),
                            'outcome': float(y),
                            'outcome_known_day': 90})
        result = ri._permutation_p(records, 'outcome')
        self.assertEqual(result['method'], 'exact')
        # 4 选 2 共 6 种排列，观测差值为最大之一
        self.assertLessEqual(result['p_value'], 2 / 6 + 1e-9)

    def test_stratified_preserves_batch_counts(self):
        records = _rows()
        result = ri._permutation_p(records, 'baseline')
        # 基线无效应：p 值不应系统性偏小
        self.assertGreater(result['p_value'], 0.05)

    def test_first_stage_detected_in_fuzzy_dgp(self):
        records = _rows()
        conclusion = ri.randomization_conclusion(records, max_k=3)
        self.assertIn(conclusion['status'], ('ok', 'weak_first_stage'))
        primary = conclusion['primary']
        self.assertLess(primary['takeup_p'], 0.05)
        self.assertGreater(primary['takeup_diff'], 0)

    def test_window_selector_uses_baseline_only(self):
        records = _rows()
        rows = ri.ri_window_report(records, max_k=3)
        chosen = ri.select_ri_window(rows)
        self.assertIsNotNone(chosen)
        self.assertTrue(chosen['baseline_balanced'])

    def test_stop_when_no_balanced_window(self):
        # 人为制造基线在所有窗口都强烈失衡
        records = _rows()
        for r in records:
            if r['eligible']:
                r['baseline'] += 50.0
        conclusion = ri.randomization_conclusion(records, max_k=3)
        self.assertEqual(conclusion['status'], 'stop')
        self.assertIn('基线', conclusion['reason'])

    def test_ci_inversion_finite(self):
        records = _rows()
        rows = ri.ri_window_report(records, max_k=2)
        for row in rows:
            lo, hi = row['outcome_ci95']
            self.assertTrue(lo < hi)


if __name__ == '__main__':
    unittest.main()
