"""局部回归、三跳跃与弱一阶段闸门测试。"""
import unittest
from threshold_data import estimation as est
from threshold_data import statsutil as su
from threshold_data.simulation import dgp_fuzzy, dgp_no_first_stage


def _records_from_rows(rows):
    return [{'application': f'a{i}', 'firm': f'firm{i}', 'batch': 'b1',
             'policy_order': 0, 'distance': float(d),
             'eligible': d >= 0, 'received': bool(r),
             'baseline': float(b), 'outcome': float(o),
             'outcome_known_day': 90, 'frozen_score': float(d),
             'revision_count': 0, 'post_publication_flip': False}
            for i, (d, r, b, o) in enumerate(rows)]


class EstimationTests(unittest.TestCase):
    def test_sharp_jump_recovered(self):
        # 构造明确的结果水平跳跃：右侧结果统一高 5
        rows = [(-2, 0, 10, 10), (-1, 0, 10, 10),
                (0, 1, 10, 15), (1, 1, 10, 15), (2, 1, 10, 15)]
        records = _records_from_rows(rows)
        result = est.estimate_window(records, 2.0, 'uniform',
                                     bootstrap=False)
        self.assertAlmostEqual(result['itt']['estimate'], 5.0, places=8)
        self.assertGreater(result['first_stage']['f_stat'], 0)

    def test_weak_first_stage_gates_late(self):
        records, _ = dgp_no_first_stage(n=200, seed=42)
        result = est.estimate_window(records, 0.75, 'triangular',
                                     bootstrap=False)
        self.assertTrue(result['weak_first_stage']
                        or result['wrong_signed_first_stage'])
        self.assertNotEqual(result['late_status'], 'ok')

    def test_wrong_signed_first_stage_flagged(self):
        # 资格侧领取率反而更低
        rows = [(-1, 1, 10, 12), (0, 0, 10, 12), (1, 0, 10, 12)]
        records = _records_from_rows(rows)
        result = est.estimate_window(records, 1.0, 'uniform',
                                     bootstrap=False)
        self.assertTrue(result['wrong_signed_first_stage'])

    def test_degenerate_outcome_does_not_crash(self):
        rows = [(-1, 0, 10, 0), (0, 0, 10, 0), (1, 0, 10, 0)]
        records = _records_from_rows(rows)
        fit = est.first_stage_f(records, 1.0, 'uniform', bootstrap=False)
        self.assertEqual(fit['estimate'], 0.0)
        self.assertTrue(math_isnan(fit['se']))

    def test_batch_structure_preserved(self):
        records, _ = dgp_fuzzy(n=120, seed=7)
        for i, r in enumerate(records):
            r['batch'] = 'b1' if i % 2 else 'b2'
            r['policy_order'] = 0 if r['batch'] == 'b1' else 1
        x, batches, names = est.build_design(records)
        self.assertEqual(set(batches), {'b1', 'b2'})
        self.assertIn('fe_b2', names)
        jumps = est.batch_specific_jumps(records, 0.75, 'triangular')
        self.assertEqual(set(jumps), {'b1', 'b2'})

    def test_placebo_cutoff_runs(self):
        records, _ = dgp_fuzzy(n=200, seed=11)
        placebo = est.placebo_cutoff(records, 0.4, 'triangular', 0.4)
        self.assertIn(placebo['status'], ('ok', 'not_identifiable'))


def math_isnan(value):
    return value != value


if __name__ == '__main__':
    unittest.main()
