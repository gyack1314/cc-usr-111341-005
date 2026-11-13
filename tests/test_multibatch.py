"""多批次异质方向与批次结构保留测试。"""
import unittest
from threshold_data.policy import is_eligible, signed_distance
from threshold_data import estimation as est
from threshold_data.support import support_summary


def _record(i, batch, distance, received, outcome):
    return {'application': f'{batch}-{i}', 'firm': f'{batch}-f{i}',
            'batch': batch, 'policy_order': 0,
            'distance': float(distance),
            'eligible': distance >= 0, 'received': received,
            'baseline': 20.0, 'outcome': float(outcome),
            'outcome_known_day': 90, 'frozen_score': float(distance),
            'revision_count': 0, 'post_publication_flip': False}


class MultiBatchTests(unittest.TestCase):
    def test_opposite_directions_orient_distance(self):
        # 两个批次资格方向相反，但有符号距离在资格侧都为非负
        high = {'id': 'h', 'cutoff': 70, 'eligible_above': True,
                'inclusive': True}
        low = {'id': 'l', 'cutoff': 50, 'eligible_above': False,
               'inclusive': True}
        self.assertTrue(is_eligible(high, 72))
        self.assertFalse(is_eligible(high, 68))
        self.assertTrue(is_eligible(low, 48))
        self.assertFalse(is_eligible(low, 52))
        self.assertEqual(signed_distance(high, 72), 2)
        self.assertEqual(signed_distance(low, 48), 2)

    def test_batch_fixed_effects_in_design(self):
        records = []
        for batch, shift in (('b1', 0.0), ('b2', 5.0)):
            for i, d in enumerate((-2, -1, 0, 1, 2)):
                received = d >= 0
                records.append(_record(i, batch, d, received,
                                       10 + shift + (4 if received else 0)))
        x, batches, names = est.build_design(records)
        self.assertEqual(batches, ['b1', 'b2'])
        self.assertIn('fe_b2', names)
        self.assertIn('slope_b2', names)
        result = est.estimate_window(records, 2.0, 'uniform',
                                     bootstrap=False)
        # 两批次处理效应均为 4，批次 FE 吸收水平差异后 ITT 仍接近 4
        self.assertAlmostEqual(result['itt']['estimate'], 4.0, delta=0.5)
        per_batch = est.batch_specific_jumps(records, 2.0, 'uniform')
        self.assertEqual(set(per_batch), {'b1', 'b2'})
        for batch in ('b1', 'b2'):
            self.assertEqual(per_batch[batch].get('status', 'ok'), 'ok')
            self.assertAlmostEqual(
                per_batch[batch]['itt']['estimate'], 4.0, delta=0.5)

    def test_heterogeneous_batches_not_pooled_into_one_curve(self):
        # 批次效应方向相反时，逐批次结果必须能暴露异质性，不能只给混合数
        records = []
        for batch, sign in (('b1', 1), ('b2', -1)):
            for i, d in enumerate((-2, -1, 0, 1, 2)):
                received = d >= 0
                records.append(_record(
                    i, batch, d, received,
                    10 + sign * (6 if received else 0)))
        per_batch = est.batch_specific_jumps(records, 2.0, 'uniform')
        itt_b1 = per_batch['b1']['itt']['estimate']
        itt_b2 = per_batch['b2']['itt']['estimate']
        self.assertGreater(itt_b1, 0)
        self.assertLess(itt_b2, 0)

    def test_discrete_detection_across_batches(self):
        records = [_record(i, 'b1', d, d >= 0, 10)
                   for i, d in enumerate((-1, 0, 1))]
        summary = support_summary(records)
        self.assertTrue(summary['discrete_support'])


if __name__ == '__main__':
    unittest.main()
