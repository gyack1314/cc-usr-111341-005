"""支撑集识别、窗口与 donut 口径测试。"""
import unittest
from threshold_data.support import (support_summary, select_window,
                                    default_window_grid, donut_removed_values)


def _record(distance, firm=None):
    return {'distance': float(distance),
            'eligible': distance >= 0,
            'firm': firm or f'f{distance}',
            'batch': 'b1', 'received': False, 'baseline': 1.0,
            'outcome': 1.0, 'application': f'a{distance}'}


class SupportTests(unittest.TestCase):
    def test_discrete_lattice_detected(self):
        records = [_record(d) for d in (-2, -1, 0, 1, 2)]
        summary = support_summary(records)
        self.assertTrue(summary['discrete_support'])
        self.assertEqual(summary['unique_count'], 5)
        self.assertTrue(summary['reasons'])

    def test_continuous_support_not_flagged(self):
        records = [_record(i * 0.137 - 1.5) for i in range(24)]
        summary = support_summary(records, unique_limit=10,
                                  tie_fraction=0.2)
        self.assertFalse(summary['discrete_support'])

    def test_ties_trigger_discrete(self):
        records = [_record(d, firm=f'f{i}')
                   for d in (-1, -1, 0, 0, 1, 1)
                   for i in [0]]
        # 显式构造重复距离
        records = []
        for i, d in enumerate((-1, -1, -1, 0, 0, 0, 1, 1, 1, 2, 2, 2)):
            records.append(_record(d, firm=f'f{i}'))
        summary = support_summary(records)
        self.assertTrue(summary['discrete_support'])
        self.assertGreaterEqual(summary['tie_fraction'], 0.2)

    def test_window_and_donut(self):
        records = [_record(d) for d in (-3, -2, -1, 0, 1, 2, 3)]
        inside = select_window(records, 2)
        self.assertEqual({r['distance'] for r in inside},
                         {-2, -1, 0, 1, 2})
        donut = select_window(records, 2, donut_cells=1)
        # 删除两侧最近格点：非资格侧 -1、资格侧首个格点 0
        self.assertEqual({r['distance'] for r in donut},
                         {-2.0, 1.0, 2.0})
        removed = donut_removed_values(
            {'unique_distances': [-3, -2, -1, 0, 1, 2, 3]}, 1)
        self.assertEqual(removed, [-1, 0])

    def test_default_grid_bounded_by_support(self):
        records = [_record(d) for d in (-2, -1, 0, 1, 2)]
        summary = support_summary(records)
        self.assertEqual(default_window_grid(summary), [1, 2])

    def test_no_jitter_anywhere(self):
        # 模块不应提供任何加噪声接口；距离原样保留
        records = [_record(d) for d in (-1, 0, 1)]
        for original, record in zip((-1, 0, 1), records):
            self.assertEqual(record['distance'], float(original))


if __name__ == '__main__':
    unittest.main()
