"""诊断模块：堆积、基线、稀缺、伪阈值、donut 与敏感性网格。"""
import unittest
from threshold_data import diagnostics as diag
from threshold_data.simulation import (dgp_fuzzy, dgp_manipulation,
                                       dgp_no_first_stage)


class DiagnosticsTests(unittest.TestCase):
    def test_manipulation_triggers_alarm(self):
        records, _ = dgp_manipulation(n=300, seed=5)
        bunch = diag.bunching_test(records, 0.5)
        base = diag.baseline_continuity_test(records, 0.5)
        self.assertTrue(bunch['flag'] or base['flag'])

    def test_clean_dgp_no_baseline_alarm(self):
        records, _ = dgp_fuzzy(n=300, seed=9)
        base = diag.baseline_continuity_test(records, 0.5)
        self.assertFalse(base['flag'])

    def test_scarcity_detects_tiny_window(self):
        records, _ = dgp_fuzzy(n=60, seed=3)
        scarcity = diag.scarcity_check(records, 0.05, min_per_side=5)
        self.assertTrue(scarcity['sparse'])

    def test_first_stage_diagnostic_jump(self):
        records, _ = dgp_fuzzy(n=300, seed=8)
        fs = diag.first_stage_diagnostic(records, 0.75, 'triangular')
        self.assertGreater(fs['jump'], 0.3)
        self.assertFalse(fs['weak'])

    def test_no_first_stage_flagged_weak(self):
        records, _ = dgp_no_first_stage(n=300, seed=6)
        fs = diag.first_stage_diagnostic(records, 0.75, 'triangular')
        self.assertTrue(fs['weak'] or fs['wrong_signed'])

    def test_sensitivity_grid_reports_all_specs(self):
        records, _ = dgp_fuzzy(n=200, seed=10)
        rows = diag.sensitivity_grid(
            records, window_grid=(0.4, 0.6, 0.8),
            weightings=('uniform', 'triangular'),
            donut_options=(0, 1))
        # 3 窗口 × 2 权重 × 2 donut = 12 行，全部报告，不筛选
        self.assertEqual(len(rows), 12)
        self.assertTrue(all('donut_removed' in row for row in rows))

    def test_placebo_bundle_structure(self):
        records, _ = dgp_fuzzy(n=200, seed=12)
        bundle = diag.diagnostic_bundle(
            records, window_grid=(0.4, 0.6))
        self.assertIn('placebo', bundle)
        self.assertIn('assumption_note', bundle)
        self.assertTrue(bundle['cells'])


if __name__ == '__main__':
    unittest.main()
