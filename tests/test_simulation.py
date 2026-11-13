"""独立 DGP 的偏差、覆盖率与闸门行为验证（小重复数快速版）。"""
import unittest
from threshold_data import simulation as sim


class SimulationValidationTests(unittest.TestCase):
    def test_continuous_coverage_near_nominal(self):
        result = sim.run_continuous_validation(reps=80)
        self.assertGreaterEqual(result['itt_coverage'], 0.85)
        self.assertGreaterEqual(result['fs_coverage'], 0.85)
        self.assertLess(abs(result['itt_bias']), 0.5)
        self.assertAlmostEqual(result['late_mean'],
                               result['late_truth'], delta=0.6)

    def test_weak_fs_gate_always_triggers(self):
        result = sim.run_weak_fs_validation(reps=60)
        self.assertGreaterEqual(result['weak_fs_gate_rate'], 0.9)
        self.assertLessEqual(result['itt_false_positive_rate'], 0.12)

    def test_manipulation_alarm_rate_high(self):
        result = sim.run_manipulation_detection(reps=30)
        self.assertGreaterEqual(result['alarm_rate'], 0.8)

    def test_discrete_support_routed_to_randomization(self):
        result = sim.run_discrete_validation(reps=20)
        self.assertTrue(result['support_discrete_detected'])
        self.assertGreater(result['completed_runs'], 0)
        self.assertIsNotNone(result['mean_ci_width'])
        self.assertGreater(result['mean_ci_width'], 0)


if __name__ == '__main__':
    unittest.main()
