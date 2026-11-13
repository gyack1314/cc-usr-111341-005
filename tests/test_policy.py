"""政策资格规则与阈值距离测试。"""
import unittest
from threshold_data.policy import is_eligible, signed_distance, policy_map


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.policies = policy_map([
            {'id': 'above_incl', 'cutoff': 60, 'eligible_above': True,
             'inclusive': True},
            {'id': 'above_excl', 'cutoff': 60, 'eligible_above': True,
             'inclusive': False},
            {'id': 'below_incl', 'cutoff': 60, 'eligible_above': False,
             'inclusive': True},
            {'id': 'below_excl', 'cutoff': 60, 'eligible_above': False,
             'inclusive': False},
        ])

    def test_above_inclusive_equality(self):
        p = self.policies['above_incl']
        self.assertTrue(is_eligible(p, 60))
        self.assertTrue(is_eligible(p, 61))
        self.assertFalse(is_eligible(p, 59))
        self.assertEqual(signed_distance(p, 60), 0)
        self.assertEqual(signed_distance(p, 58), -2)

    def test_above_exclusive_equality(self):
        p = self.policies['above_excl']
        self.assertFalse(is_eligible(p, 60))
        self.assertTrue(is_eligible(p, 61))

    def test_below_direction(self):
        incl = self.policies['below_incl']
        excl = self.policies['below_excl']
        self.assertTrue(is_eligible(incl, 60))
        self.assertTrue(is_eligible(incl, 59))
        self.assertFalse(is_eligible(incl, 61))
        self.assertFalse(is_eligible(excl, 60))
        self.assertTrue(is_eligible(excl, 59))
        # 资格一侧距离为正
        self.assertEqual(signed_distance(incl, 55), 5)
        self.assertEqual(signed_distance(incl, 65), -5)

    def test_policy_order_recorded(self):
        self.assertEqual(self.policies['below_excl']['policy_order'], 3)


if __name__ == '__main__':
    unittest.main()
