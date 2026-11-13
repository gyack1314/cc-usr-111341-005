"""统计分布、线性代数与聚类稳健推断的数值正确性测试。"""
import math
import unittest
from threshold_data import statsutil as su


class DistributionTests(unittest.TestCase):
    def test_normal_cdf_known_values(self):
        self.assertAlmostEqual(su.normal_cdf(0), 0.5, places=12)
        self.assertAlmostEqual(su.normal_cdf(1.96), 0.975, places=4)
        self.assertAlmostEqual(su.normal_cdf(-1.96), 0.025, places=4)

    def test_t_two_sided_p_matches_known_table(self):
        # df=10, t=2.228 双侧 p≈0.05
        p = su.t_two_sided_p(2.228, 10)
        self.assertAlmostEqual(p, 0.05, places=3)
        # df=∞ 时趋近正态
        p_large = su.t_two_sided_p(1.96, 10000)
        self.assertAlmostEqual(p_large, 0.05, places=2)

    def test_f_sf_monotone(self):
        self.assertAlmostEqual(su.f_sf(0, 2, 10), 1.0)
        self.assertLess(su.f_sf(5, 2, 10), su.f_sf(2, 2, 10))

    def test_fisher_exact_known_table(self):
        # 教科书示例 [[4,0],[0,4]] 双侧 p = 2/70
        p = su.fisher_exact_two_sided([[4, 0], [0, 4]])
        self.assertAlmostEqual(p, 2.0 / 70.0, places=6)

    def test_fisher_exact_independence_p_high(self):
        p = su.fisher_exact_two_sided([[5, 5], [5, 5]])
        self.assertGreater(p, 0.5)

    def test_binom_two_sided(self):
        self.assertAlmostEqual(su.binom_two_sided_p(0, 10, 0.5),
                               2 / 1024, places=6)


class LinearAlgebraTests(unittest.TestCase):
    def test_solve_known_system(self):
        a = [[3.0, 2.0], [1.0, 2.0]]
        b = [7.0, 5.0]
        x = su.solve(a, b)
        self.assertAlmostEqual(x[0], 1.0)
        self.assertAlmostEqual(x[1], 2.0)

    def test_inverse_roundtrip(self):
        a = [[4.0, 1.0], [2.0, 3.0]]
        inv = su.inverse(a)
        identity = su.matmul(a, inv)
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(identity[i][j],
                                       1.0 if i == j else 0.0, places=9)

    def test_ols_recovers_coefficients(self):
        # y = 2 + 3x 的确定性数据
        x = [[1.0, float(i)] for i in range(6)]
        y = [2 + 3 * i for i in range(6)]
        beta, resid, _ = su.ols(x, y)
        self.assertAlmostEqual(beta[0], 2.0, places=9)
        self.assertAlmostEqual(beta[1], 3.0, places=9)
        self.assertTrue(all(abs(r) < 1e-9 for r in resid))

    def test_cluster_vcov_reduces_groups(self):
        x = [[1.0, float(i) % 2, float(i)] for i in range(12)]
        y = [float(i) + (1 if i % 2 else 0) for i in range(12)]
        clusters = [i // 3 for i in range(12)]
        vcov, g = su.cluster_robust_vcov(x, [0.0] * 12, clusters)
        self.assertEqual(g, 4)
        self.assertTrue(all(math.isfinite(vcov[0][j])
                            for j in range(3)))

    def test_quantile_linear_interpolation(self):
        self.assertEqual(su.quantile([1, 2, 3, 4], 0.5), 2.5)
        self.assertEqual(su.quantile([1, 2, 3, 4], 0.0), 1)


if __name__ == '__main__':
    unittest.main()
