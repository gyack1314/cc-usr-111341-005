"""端到端流水线：产物留痕、口径纪律与两个业务资料包的结论。"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from threshold_data.pipeline import analyze, render_markdown

REQUIRED_ARTIFACTS = ['included_firms.jsonl', 'exclusions.json',
                      'revisions_audit.json', 'diagnostics.json',
                      'estimates.json', 'sensitivity.csv',
                      'report.md', 'report.json']


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, manifest):
        findings, target = analyze(manifest, out_dir=self.tmp)
        for name in REQUIRED_ARTIFACTS:
            self.assertTrue((target / name).exists(),
                            f'缺少产物 {name}')
        return findings, target

    def test_normal_package_routes_discrete(self):
        findings, target = self._run('data/normal.json')
        self.assertEqual(findings['package_path'],
                         'fisher_randomization')
        # 修订分 a0 会翻转资格但冻结分仍判不合格
        self.assertEqual(
            findings['frozen_score_audit']
            ['post_publication_revision_flips'], 1)
        weights = [json.loads(line) for line in
                   (target / 'included_firms.jsonl').read_text(
                       encoding='utf-8').splitlines()]
        self.assertTrue(weights)
        self.assertIn('weight', weights[0])
        # 小样本第一阶段不显著 → 停止 LATE
        ri_ = findings['randomization']
        self.assertEqual(ri_['status'], 'weak_first_stage')
        self.assertIn('late_note', ri_)

    def test_boundary_package_no_takeup_jump(self):
        findings, _ = self._run('data/boundary.json')
        fs = findings['diagnostics']['first_stage']
        self.assertEqual(fs['jump'], 0.0)
        self.assertTrue(fs['weak'] or fs['wrong_signed'])
        report = render_markdown(findings)
        self.assertIn('停止报告 LATE', report)
        self.assertIn('扩围', report)

    def test_sensitivity_grid_never_selects_significant_bandwidth(self):
        findings, target = self._run('data/normal.json')
        lines = (target / 'sensitivity.csv').read_text(
            encoding='utf-8').strip().splitlines()
        # 表头 + 至少一组窗口×权重×donut
        self.assertGreater(len(lines), 2)
        self.assertIn('donut_cells', lines[0])

    def test_scope_limits_present(self):
        findings, _ = self._run('data/normal.json')
        joined = ' '.join(findings['scope_and_limits'])
        for keyword in ('局部', '扩围', '排除限制', '单调性'):
            self.assertIn(keyword, joined)

    def test_report_contains_diagnostic_before_effect(self):
        findings, _ = self._run('data/normal.json')
        report = render_markdown(findings)
        self.assertLess(report.index('诊断结果'),
                        report.index('效果解释'))


if __name__ == '__main__':
    unittest.main()
