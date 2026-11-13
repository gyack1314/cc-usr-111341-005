"""样本构建、冻结分纪律与观察窗测试。"""
import unittest
from threshold_data.io import load
from threshold_data.prep import build_sample


class PrepTests(unittest.TestCase):
    def setUp(self):
        self.sample = build_sample(load('data/normal.json'))

    def test_frozen_score_used_not_revision(self):
        # a0 冻结分 58（不合格），修订分 61（合格）；资格必须按冻结分
        a0 = next(r for r in self.sample['primary']
                  if r['application'] == 'a0')
        self.assertEqual(a0['frozen_score'], 58)
        self.assertFalse(a0['eligible'])
        audit = self.sample['revision_audit']['a0']
        self.assertEqual(audit['revision_count'], 1)
        self.assertTrue(audit['post_publication_flip'])
        self.assertEqual(audit['details'][0]['known_day'], 120)

    def test_first_valid_application_is_primary(self):
        firms = [r['firm'] for r in self.sample['primary']]
        self.assertEqual(len(firms), len(set(firms)))
        self.assertEqual(len(self.sample['primary']), 5)

    def test_repeated_firm_goes_to_auxiliary(self):
        tables = load('data/normal.json')
        extra = dict(tables['applications'][0])
        extra['id'] = 'a9'
        extra['firm'] = 'f0'
        extra['batch'] = 'b2'
        extra['score'] = 75
        tables['applications'].append(extra)
        sample = build_sample(tables)
        self.assertEqual(len(sample['primary']), 5)
        self.assertTrue(any(r['application'] == 'a9'
                            for r in sample['auxiliary']))
        self.assertTrue(any(d['application'] == 'a9'
                            for d in sample['dropped_ids']))

    def test_observation_window_exclusion(self):
        tables = load('data/normal.json')
        tables['applications'][0]['outcome_known_day'] = 120
        sample = build_sample(tables, horizon_days=90)
        self.assertTrue(any(r['application'] == 'a0'
                            for r in sample['not_visible']))
        self.assertFalse(any(r['application'] == 'a0'
                             for r in sample['primary']))

    def test_signed_distance_and_eligibility(self):
        a2 = next(r for r in self.sample['primary']
                  if r['application'] == 'a2')
        self.assertEqual(a2['distance'], 0)
        self.assertTrue(a2['eligible'])


if __name__ == '__main__':
    unittest.main()
