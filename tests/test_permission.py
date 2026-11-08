import unittest

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

from voila3d.permission import bootstrap_lcb, paired_test, risk_benefit


class PermissionTests(unittest.TestCase):
    def test_identical_predictions_deny_permission(self):
        y = np.arange(10, dtype=float)
        self.assertFalse(paired_test('regression', y, y + 1, y + 1)['permission'])
        self.assertEqual(bootstrap_lcb('regression', y, y + 1, y + 1, 4), 0)

    def test_positive_deterministic_difference(self):
        y = np.arange(10, dtype=float)
        self.assertTrue(paired_test('regression', y, y + 2, y + 1)['permission'])

    def test_swap_reverses_gain(self):
        rng = np.random.default_rng(4)
        y = np.array([0] * 20 + [1] * 20)
        a, b = rng.random((2, 40))
        self.assertAlmostEqual(paired_test('classification', y, a, b)['gain'],
                               -paired_test('classification', y, b, a)['gain'])

    def test_stratified_bootstrap_matches_original_metric_loop(self):
        y = np.array([0, 0, 1, 1, 1, 0, 1])
        reference = np.array([.1, .6, .8, .5, .5, .4, .8])
        augmented = np.array([.2, .3, .8, .8, .4, .3, .6])
        rng = np.random.default_rng(33)
        positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        values = []
        for _ in range(99):
            indices = np.r_[rng.choice(positive, len(positive), replace=True),
                            rng.choice(negative, len(negative), replace=True)]
            values.append(roc_auc_score(y[indices], augmented[indices]) -
                          roc_auc_score(y[indices], reference[indices]))
        self.assertAlmostEqual(bootstrap_lcb('classification', y, reference, augmented, 33, 99),
                               np.quantile(values, .025))

    def test_unconditional_reference_has_no_gain(self):
        result = risk_benefit(np.array([1., -2., 3.]), np.array([False] * 3))
        self.assertEqual(result['gain_retention'], 0)
        self.assertEqual(result['harm_avoidance'], 1)

    def test_paired_delong_uses_covariance(self):
        y = np.r_[np.zeros(10), np.ones(10)]
        prediction = np.arange(20.)
        result = paired_test('classification', y, prediction, prediction + 1)
        self.assertEqual(result['pvalue'], 1)
        self.assertFalse(result['permission'])

    def test_regression_probability_matches_scipy(self):
        y = np.zeros(6)
        reference = np.array([1., 2., 3., 4., 5., 6.])
        augmented = np.array([.9, 1.9, 2.7, 3.8, 4.6, 5.8])
        expected = stats.ttest_rel(abs(y - reference), abs(y - augmented), alternative='greater')
        self.assertAlmostEqual(paired_test('regression', y, reference, augmented)['pvalue'], expected.pvalue)

    def test_insufficient_classification_evidence_denies(self):
        result = paired_test('classification', np.array([0, 0, 1]),
                             np.array([.3, .4, .5]), np.array([.1, .2, .9]))
        self.assertFalse(result['permission'])

    def test_no_negative_mass_is_undefined_not_perfect_avoidance(self):
        result = risk_benefit(np.array([1., 2.]), np.array([True, True]))
        self.assertTrue(np.isnan(result['harm_avoidance']))


if __name__ == '__main__':
    unittest.main()
