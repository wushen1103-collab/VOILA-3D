from __future__ import annotations

import unittest

import numpy as np

from voila3d.data import canonical_smiles
from voila3d.metrics import per_sample_loss
from voila3d.routing import _select_top, route_predictions, true_benefit, uncertainty_scores


class DataTests(unittest.TestCase):
    def test_canonical_smiles(self) -> None:
        self.assertEqual(canonical_smiles("C(C)O"), "CCO")
        self.assertIsNone(canonical_smiles("not-a-smiles"))


class RoutingTests(unittest.TestCase):
    def test_budget_selection(self) -> None:
        scores = np.array([0.1, 0.8, 0.2, 0.7, 0.4])
        selected = _select_top(scores, 40.0)
        np.testing.assert_array_equal(selected, [False, True, False, True, False])

    def test_zero_budget_uses_2d_predictions(self) -> None:
        pred2d = np.array([1.0, 2.0, 3.0])
        pred3d = np.array([4.0, 5.0, 6.0])
        routed, selected = route_predictions(pred2d, pred3d, np.ones(3), 0.0)
        np.testing.assert_array_equal(routed, pred2d)
        self.assertFalse(selected.any())

    def test_regression_utility_is_loss_reduction(self) -> None:
        y = np.array([0.0, 1.0])
        pred2d = np.array([1.0, 3.0])
        pred3d = np.array([0.25, 2.0])
        expected = per_sample_loss("regression", y, pred2d) - per_sample_loss(
            "regression", y, pred3d
        )
        np.testing.assert_allclose(true_benefit("regression", y, pred2d, pred3d), expected)

    def test_classification_uncertainty_peaks_at_half(self) -> None:
        scores = uncertainty_scores("classification", np.array([0.0, 0.5, 1.0]))
        np.testing.assert_allclose(scores, [0.0, 0.5, 0.0])


if __name__ == "__main__":
    unittest.main()
