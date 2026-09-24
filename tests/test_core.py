from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from voila3d.data import assign_splits, canonical_smiles, scaffold_for_smiles
from voila3d.metrics import per_sample_loss
from voila3d.routing import _select_top, route_predictions, true_benefit, uncertainty_scores


class DataTests(unittest.TestCase):
    def test_canonical_smiles(self) -> None:
        self.assertEqual(canonical_smiles("C(C)O"), "CCO")
        self.assertIsNone(canonical_smiles("not-a-smiles"))

    def test_randomized_scaffold_splits_are_seeded_and_disjoint(self) -> None:
        smiles = [
            "c1ccccc1", "Cc1ccccc1", "Oc1ccccc1", "c1ccncc1", "Cc1ccncc1",
            "C1CCCCC1", "CC1CCCCC1", "C1CCNCC1", "CC1CCNCC1", "c1ccc2ccccc2c1",
            "CCO", "CCN", "CCCO", "CCCCN", "CC(=O)O", "CC(=O)N",
        ]
        frame = pd.DataFrame(
            {
                "canonical_smiles": smiles,
                "y": np.arange(len(smiles)) % 2,
            }
        )
        split_a = assign_splits(frame, "scaffold_randomized", seed=1, task_type="classification")
        split_b = assign_splits(frame, "scaffold_randomized", seed=2, task_type="classification")
        self.assertFalse(split_a.equals(split_b))
        for split in (split_a, split_b):
            scaffold_sets = {
                name: {
                    scaffold_for_smiles(smi)
                    for smi in frame.loc[split == name, "canonical_smiles"]
                }
                for name in ("train", "val", "test")
            }
            self.assertFalse(scaffold_sets["train"] & scaffold_sets["val"])
            self.assertFalse(scaffold_sets["train"] & scaffold_sets["test"])
            self.assertFalse(scaffold_sets["val"] & scaffold_sets["test"])


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

    def test_regression_uncertainty_uses_non_test_reference(self) -> None:
        reference = np.array([0.0, 1.0, 2.0])
        test_predictions = np.array([10.0, 11.0])
        scores = uncertainty_scores("regression", test_predictions, reference)
        expected = np.abs((test_predictions - reference.mean()) / reference.std())
        np.testing.assert_allclose(scores, expected)


if __name__ == "__main__":
    unittest.main()
