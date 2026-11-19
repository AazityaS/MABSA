import unittest

import torch

from src.structured_loss import StructuredSetLoss


class TestExpectedCardinalityLoss(unittest.TestCase):
    def _inputs(self, batch=2):
        return {
            "objectness": torch.zeros(batch, 23, requires_grad=True),
            "category": torch.zeros(batch, 23, 6, requires_grad=True),
            "sentiment": torch.zeros(batch, 23, 3, requires_grad=True),
            "aspect_implicit": torch.zeros(batch, 23, requires_grad=True),
            "opinion_implicit": torch.zeros(batch, 23, requires_grad=True),
        }

    def _targets(self):
        return {
            "category": torch.zeros(2, 2, dtype=torch.long),
            "sentiment": torch.zeros(2, 2, dtype=torch.long),
            "aspect_implicit": torch.zeros(2, 2),
            "opinion_implicit": torch.zeros(2, 2),
            "mask": torch.tensor([[True, True], [True, False]]),
        }

    def test_expected_count_uses_sigmoid_and_gold_mask_count(self):
        outputs = self._inputs()
        # Zero logits yield an expected count of 11.5 for both rows. Gold
        # counts are 2 and 1, so SmoothL1(beta=1) averages 9 and 10.
        criterion = StructuredSetLoss(category=0, sentiment=0, aspect_implicit=0,
                                      opinion_implicit=0, objectness_positive=0,
                                      objectness_empty=0, cardinality=1)
        result = criterion(outputs, self._targets())
        self.assertAlmostEqual(result["cardinality"].item(), (9.0 + 10.0) / 2)
        self.assertAlmostEqual(result["cardinality_weighted"].item(), 9.5)
        self.assertAlmostEqual(result["loss"].item(), 9.5)

    def test_cardinality_term_is_differentiable_and_finite(self):
        outputs = self._inputs()
        criterion = StructuredSetLoss(category=0, sentiment=0, aspect_implicit=0,
                                      opinion_implicit=0, objectness_positive=0,
                                      objectness_empty=0, cardinality=0.1334)
        result = criterion(outputs, self._targets())
        self.assertTrue(torch.isfinite(result["loss"]))
        result["loss"].backward()
        self.assertIsNotNone(outputs["objectness"].grad)
        self.assertTrue(torch.isfinite(outputs["objectness"].grad).all())
        self.assertGreater(outputs["objectness"].grad.abs().sum().item(), 0)

    def test_zero_weight_preserves_existing_objective(self):
        outputs = self._inputs()
        targets = self._targets()
        baseline = StructuredSetLoss(cardinality=0.0)(outputs, targets)
        weighted = StructuredSetLoss(cardinality=0.25)(outputs, targets)
        self.assertAlmostEqual(baseline["loss"].item(),
                               weighted["loss"].item() - weighted["cardinality_weighted"].item(),
                               places=6)


if __name__ == "__main__":
    unittest.main()
