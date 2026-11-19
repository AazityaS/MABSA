import unittest

import torch

from src.structured_loss import StructuredSetLoss, hungarian_assignment
from src.structured_metrics import (evaluate_by_language, evaluate_structured,
                                    paired_hotel_bootstrap, select_objectness_threshold)
from src.structured_model import StructuredSetModel


def item(category=0, sentiment=0, aspect_implicit=0, opinion_implicit=0):
    return {"category": category, "sentiment": sentiment,
            "aspect_implicit": aspect_implicit, "opinion_implicit": opinion_implicit}


class TestStructuredProtocol(unittest.TestCase):
    def test_hungarian_assignment_preserves_duplicate_targets(self):
        rows, cols = hungarian_assignment(torch.tensor([[3.0, 0.0], [0.0, 3.0]]))
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(set(rows)), 2)
        self.assertEqual(len(set(cols)), 2)
        self.assertEqual(sum(torch.tensor([[3.0, 0.0], [0.0, 3.0]])[r, c].item() for r, c in zip(rows, cols)), 0.0)

    def test_duplicate_targets_are_multiset_scored(self):
        gold = [[item(2, 2, 0, 0), item(2, 2, 0, 0)]]
        exact = evaluate_structured([[item(2, 2, 0, 0), item(2, 2, 0, 0)]], gold)
        one_missing = evaluate_structured([[item(2, 2, 0, 0)]], gold)
        self.assertEqual(exact["complete_set"]["f1"], 1.0)
        self.assertEqual(exact["complete_set"]["exact_set_accuracy"], 1.0)
        self.assertEqual(one_missing["complete_set"]["tp"], 1)
        self.assertEqual(one_missing["complete_set"]["fn"], 1)
        self.assertAlmostEqual(one_missing["cardinality"]["mae"], 1.0)

    def test_incorrect_implicitness_separates_diagnostic_from_primary(self):
        gold = [[item(3, 1, 1, 0)]]
        pred = [[item(3, 1, 0, 0)]]
        metrics = evaluate_structured(pred, gold)
        self.assertEqual(metrics["primary_implicit_category_sentiment"]["tp"], 0)
        self.assertEqual(metrics["primary_implicit_category_sentiment"]["fn"], 1)
        self.assertEqual(metrics["gold_implicit_conditioned"]["exact_category_sentiment_rate"], 1.0)
        self.assertEqual(metrics["aspect_implicit_positive"]["fn"], 1)

    def test_unmatched_prediction_counts_false_positive(self):
        gold = [[item(0, 2, 0, 0)]]
        pred = [[item(0, 2, 0, 0), item(5, 0, 1, 0)]]
        metrics = evaluate_structured(pred, gold)
        self.assertEqual(metrics["complete_set"]["tp"], 1)
        self.assertEqual(metrics["complete_set"]["fp"], 1)
        self.assertEqual(metrics["primary_implicit_category_sentiment"]["fp"], 1)

    def test_empty_prediction_is_missed_gold(self):
        metrics = evaluate_structured([[]], [[item(2, 2, 1, 0)]])
        self.assertEqual(metrics["primary_implicit_category_sentiment"]["fn"], 1)
        self.assertEqual(metrics["complete_set"]["fn"], 1)
        self.assertEqual(metrics["complete_set"]["exact_set_accuracy"], 0.0)
        self.assertEqual(metrics["cardinality"]["mae"], 1.0)

    def test_end_to_end_implicit_component_macro_f1_uses_implicit_gate(self):
        gold = [[item(0, 0, 1), item(1, 1, 1)]]
        pred = [[item(0, 1, 1), item(1, 1, 1)]]
        metrics = evaluate_structured(pred, gold)
        component = metrics["end_to_end_implicit_components"]
        self.assertAlmostEqual(component["category"]["macro_f1"], 2 / 6)
        self.assertAlmostEqual(component["sentiment"]["macro_f1"], 2 / 9)
        self.assertEqual(component["category"]["gold_support_by_class"]["AMENITY"], 1)
        self.assertEqual(metrics["primary_implicit_category_sentiment"]["tp"], 1)
        self.assertEqual(metrics["primary_implicit_category_sentiment"]["fp"], 1)
        self.assertEqual(metrics["primary_implicit_category_sentiment"]["fn"], 1)

    def test_end_to_end_component_scores_preserve_implicitness_false_positives_and_misses(self):
        missed = evaluate_structured([[item(2, 2, 0)]], [[item(2, 2, 1)]])
        cat = missed["end_to_end_implicit_components"]["category"]["per_class"]["EXPERIENCE"]
        sent = missed["end_to_end_implicit_components"]["sentiment"]["per_class"]["positive"]
        self.assertEqual(cat["fn"], 1)
        self.assertEqual(sent["fn"], 1)
        spurious = evaluate_structured([[item(5, 0, 1)]], [[item(0, 2, 0)]])
        cat_fp = spurious["end_to_end_implicit_components"]["category"]["per_class"]["SERVICE"]
        sent_fp = spurious["end_to_end_implicit_components"]["sentiment"]["per_class"]["negative"]
        self.assertEqual(cat_fp["fp"], 1)
        self.assertEqual(sent_fp["fp"], 1)

    def test_diagnostic_confusion_matrices_include_miss_ignore_extra_and_use_names(self):
        category_names = [f"C{i}" for i in range(6)]
        sentiment_names = [f"S{i}" for i in range(3)]
        gold = [[item(1, 1, 1)], [item(1, 1, 1), item(2, 2, 1)]]
        pred = [[item(1, 1, 0), item(4, 1, 1)], [item(1, 1, 1)]]
        matrices = evaluate_structured(pred, gold, category_names=category_names,
                                       sentiment_names=sentiment_names)["gold_implicit_conditioned"]["confusion_matrices"]
        cat, sent = matrices["category"], matrices["sentiment"]
        self.assertEqual(cat["rows_gold"], category_names)
        self.assertEqual(cat["columns_prediction"], category_names + ["MISS"])
        self.assertEqual(cat["values"][1][1], 2)
        self.assertEqual(cat["values"][2][6], 1)
        self.assertEqual(sum(map(sum, cat["values"])), 3)
        self.assertEqual(sent["rows_gold"], sentiment_names)
        self.assertEqual(sent["columns_prediction"], sentiment_names + ["MISS"])
        self.assertEqual(sent["values"][1][1], 2)
        self.assertEqual(sent["values"][2][3], 1)
        self.assertEqual(sum(map(sum, sent["values"])), 3)

    def test_diagnostic_confusion_matrices_preserve_duplicate_items(self):
        gold = [[item(0, 0, 1), item(0, 0, 1)]]
        pred = [[item(0, 0, 0), item(0, 0, 1)]]
        matrices = evaluate_structured(pred, gold)["gold_implicit_conditioned"]["confusion_matrices"]
        self.assertEqual(matrices["category"]["values"][0][0], 2)
        self.assertEqual(matrices["sentiment"]["values"][0][0], 2)

    def test_per_language_aggregation_reports_support_and_withholds_unsupported_macros(self):
        gold = [[item(0, 0, 1)], [item(1, 1, 1), item(2, 2, 0)]]
        pred = [[item(0, 0, 1)], []]
        result = evaluate_by_language(pred, gold, ["en", "vi"])
        en, vi = result["en"], result["vi"]
        self.assertEqual(en["support"]["eligible_segments"], 1)
        self.assertEqual(en["support"]["gold_implicit_quad_count"], 1)
        self.assertEqual(en["support"]["primary_predicted_implicit_count"], 1)
        self.assertEqual(en["support"]["primary_gold_implicit_count"], 1)
        self.assertEqual(en["metrics"]["primary_implicit_category_sentiment"]["tp"], 1)
        self.assertEqual(en["metrics"]["primary_implicit_category_sentiment"]["f1"], 1.0)
        self.assertIsNone(en["metrics"]["end_to_end_implicit_components"]["category"]["macro_f1"])
        self.assertIn("withheld", en["metrics"]["end_to_end_implicit_components"]["category"]["macro_f1_status"])
        self.assertEqual(vi["support"]["eligible_segments"], 1)
        self.assertEqual(vi["support"]["gold_implicit_quad_count"], 1)
        self.assertEqual(vi["metrics"]["primary_implicit_category_sentiment"]["fn"], 1)

    def test_23_slot_forward_backward_and_loss(self):
        model = StructuredSetModel(text_dim=12, image_dim=8, hidden_size=16, slots=23, multimodal=True)
        self.assertEqual(model.multimodal_fusion[0].in_features, 32)
        self.assertEqual(model.multimodal_fusion[0].out_features, 16)
        text = torch.randn(2, 12)
        images = torch.randn(2, 3, 8)
        image_mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
        outputs = model(text, torch.tensor([0, 1]), images, image_mask)
        target = {"category": torch.tensor([[1, 1, 2], [0, 0, 0]]),
                  "sentiment": torch.tensor([[2, 2, 0], [1, 0, 0]]),
                  "aspect_implicit": torch.tensor([[0., 0., 1.], [0., 0., 0.]]),
                  "opinion_implicit": torch.tensor([[0., 0., 0.], [0., 0., 0.]]),
                  "mask": torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.bool)}
        loss = StructuredSetLoss()(outputs, target)
        self.assertTrue(torch.isfinite(loss["loss"]))
        loss["loss"].backward()
        self.assertIsNotNone(model.slot_embeddings.grad)
        self.assertEqual(outputs["category"].shape, (2, 23, 6))
        self.assertEqual(outputs["sentiment"].shape, (2, 23, 3))

    def test_text_only_path_has_no_image_fusion(self):
        model = StructuredSetModel(text_dim=12, image_dim=8, hidden_size=16, slots=23, multimodal=False)
        outputs = model(torch.randn(2, 12), torch.tensor([0, 1]))
        self.assertFalse(hasattr(model, "multimodal_fusion"))
        self.assertEqual(outputs["category"].shape, (2, 23, 6))

    def test_cardinality_23_supported_without_truncation(self):
        model = StructuredSetModel(text_dim=4, image_dim=4, hidden_size=8, slots=23, multimodal=False)
        outputs = model(torch.randn(1, 4), torch.tensor([2]))
        target = {"category": torch.arange(23) % 6,
                  "sentiment": torch.arange(23) % 3,
                  "aspect_implicit": torch.zeros(23),
                  "opinion_implicit": torch.zeros(23),
                  "mask": torch.ones(23, dtype=torch.bool)}
        target = {k: v.unsqueeze(0) for k, v in target.items()}
        loss = StructuredSetLoss()(outputs, target)
        self.assertTrue(torch.isfinite(loss["loss"]))
        self.assertEqual(len(loss["assignments"][0][0]), 23)

    def test_threshold_grid_tie_breaks_to_highest(self):
        outputs = {"objectness": torch.zeros(1, 23),
                   "category": torch.zeros(1, 23, 6),
                   "sentiment": torch.zeros(1, 23, 3),
                   "aspect_implicit": torch.zeros(1, 23),
                   "opinion_implicit": torch.zeros(1, 23)}
        gold = [[]]
        selected = select_objectness_threshold(outputs, gold, [0.0, 0.5, 1.0])
        self.assertEqual(selected["threshold"], 1.0)

    def test_hotel_bootstrap_is_paired_and_deterministic(self):
        gold = [[item(1, 2, 1, 0)], [item(3, 0, 1, 0)]]
        pred = [[item(1, 2, 1, 0)], [item(3, 0, 1, 0)]]
        ci = paired_hotel_bootstrap(pred, pred, gold, ["hotel-a", "hotel-b"],
                                    replicates=20, seed=44)
        self.assertEqual(ci["difference_ci95"], [0.0, 0.0])
        self.assertEqual(ci["replicates"], 20)
        self.assertEqual(ci["multimodal_better_replicates"], 0)
        self.assertEqual(ci["ties"], 20)
        self.assertEqual(ci["text_only_better_replicates"], 0)
        self.assertEqual(ci["proportion_multimodal_better"], 0.0)

    def test_hotel_bootstrap_win_proportion_for_consistent_multimodal_wins(self):
        gold = [[item(1, 2, 1, 0)], [item(3, 0, 1, 0)]]
        text = [[], []]
        multimodal = [[item(1, 2, 1, 0)], [item(3, 0, 1, 0)]]
        ci = paired_hotel_bootstrap(text, multimodal, gold, ["hotel-a", "hotel-b"],
                                    replicates=40, seed=1729)
        self.assertEqual(ci["multimodal_better_replicates"], 40)
        self.assertEqual(ci["ties"], 0)
        self.assertEqual(ci["text_only_better_replicates"], 0)
        self.assertEqual(ci["proportion_multimodal_better"], 1.0)
        reverse = paired_hotel_bootstrap(multimodal, text, gold, ["hotel-a", "hotel-b"],
                                         replicates=40, seed=1729)
        self.assertEqual(reverse["multimodal_better_replicates"], 0)
        self.assertEqual(reverse["ties"], 0)
        self.assertEqual(reverse["text_only_better_replicates"], 40)
        self.assertEqual(reverse["proportion_multimodal_better"], 0.0)


if __name__ == "__main__": unittest.main()
