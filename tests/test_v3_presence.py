import unittest
from types import SimpleNamespace

import numpy as np
import torch
from pathlib import Path
import json
import tempfile

from src.v3_presence_dataset import (CATEGORIES, LABEL_ORDER, NUM_LABELS, SENTIMENTS,
                                     ImplicitPresenceDataset, assert_hotel_disjoint, encode_implicit_presence,
                                     load_split_language_metadata, validate_target)
from src.v3_presence_metrics import evaluate_presence, select_presence_threshold
from src.v3_presence_train import compute_train_pos_weight


class TestV3ImplicitPresence(unittest.TestCase):
    def test_label_order_matches_6_by_3_cartesian_product(self):
        self.assertEqual(len(LABEL_ORDER), 18)
        self.assertEqual(LABEL_ORDER[0], "AMENITY / negative")
        self.assertEqual(LABEL_ORDER[2], "AMENITY / positive")
        self.assertEqual(LABEL_ORDER[3], "BRANDING / negative")
        self.assertEqual(LABEL_ORDER[-1], "SERVICE / positive")
        self.assertEqual(LABEL_ORDER, tuple(f"{c} / {s}" for c in CATEGORIES for s in SENTIMENTS))

    def test_only_aspect_implicit_quads_set_labels(self):
        target, counts = encode_implicit_presence([
            {"aspect_implicit": True, "aspect_category": "FACILITY", "sentiment": "positive"},
            {"aspect_implicit": False, "aspect_category": "SERVICE", "sentiment": "negative"},
        ])
        self.assertEqual(target.shape, (18,))
        self.assertEqual(int(target.sum()), 1)
        self.assertEqual(target[11].item(), 1.0)
        self.assertEqual(counts["implicit_quad_count"], 1)

    def test_repeated_pair_collapses_to_one_presence(self):
        target, counts = encode_implicit_presence([
            {"aspect_implicit": True, "aspect_category": "EXPERIENCE", "sentiment": "positive"},
            {"aspect_implicit": True, "aspect_category": "EXPERIENCE", "sentiment": "positive"},
            {"aspect_implicit": True, "aspect_category": "FACILITY", "sentiment": "positive"},
        ])
        self.assertEqual(int(target.sum()), 2)
        self.assertEqual(counts, {"implicit_quad_count": 3, "unique_implicit_pair_count": 2,
                                  "collapsed_repeat_count": 1})

    def test_all_zero_target_for_no_implicit_aspects(self):
        target, counts = encode_implicit_presence([
            {"aspect_implicit": False, "aspect_category": "FACILITY", "sentiment": "negative"}
        ])
        self.assertEqual(target.tolist(), [0.0] * NUM_LABELS)
        self.assertEqual(counts["collapsed_repeat_count"], 0)
        validate_target(target)

    def test_split_hotel_separation(self):
        assert_hotel_disjoint(SimpleNamespace(hotel_ids={"H1", "H2"}),
                              SimpleNamespace(hotel_ids={"H3"}))
        with self.assertRaises(AssertionError):
            assert_hotel_disjoint(SimpleNamespace(hotel_ids={"H1"}),
                                  SimpleNamespace(hotel_ids={"H1", "H3"}))

    def test_official_train_dev_data_shapes_supports_and_hotel_separation(self):
        root = Path(__file__).resolve().parents[1]
        train = ImplicitPresenceDataset(root, "train")
        dev = ImplicitPresenceDataset(root, "dev")
        assert_hotel_disjoint(train, dev)
        self.assertEqual((len(train), len(dev)), (8816, 1895))
        for dataset in (train, dev):
            self.assertTrue(all(row["target"].shape == (18,) for row in dataset.records))
            self.assertTrue(all(torch.all((row["target"] == 0) | (row["target"] == 1))
                                for row in dataset.records))
        self.assertEqual(sum(int(row["target"].sum() > 0) for row in train.records), 370)
        self.assertEqual(sum(int(row["target"].sum() > 0) for row in dev.records), 77)
        self.assertEqual(sum(row["metadata"]["collapsed_repeat_count"] for row in train.records), 96)
        self.assertEqual(sum(row["metadata"]["collapsed_repeat_count"] for row in dev.records), 21)
        self.assertEqual(sum(int(row["target"].sum()) for row in train.records), 451)
        self.assertEqual(sum(int(row["target"].sum()) for row in dev.records), 98)

    def test_dataset_rejects_test_split_before_reading_any_file(self):
        with self.assertRaises(ValueError):
            ImplicitPresenceDataset("path-that-must-not-be-opened", "test")

    def test_threshold_selection_uses_micro_f1_and_highest_tie(self):
        probs = np.zeros((1, 18), dtype=float)
        targets = np.zeros((1, 18), dtype=int)
        result = select_presence_threshold(probs, targets, [0.0, 0.5, 1.0])
        self.assertEqual(result["threshold"], 1.0)
        self.assertEqual(result["dev_micro_f1"], 0.0)

    def test_train_only_weight_derivation_and_zero_support_rule(self):
        train = torch.zeros((3, 18), dtype=torch.float32)
        train[0, 0] = 1
        weights, details = compute_train_pos_weight(train, cap=2.0)
        self.assertEqual(details["source"], "train_targets_only")
        self.assertEqual(weights[0].item(), 2.0)
        self.assertEqual(weights[1].item(), 1.0)
        self.assertIn(LABEL_ORDER[1], details["zero_train_support"])
        self.assertEqual(details["positive_support"][LABEL_ORDER[0]], 1)

    def test_presence_metrics_confusion_macro_exact_set_and_cardinality(self):
        probs = np.zeros((2, 18), dtype=float)
        probs[0, 0] = 0.9
        probs[1, 0] = 0.8
        probs[1, 1] = 0.9
        targets = np.zeros((2, 18), dtype=int)
        targets[0, 0] = 1
        result = evaluate_presence(probs, targets, 0.5)
        self.assertEqual(result["micro"]["tp"], 1)
        self.assertEqual(result["micro"]["fp"], 2)
        self.assertEqual(result["micro"]["fn"], 0)
        self.assertAlmostEqual(result["micro"]["precision"], 1 / 3)
        self.assertEqual(result["per_label"][LABEL_ORDER[0]]["support"], 1)
        self.assertEqual(result["macro_unsupported_labels"], list(LABEL_ORDER[1:]))
        self.assertAlmostEqual(result["macro_f1"], (2 / 3) / 18)
        self.assertEqual(result["exact_set_accuracy"], 0.5)
        self.assertEqual(result["cardinality_mae"], 1.0)
        self.assertEqual(result["cardinality_signed_bias"], 1.0)

    def test_empty_batch_shapes_are_rejected_and_all_zero_prediction_supported(self):
        result = evaluate_presence(np.zeros((1, 18)), np.zeros((1, 18), dtype=int), 0.5)
        self.assertEqual(result["micro"]["tp"], 0)
        self.assertEqual(result["micro"]["fp"], 0)
        self.assertEqual(result["micro"]["fn"], 0)
        self.assertEqual(result["exact_set_accuracy"], 1.0)
        with self.assertRaises(ValueError):
            evaluate_presence(np.zeros((1, 17)), np.zeros((1, 17), dtype=int), 0.5)

    def test_ids_and_language_stay_metadata_not_model_features(self):
        from src.v3_presence_dataset import collate_presence

        class FakeTokenizer:
            def __call__(self, texts, **kwargs):
                return {"input_ids": torch.ones((len(texts), 3), dtype=torch.long),
                        "attention_mask": torch.ones((len(texts), 3), dtype=torch.long)}

        sample = {"text": "sample text", "target": torch.zeros(18),
                  "metadata": {"segment_id": "seg", "source_review_id": "review",
                               "hotel_id": "hotel", "language": "vi"}}
        batch = collate_presence([sample], FakeTokenizer())
        self.assertEqual(set(batch), {"input_ids", "attention_mask", "targets", "metadata"})
        self.assertEqual(batch["metadata"][0]["language"], "vi")
        self.assertNotIn("hotel_id", batch)
        self.assertNotIn("segment_id", batch)

    def test_language_join_decodes_only_requested_metadata_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            metadata = Path(temp) / "metadata"
            metadata.mkdir()
            rows = [
                {"segment_id": "unrequested", "language": "other", "segment_text": "not retained"},
                {"segment_id": "requested", "language": "vi", "segment_text": "train text"},
            ]
            (metadata / "segments.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            self.assertEqual(load_split_language_metadata(temp, {"requested"}), {"requested": "vi"})


if __name__ == "__main__":
    unittest.main()
