import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.v3_frozen_cache import EXPECTED_DIM, load_train_dev_text_features
from src.v3_frozen_dataset import FrozenPresenceDataset
from src.v3_frozen_model import FrozenXLMRPresenceHead
from src.v3_presence_dataset import ImplicitPresenceDataset, LABEL_ORDER, assert_hotel_disjoint
from src.v3_presence_metrics import evaluate_presence, select_presence_threshold


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache" / "structured_features" / "9072ce0b71cd3470"


class TestV3Frozen(unittest.TestCase):
    def test_embedding_to_target_alignment_uses_segment_id_not_order(self):
        records = [
            {"metadata": {"segment_id": "seg-a"}, "target": torch.tensor([1.0] + [0.0] * 17)},
            {"metadata": {"segment_id": "seg-b"}, "target": torch.tensor([0.0, 1.0] + [0.0] * 16)},
        ]
        features = {"seg-b": torch.full((768,), 2.0), "seg-a": torch.full((768,), 1.0)}
        ds = FrozenPresenceDataset(SimpleNamespace(records=records), features)
        self.assertEqual(ds[0]["features"][0].item(), 1.0)
        self.assertEqual(ds[0]["target"][0].item(), 1.0)
        self.assertEqual(ds[1]["features"][0].item(), 2.0)
        self.assertEqual(ds[1]["target"][1].item(), 1.0)

    def test_feature_id_mismatch_is_rejected(self):
        records = [{"metadata": {"segment_id": "seg-a"}, "target": torch.zeros(18)}]
        with self.assertRaises(ValueError):
            FrozenPresenceDataset(SimpleNamespace(records=records), {"other": torch.zeros(768)})

    def test_head_output_dimension_and_finite_logits(self):
        model = FrozenXLMRPresenceHead()
        logits = model(torch.randn(4, EXPECTED_DIM))
        self.assertEqual(tuple(logits.shape), (4, 18))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(model.classifier.in_features, 768)
        self.assertEqual(model.classifier.out_features, 18)

    def test_actual_train_dev_cache_alignment_and_selective_row_read(self):
        train_features, dev_features, stats = load_train_dev_text_features(ROOT, CACHE, return_stats=True)
        train_source = ImplicitPresenceDataset(ROOT, "train")
        dev_source = ImplicitPresenceDataset(ROOT, "dev")
        assert_hotel_disjoint(train_source, dev_source)
        train_ds = FrozenPresenceDataset(train_source, train_features)
        dev_ds = FrozenPresenceDataset(dev_source, dev_features)
        self.assertEqual((len(train_ds), len(dev_ds)), (8816, 1895))
        self.assertEqual(stats["train_rows"], 8816)
        self.assertEqual(stats["dev_rows"], 1895)
        self.assertEqual(stats["feature_payload_bytes_read"], (8816 + 1895) * 768 * 4)
        self.assertEqual(stats["feature_payload_bytes_read"], stats["expected_selected_row_bytes"])
        self.assertEqual(stats["partial_storage_reads"], 1)
        self.assertFalse(set(train_features) & set(dev_features))
        by_id = {row["segment_id"]: row for row in train_ds.records}
        source_by_id = {row["metadata"]["segment_id"]: row for row in train_source.records}
        check_id = next(iter(source_by_id))
        self.assertTrue(torch.equal(by_id[check_id]["target"], source_by_id[check_id]["target"]))
        self.assertEqual(by_id[check_id]["feature"].shape, (768,))
        self.assertEqual(len(LABEL_ORDER), 18)

    def test_presence_metrics_and_dev_threshold(self):
        probs = np.zeros((2, 18), dtype=float)
        probs[0, 0] = 0.9
        probs[1, 1] = 0.9
        gold = np.zeros((2, 18), dtype=int)
        gold[0, 0] = 1
        metrics = evaluate_presence(probs, gold, 0.5)
        self.assertEqual(metrics["micro"]["tp"], 1)
        self.assertEqual(metrics["micro"]["fp"], 1)
        self.assertEqual(metrics["micro"]["fn"], 0)
        self.assertAlmostEqual(metrics["micro"]["precision"], 0.5)
        self.assertEqual(metrics["exact_set_accuracy"], 0.5)
        self.assertEqual(metrics["per_label"][LABEL_ORDER[0]]["support"], 1)
        selection = select_presence_threshold(probs, gold, [0.5, 0.95, 1.0])
        self.assertIn(selection["threshold"], [0.5, 0.95, 1.0])


if __name__ == "__main__":
    unittest.main()
