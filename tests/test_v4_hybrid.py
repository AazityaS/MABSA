"""V4 regression tests runnable with the standard-library unittest runner."""

import unittest
from pathlib import Path

import torch
from torch import nn

from src.v3_presence_dataset import CATEGORIES, LABEL_ORDER, NUM_LABELS, SENTIMENTS
from src.v4_hybrid_dataset import (HybridCandidateDataset, build_status_targets,
                                   summarize_targets, validate_split_pair,
                                   audit_repeated_text_targets)
from src.v4_hybrid_loss import hybrid_loss
from src.v4_hybrid_metrics import evaluate_hybrid, select_combined_threshold, _ranking
from src.v4_hybrid_model import HybridCandidateModel
from src.v4_hybrid_train import _rebase_scheduler
from transformers import get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": 8})()
        self.emb = nn.Embedding(32, 8)

    def forward(self, input_ids, attention_mask):
        return type("Out", (), {"last_hidden_state": self.emb(input_ids)})()


class V4HybridTests(unittest.TestCase):
    def test_candidate_order_and_dimensions(self):
        self.assertEqual(LABEL_ORDER, tuple(f"{c} / {s}" for c in CATEGORIES for s in SENTIMENTS))
        self.assertEqual(len(LABEL_ORDER), NUM_LABELS)
        self.assertEqual(NUM_LABELS, 18)

    def test_target_construction_presence_and_explicit_only(self):
        quads = [
            {"aspect_category": "AMENITY", "sentiment": "negative", "aspect_implicit": False},
            {"aspect_category": "AMENITY", "sentiment": "negative", "aspect_implicit": True},
            {"aspect_category": "SERVICE", "sentiment": "positive", "aspect_implicit": False},
            {"aspect_category": "SERVICE", "sentiment": "positive", "aspect_implicit": False},
        ]
        any_y, cond_y, mask = build_status_targets(quads)
        self.assertEqual(any_y.sum().item(), 2)
        self.assertEqual(cond_y.sum().item(), 1)
        self.assertEqual(mask.sum().item(), 2)
        self.assertEqual(cond_y[0].item(), 1.0)  # mixed explicit+implicit is positive
        idx = CATEGORIES.index("SERVICE") * 3 + SENTIMENTS.index("positive")
        self.assertEqual(any_y[idx].item(), 1.0)
        self.assertEqual(cond_y[idx].item(), 0.0)  # explicit-only
        self.assertTrue(mask[idx].item())
        self.assertEqual(cond_y[1].item(), 0.0)  # absent
        self.assertFalse(mask[1].item())

    def test_repeated_pairs_collapse_and_empty_segment_is_all_zero(self):
        q = {"aspect_category": "EXPERIENCE", "sentiment": "neutral", "aspect_implicit": True}
        any_y, cond_y, mask = build_status_targets([q, q, q])
        self.assertEqual(any_y.sum().item(), 1)
        self.assertEqual(cond_y.sum().item(), 1)
        self.assertEqual(mask.sum().item(), 1)
        self.assertTrue(all(t.sum().item() == 0 for t in build_status_targets([])))

    def test_masked_conditional_loss_and_zero_valid_batch(self):
        any_logits = torch.zeros(2, 18, requires_grad=True)
        cond_logits = torch.randn(2, 18, requires_grad=True)
        any_y = torch.zeros(2, 18)
        cond_y = torch.zeros(2, 18)
        valid = torch.zeros(2, 18, dtype=torch.bool)
        result = hybrid_loss(any_logits, cond_logits, any_y, cond_y, valid, 1.0)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(any_logits, any_y)
        self.assertTrue(torch.allclose(result["conditional_loss"], torch.tensor(0.0)))
        self.assertTrue(torch.allclose(result["loss"], expected))
        result["loss"].backward()
        self.assertIsNotNone(any_logits.grad)
        self.assertIsNotNone(cond_logits.grad)

    def test_conditional_loss_uses_only_valid_rows(self):
        logits = torch.tensor([[0.0, 10.0]], requires_grad=True)
        targets = torch.tensor([[1.0, 0.0]])
        mask = torch.tensor([[True, False]])
        result = hybrid_loss(torch.zeros_like(logits), logits, torch.zeros_like(targets), targets, mask)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(logits[:, :1], targets[:, :1])
        self.assertTrue(torch.allclose(result["conditional_loss"], expected))
        result["loss"].backward()
        self.assertEqual(logits.grad[0, 1].item(), 0.0)

    def test_candidate_conditioning_gradient_and_combined_probability(self):
        model = HybridCandidateModel(TinyEncoder(), hidden_size=12, category_embedding_dim=4,
                                     sentiment_embedding_dim=3, encoder_hidden_size=8)
        out = model(torch.randint(0, 32, (2, 5)), torch.ones(2, 5, dtype=torch.long))
        self.assertEqual(out["any_logits"].shape, (2, 18))
        self.assertTrue(torch.allclose(out["combined_probabilities"],
                                       out["any_probabilities"] * out["conditional_probabilities"]))
        (out["any_logits"].mean() + out["conditional_logits"].mean()).backward()
        self.assertIsNotNone(model.encoder.emb.weight.grad)
        self.assertIsNotNone(model.category_embeddings.weight.grad)
        self.assertIsNotNone(model.sentiment_embeddings.weight.grad)

    def test_threshold_sweep_and_metrics(self):
        targets = torch.zeros(2, 18).numpy()
        targets[0, 0] = 1
        probs = torch.zeros(2, 18).numpy()
        probs[0, 0] = .8
        probs[1, 1] = .2
        selected = select_combined_threshold(probs, targets)
        self.assertAlmostEqual(selected["threshold"], .8)
        self.assertEqual(selected["threshold_comparison"], ">=")
        mask = targets.astype(bool)
        metrics = evaluate_hybrid(probs, targets, probs, targets, probs, targets, mask, .8)
        micro = metrics["primary_implicit_pair_presence"]["micro"]
        self.assertEqual((micro["tp"], micro["fp"], micro["fn"]), (1, 0, 0))

    def test_threshold_sweep_ties_match_evaluated_counts(self):
        probs = torch.tensor([[.9] * 6 + [.8] * 6 + [.1] * 6]).numpy()
        targets = torch.tensor([[1, 1, 0, 0, 0, 0,
                                 1, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0]]).numpy()
        selected = select_combined_threshold(probs, targets)
        measured = evaluate_hybrid(probs, targets, probs, targets, probs, targets,
                                   targets.astype(bool), selected["threshold"])
        micro = measured["primary_implicit_pair_presence"]["micro"]
        self.assertEqual((micro["tp"], micro["fp"], micro["fn"]),
                         (selected["tp"], selected["fp"], selected["fn"]))
        self.assertAlmostEqual(micro["f1"], selected["micro_f1"])

    def test_ranking_metrics_with_ties(self):
        probs = torch.tensor([[.9, .1], [.9, .2]]).numpy()
        targets = torch.tensor([[1, 0], [0, 1]]).numpy()
        result = _ranking(probs, targets)
        self.assertAlmostEqual(result["roc_auc"], .625)
        self.assertAlmostEqual(result["average_precision"], 7 / 12)

    def test_continuation_scheduler_rebases_to_global_update(self):
        param = nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([param], lr=2e-5)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=4,
                                                     num_training_steps=20)
        lr = _rebase_scheduler(scheduler, optimizer, 3)
        self.assertEqual(scheduler.last_epoch, 3)
        self.assertAlmostEqual(lr, 1.5e-5, places=10)

    def test_official_train_dev_counts_and_hotel_separation(self):
        train = HybridCandidateDataset(ROOT, "train")
        dev = HybridCandidateDataset(ROOT, "dev")
        validate_split_pair(train, dev)
        expected_train = (8816, 158688, 11663, 147025, 11663, 451, 11212)
        expected_dev = (1895, 34110, 2501, 31609, 2501, 98, 2403)
        for dataset, expected in ((train, expected_train), (dev, expected_dev)):
            s = summarize_targets(dataset)
            actual = (s["segments"], s["candidate_rows"], s["any_positive_rows"],
                      s["any_negative_rows"], s["conditional_valid_rows"],
                      s["conditional_positive_rows"], s["conditional_negative_rows"])
            self.assertEqual(actual, expected)
        self.assertFalse(train.hotel_ids & dev.hotel_ids)
        item = train[0]
        self.assertEqual(set(item), {"text", "any_targets", "conditional_targets", "conditional_mask", "metadata"})
        self.assertNotIn("segment_id", item)
        self.assertNotIn("hotel_id", item)
        self.assertNotIn("aspect_term", item)
        self.assertNotIn("opinion_term", item)
        collision_audit = audit_repeated_text_targets(train, dev)
        self.assertEqual(collision_audit["conflicting_repeated_groups"], 0)

    def test_test_split_rejected(self):
        with self.assertRaises(ValueError):
            HybridCandidateDataset(ROOT, "test")


if __name__ == "__main__":
    unittest.main()
