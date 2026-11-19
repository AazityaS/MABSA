import unittest
from pathlib import Path

from src.label_mapping import build_label_mappings
from src.structured_metrics import CATEGORY_NAMES, SENTIMENT_NAMES


class TestEvaluationLabelMappings(unittest.TestCase):
    def test_metric_names_match_taxonomy_category_indices(self):
        root = Path(__file__).resolve().parents[1]
        mappings = build_label_mappings(root / "taxonomy" / "hotel_absa_taxonomy.xlsx")
        category_names_by_index = [
            name for name, _ in sorted(mappings["aspect_categories"].items(), key=lambda pair: pair[1])
        ]
        self.assertEqual(category_names_by_index, CATEGORY_NAMES)

    def test_metric_names_match_dataset_sentiment_indices(self):
        root = Path(__file__).resolve().parents[1]
        mappings = build_label_mappings(root / "taxonomy" / "hotel_absa_taxonomy.xlsx")
        sentiment_names_by_index = [
            name for name, _ in sorted(mappings["sentiments"].items(), key=lambda pair: pair[1])
        ]
        self.assertEqual(sentiment_names_by_index, SENTIMENT_NAMES)


if __name__ == "__main__":
    unittest.main()
