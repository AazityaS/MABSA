import hashlib
import json
import unittest
from collections import defaultdict
from pathlib import Path

from src.structured_dataset import StructuredDataset, collate_structured


ROOT = Path(__file__).resolve().parents[1]


class TestStructuredDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.datasets = {(split, mode): StructuredDataset(ROOT, split, mode=mode,
                           load_images=False, verify_images=False)
                        for split in ("train", "dev", "test") for mode in ("text", "multimodal")}

    def test_split_counts_and_primary_eligibility(self):
        expected = {"train": (8813, 3), "dev": (1895, 0), "test": (1890, 0)}
        for split, (length, exclusions) in expected.items():
            text = self.datasets[(split, "text")]
            multi = self.datasets[(split, "multimodal")]
            self.assertEqual(len(text), length)
            self.assertEqual(len(multi), length)
            self.assertEqual(len(text.excluded_instances), exclusions)
            self.assertEqual(len(multi.excluded_instances), exclusions)
            self.assertEqual([r["segment_id"] for r in text.instances],
                             [r["segment_id"] for r in multi.instances])

    def test_future_full_coverage_text_view_is_available(self):
        ds = StructuredDataset(ROOT, "train", mode="text", load_images=False,
                               verify_images=False, require_candidate_images=False)
        self.assertEqual(len(ds), 8816)
        self.assertEqual(len(ds.unavailable_instances), 3)
        self.assertEqual(sum(len(ds[i]["targets"]) for i in range(len(ds))), 16789)

    def test_quad_totals_and_no_hotel_leakage(self):
        expected = {"train": 16784, "dev": 3600, "test": 3606}
        hotels = {}
        for split in expected:
            ds = self.datasets[(split, "text")]
            self.assertEqual(sum(len(r["targets"]) for r in (ds[i] for i in range(len(ds)))), expected[split])
            hotels[split] = {r["hotel_id"] for r in ds.instances}
        self.assertFalse(hotels["train"] & hotels["dev"])
        self.assertFalse(hotels["train"] & hotels["test"])
        self.assertFalse(hotels["dev"] & hotels["test"])

    def test_candidate_pools_are_metadata_derived_and_paths_exist(self):
        images = [json.loads(x) for x in (ROOT / "metadata/images.jsonl").read_text(encoding="utf-8").splitlines()]
        by_review = defaultdict(set)
        by_id = {}
        for image in images:
            by_id[image["image_id"]] = image
            path = Path(image.get("local_path") or f"images/{image['image_id']}.jpg")
            if not path.is_absolute(): path = ROOT / path
            if path.is_file(): by_review[image["source_review_id"]].add(image["image_id"])
        for split in ("train", "dev", "test"):
            ds = self.datasets[(split, "multimodal")]
            for row in ds.instances:
                actual = {x["image_id"] for x in row["candidate_images"]}
                self.assertEqual(actual, by_review[row["source_review_id"]])
                self.assertTrue(all(x["path"].is_file() for x in row["candidate_images"]))
                self.assertTrue(all(by_id[i]["source_review_id"] == row["source_review_id"] for i in actual))

    def test_model_input_boundary_and_target_preservation(self):
        ds = self.datasets[("train", "multimodal")]
        sample = ds[0]
        self.assertEqual(set(sample["inputs"]), {"text", "language"})
        self.assertEqual(set(sample["metadata"]), {"segment_id", "source_review_id", "hotel_id",
                                                   "candidate_image_ids", "quad_count"})
        self.assertTrue(all(set(t) == {"aspect_category", "sentiment", "aspect_implicit", "opinion_implicit"}
                            for t in sample["targets"]))
        self.assertEqual(sample["metadata"]["quad_count"], len(sample["targets"]))
        self.assertNotIn("aspect_term", sample["inputs"])
        self.assertNotIn("primary_image_id", sample["inputs"])

    def test_batch_retains_multiple_candidate_photos_and_all_quad_targets(self):
        ds = StructuredDataset(ROOT, "test", mode="multimodal", load_images=True,
                               verify_images=False)
        index = next(i for i, row in enumerate(ds.instances)
                     if len(row["candidate_images"]) > 1 and len(row["quads"]) > 1)
        single_index = next(i for i, row in enumerate(ds.instances)
                            if len(row["candidate_images"]) == 1 and len(row["quads"]) == 1)
        sample = ds[index]
        single_sample = ds[single_index]
        batch = collate_structured([sample, single_sample])
        self.assertEqual(batch["inputs"]["candidate_images"].shape[1], len(ds.instances[index]["candidate_images"]))
        self.assertEqual(batch["inputs"]["image_mask"].sum(dim=1).tolist(),
                         [len(ds.instances[index]["candidate_images"]), 1])
        self.assertEqual(batch["targets"]["mask"].sum(dim=1).tolist(),
                         [len(sample["targets"]), len(single_sample["targets"])])
        self.assertEqual(batch["metadata"][0]["quad_count"], len(sample["targets"]))

    def test_source_fingerprints_and_official_split_files_unchanged(self):
        summary = json.loads((ROOT / "reports/quad_dataset_summary.json").read_text(encoding="utf-8"))
        for relpath, expected in summary["source_fingerprints_sha256"].items():
            digest = hashlib.sha256((ROOT / relpath).read_bytes()).hexdigest()
            self.assertEqual(digest, expected, relpath)


if __name__ == "__main__": unittest.main()
