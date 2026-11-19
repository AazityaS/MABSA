"""Quad-level views for supervised per-quad classification experiments."""

from collections import Counter
import hashlib
from pathlib import Path

try:
    from torch.utils.data import Dataset
except ImportError:  # Keep read-only schema tools usable without PyTorch.
    class Dataset:  # type: ignore[no-redef]
        pass

from .data_loader import HAMOSDataset
from .image_utils import load_rgb_image, preprocess_image, resolve_image_path


class QuadDataset(Dataset):
    """One instance per quad, with modality-specific image availability policy.

    ``mode='text'`` retains every quad. ``mode='multimodal'`` filters only
    quads whose own required primary image file is unavailable; excluded IDs
    are available through ``excluded_instances``.
    """

    def __init__(self, root, split="train", *, mode="multimodal", split_view="single_image",
                 image_size=224, tokenizer=None, max_length=256, include_taxonomy_target=True):
        if mode not in {"multimodal", "text"}:
            raise ValueError("mode must be 'multimodal' or 'text'")
        self.root = Path(root).resolve()
        self.split, self.mode = split, mode
        self.image_size, self.tokenizer, self.max_length = image_size, tokenizer, max_length
        self.include_taxonomy_target = include_taxonomy_target
        self.segment_dataset = HAMOSDataset(self.root, split, split_view=split_view, load_images=False,
                                            image_size=image_size, tokenizer=None)
        self.instances, self.excluded_instances = [], []
        self.label_mappings = self.segment_dataset.label_mappings
        for row in self.segment_dataset.rows:
            sid = row["segment_id"]
            segment = self.segment_dataset.segments[sid]
            review = self.segment_dataset.reviews[row["source_review_id"]]
            quads = self.segment_dataset.quads_by_segment[sid]
            for quad in quads:
                image_id = quad["primary_image_id"]
                image_meta = self.segment_dataset.images[image_id]
                image_path = resolve_image_path(self.root, image_meta.get("local_path"), image_id)
                available = image_path.is_file()
                record = (row, segment, review, quad, image_path, available, len(quads))
                if mode == "multimodal" and not available:
                    self.excluded_instances.append({"quad_id": quad["quad_id"],
                                                    "segment_id": sid,
                                                    "primary_image_id": image_id,
                                                    "reason": "primary image unavailable"})
                else:
                    self.instances.append(record)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        row, segment, review, quad, image_path, available, segment_quad_count = self.instances[index]
        image = None
        if self.mode == "multimodal":
            if not available:
                raise RuntimeError(f"Filtered multimodal instance has missing image: {quad['quad_id']}")
            image = preprocess_image(load_rgb_image(image_path), image_size=self.image_size)
        targets = {
            "aspect_category": self.label_mappings["aspect_categories"][quad["aspect_category"]],
            "sentiment": self.label_mappings["sentiments"][quad["sentiment"]],
            "aspect_implicit": self.label_mappings["implicitness"]["implicit" if quad["aspect_implicit"] else "explicit"],
            "opinion_implicit": self.label_mappings["implicitness"]["implicit" if quad["opinion_implicit"] else "explicit"],
        }
        target_labels = {"aspect_category": quad["aspect_category"], "sentiment": quad["sentiment"]}
        if self.include_taxonomy_target:
            targets["taxonomy_code"] = self.label_mappings["taxonomy_codes"][quad["taxonomy_code"]]
            target_labels["taxonomy_code"] = quad["taxonomy_code"]
        metadata = {
            "quad_id": quad["quad_id"], "segment_id": row["segment_id"],
            "source_review_id": row["source_review_id"], "hotel_id": review["hotel_id"],
            "primary_image_id": quad["primary_image_id"], "primary_image_path": str(image_path),
            "primary_image_available": available,
            "supporting_image_ids": list(quad.get("supporting_image_ids") or []),
            "segment_quad_count": segment_quad_count,
            "segment_has_multiple_quads": segment_quad_count > 1,
        }
        result = {
            "inputs": {"text": row["text"], "image": image, "language": segment["language"]},
            "targets": targets,
            "target_labels": target_labels,
            "metadata": metadata,
        }
        if self.tokenizer is not None:
            encoded = self.tokenizer(row["text"], truncation=True, max_length=self.max_length,
                                     return_tensors="pt")
            result["inputs"]["input_ids"] = encoded["input_ids"].squeeze(0)
            result["inputs"]["attention_mask"] = encoded["attention_mask"].squeeze(0)
        return result


def collate_quad_instances(samples, *, tokenizer=None, max_length=256):
    """Stack model inputs/targets while keeping gold text terms out of inputs."""
    import torch
    from torch.nn.utils.rnn import pad_sequence

    inputs = {"text": [s["inputs"]["text"] for s in samples],
              "language": [s["inputs"]["language"] for s in samples]}
    available_images = [s["inputs"]["image"] for s in samples]
    inputs["image_mask"] = torch.tensor([image is not None for image in available_images], dtype=torch.bool)
    if any(image is not None for image in available_images):
        shape = next(tuple(image.shape) for image in available_images if image is not None)
        inputs["images"] = torch.stack([image if image is not None else torch.zeros(shape, dtype=torch.float32)
                                       for image in available_images])
    if tokenizer is not None:
        encoded = tokenizer(inputs["text"], padding=True, truncation=True, max_length=max_length,
                            return_tensors="pt")
        inputs["input_ids"], inputs["attention_mask"] = encoded["input_ids"], encoded["attention_mask"]
    elif all("input_ids" in s["inputs"] for s in samples):
        inputs["input_ids"] = pad_sequence([s["inputs"]["input_ids"] for s in samples],
                                           batch_first=True, padding_value=1)
        inputs["attention_mask"] = pad_sequence([s["inputs"]["attention_mask"] for s in samples],
                                                batch_first=True, padding_value=0)
    target_names = list(samples[0]["targets"])
    targets = {name: torch.tensor([s["targets"][name] for s in samples], dtype=torch.long)
               for name in target_names}
    return {"inputs": inputs, "targets": targets,
            "target_labels": [s["target_labels"] for s in samples],
            "metadata": [s["metadata"] for s in samples]}


def summarize_quad_datasets(root, *, split_view="single_image"):
    """Compute counts and target distributions without opening images into RAM."""
    root = Path(root).resolve()
    def sha256(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    result = {"splits": {}, "hotel_overlap_counts": {}, "all_quads_preserved_in_text_view": True,
              "source_fingerprints_sha256": {"annotations/quads.jsonl": sha256(root / "annotations/quads.jsonl"),
                  **{f"splits/{split_view}/{s}.jsonl": sha256(root / "splits" / split_view / f"{s}.jsonl")
                     for s in ("train", "dev", "test")}}}
    hotels_by_split = {}
    category_labels = ["FACILITY", "AMENITY", "SERVICE", "EXPERIENCE", "LOYALTY", "BRANDING"]
    sentiment_labels = ["negative", "neutral", "positive"]
    implicit_labels = ["explicit", "implicit"]
    for split in ("train", "dev", "test"):
        text_ds = QuadDataset(root, split, mode="text", split_view=split_view)
        multi_ds = QuadDataset(root, split, mode="multimodal", split_view=split_view)
        hotels = {str(record[2]["hotel_id"]) for record in text_ds.instances}
        hotels_by_split[split] = hotels

        def distribution(dataset, key, labels):
            counter = Counter()
            for _, _, _, quad, _, _, _ in dataset.instances:
                if key == "aspect_category":
                    value = quad["aspect_category"]
                elif key == "sentiment":
                    value = quad["sentiment"]
                elif key == "taxonomy_code":
                    value = quad["taxonomy_code"]
                elif key in ("aspect_implicit", "opinion_implicit"):
                    value = "implicit" if quad[key] else "explicit"
                counter[value] += 1
            return {label: counter.get(label, 0) for label in labels}

        quad_count_by_segment = Counter(record[0]["segment_id"] for record in text_ds.instances)
        multi_quad_segment_ids = {sid for sid, count in quad_count_by_segment.items() if count > 1}
        multi_quad_instance_count = sum(record[0]["segment_id"] in multi_quad_segment_ids
                                        for record in text_ds.instances)
        result["splits"][split] = {
            "quad_instances": len(text_ds),
            "multimodal_usable_instances": len(multi_ds),
            "text_only_instances": len(text_ds),
            "missing_primary_image_excluded_from_multimodal": len(multi_ds.excluded_instances),
            "excluded_quad_ids": [r["quad_id"] for r in multi_ds.excluded_instances],
            "segments_with_multiple_quads": len(multi_quad_segment_ids),
            "instances_from_multi_quad_segments": multi_quad_instance_count,
            "instances_with_implicit_aspect": sum(record[3]["aspect_implicit"] for record in text_ds.instances),
            "multimodal_instances_with_implicit_aspect": sum(record[3]["aspect_implicit"] for record in multi_ds.instances),
            "text_only_label_distributions": {
                "aspect_category": distribution(text_ds, "aspect_category", category_labels),
                "sentiment": distribution(text_ds, "sentiment", sentiment_labels),
                "aspect_implicit": distribution(text_ds, "aspect_implicit", implicit_labels),
                "opinion_implicit": distribution(text_ds, "opinion_implicit", implicit_labels),
                "taxonomy_code": distribution(text_ds, "taxonomy_code", sorted(text_ds.label_mappings["taxonomy_codes"])),
            },
            "multimodal_label_distributions": {
                "aspect_category": distribution(multi_ds, "aspect_category", category_labels),
                "sentiment": distribution(multi_ds, "sentiment", sentiment_labels),
                "aspect_implicit": distribution(multi_ds, "aspect_implicit", implicit_labels),
                "opinion_implicit": distribution(multi_ds, "opinion_implicit", implicit_labels),
                "taxonomy_code": distribution(multi_ds, "taxonomy_code", sorted(multi_ds.label_mappings["taxonomy_codes"])),
            },
        }
        expected_split_quads = sum(len(row.get("quads", [])) for row in text_ds.segment_dataset.rows)
        result["all_quads_preserved_in_text_view"] &= len(text_ds) == expected_split_quads
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        result["hotel_overlap_counts"][f"{left}_{right}"] = len(hotels_by_split[left] & hotels_by_split[right])
    result["no_hotel_leakage"] = not any(result["hotel_overlap_counts"].values())
    return result


def write_quad_dataset_summary(root, summary=None):
    import json
    root = Path(root).resolve()
    summary = summary or summarize_quad_datasets(root)
    out = root / "reports/quad_dataset_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return out

