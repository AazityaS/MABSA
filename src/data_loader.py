"""Segment-level dataset preserving each quad and its own image associations."""

import json
from collections import defaultdict
from pathlib import Path

try:
    from torch.utils.data import Dataset
except ImportError:  # Allows schema inspection on machines without PyTorch installed.
    class Dataset:  # type: ignore[no-redef]
        pass

from .image_utils import load_rgb_image, preprocess_image, resolve_image_path
from .label_mapping import build_label_mappings


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class HAMOSDataset(Dataset):
    """One item per official split segment. Missing per-quad images stay explicit."""

    def __init__(self, root, split="train", *, split_view="single_image", load_images=True,
                 image_size=224, image_mean=(0.48145466, 0.4578275, 0.40821073),
                 image_std=(0.26862954, 0.26130258, 0.27577711), tokenizer=None,
                 max_length=256):
        self.root = Path(root).resolve()
        self.split, self.load_images = split, load_images
        self.image_size, self.image_mean, self.image_std = image_size, image_mean, image_std
        self.tokenizer, self.max_length = tokenizer, max_length
        self.label_mappings = build_label_mappings(self.root / "taxonomy/hotel_absa_taxonomy.xlsx")
        split_rows = read_jsonl(self.root / "splits" / split_view / f"{split}.jsonl")
        self.segments = {r["segment_id"]: r for r in read_jsonl(self.root / "metadata/segments.jsonl")}
        self.reviews = {r["source_review_id"]: r for r in read_jsonl(self.root / "metadata/reviews.jsonl")}
        self.images = {r["image_id"]: r for r in read_jsonl(self.root / "metadata/images.jsonl")}
        self.quads_by_segment = defaultdict(list)
        for quad in read_jsonl(self.root / "annotations/quads.jsonl"):
            self.quads_by_segment[quad["segment_id"]].append(quad)
        self.rows = split_rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        sid = row["segment_id"]
        segment = self.segments[sid]
        review = self.reviews[row["source_review_id"]]
        annotations, quad_images = [], []
        for quad in self.quads_by_segment[sid]:
            item = dict(quad)
            item["aspect_category_id"] = self.label_mappings["aspect_categories"][quad["aspect_category"]]
            item["taxonomy_code_id"] = self.label_mappings["taxonomy_codes"][quad["taxonomy_code"]]
            item["sentiment_id"] = self.label_mappings["sentiments"][quad["sentiment"]]
            item["aspect_implicit_id"] = self.label_mappings["implicitness"]["implicit" if quad["aspect_implicit"] else "explicit"]
            item["opinion_implicit_id"] = self.label_mappings["implicitness"]["implicit" if quad["opinion_implicit"] else "explicit"]
            image_id = quad["primary_image_id"]
            image_meta = self.images[image_id]
            path = resolve_image_path(self.root, image_meta.get("local_path"), image_id)
            item["primary_image_path"] = str(path)
            item["primary_image_available"] = path.is_file()
            supporting = []
            for support_id in quad.get("supporting_image_ids") or []:
                meta = self.images[support_id]
                support_path = resolve_image_path(self.root, meta.get("local_path"), support_id)
                supporting.append({"image_id": support_id, "path": str(support_path),
                                   "available": support_path.is_file()})
            item["supporting_images"] = supporting
            annotations.append(item)
            if self.load_images and path.is_file():
                try:
                    quad_images.append(preprocess_image(load_rgb_image(path), self.image_size,
                                                        self.image_mean, self.image_std))
                except Exception as exc:
                    raise OSError(f"Invalid primary image for segment {sid}, quad {quad['quad_id']}, "
                                  f"image {image_id}: {exc}") from exc
            else:
                quad_images.append(None)
        item = {"segment_id": sid, "source_review_id": row["source_review_id"],
                "hotel_id": review["hotel_id"], "text": row["text"],
                "language": segment["language"], "annotations": annotations,
                "quad_images": quad_images, "split": self.split}
        if self.tokenizer is not None:
            encoded = self.tokenizer(item["text"], truncation=True, max_length=self.max_length,
                                     return_tensors="pt")
            item["input_ids"] = encoded["input_ids"].squeeze(0)
            item["attention_mask"] = encoded["attention_mask"].squeeze(0)
        return item


def collate_segments(samples):
    """Collate variable quad counts; image_mask[b,q] marks usable primary images."""
    import torch
    from torch.nn.utils.rnn import pad_sequence

    batch = {"segment_id": [s["segment_id"] for s in samples],
             "source_review_id": [s["source_review_id"] for s in samples],
             "hotel_id": [s["hotel_id"] for s in samples],
             "text": [s["text"] for s in samples], "language": [s["language"] for s in samples],
             "annotations": [s["annotations"] for s in samples],
             "quad_counts": torch.tensor([len(s["annotations"]) for s in samples], dtype=torch.long)}
    max_quads = max((len(s["annotations"]) for s in samples), default=0)
    batch["image_mask"] = torch.zeros((len(samples), max_quads), dtype=torch.bool)
    batch["quad_mask"] = torch.zeros((len(samples), max_quads), dtype=torch.bool)
    image_shape = next((tuple(t.shape) for s in samples for t in s["quad_images"] if t is not None), None)
    if image_shape is not None:
        batch["images"] = torch.zeros((len(samples), max_quads, *image_shape), dtype=torch.float32)
        for b, sample in enumerate(samples):
            batch["quad_mask"][b, :len(sample["annotations"])] = True
            for q, tensor in enumerate(sample["quad_images"]):
                if tensor is not None:
                    batch["images"][b, q] = tensor
                    batch["image_mask"][b, q] = True
    else:
        for b, sample in enumerate(samples):
            batch["quad_mask"][b, :len(sample["annotations"])] = True
    if all("input_ids" in s for s in samples):
        batch["input_ids"] = pad_sequence([s["input_ids"] for s in samples], batch_first=True, padding_value=1)
        batch["attention_mask"] = pad_sequence([s["attention_mask"] for s in samples], batch_first=True, padding_value=0)
    return batch

