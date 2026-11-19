"""Segment-level set-prediction dataset with review-derived photo pools."""

from __future__ import annotations

import json
from collections import OrderedDict, defaultdict
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from .data_loader import read_jsonl
from .image_utils import load_rgb_image, preprocess_image, resolve_image_path
from .label_mapping import build_label_mappings


class StructuredDataset(Dataset):
    """One segment per item; candidate photos are joined by review metadata only."""

    def __init__(self, root, split="train", *, mode="multimodal", split_view="single_image",
                 image_size=224, max_slots=23, load_images=True, image_transform=None,
                 verify_images=True, require_candidate_images=True):
        if mode not in {"text", "multimodal"}:
            raise ValueError("mode must be 'text' or 'multimodal'")
        if split not in {"train", "dev", "test"}:
            raise ValueError(f"Unknown split: {split}")
        self.root = Path(root).resolve()
        self.split, self.mode, self.max_slots = split, mode, int(max_slots)
        if mode == "multimodal" and not require_candidate_images:
            raise ValueError("multimodal mode requires an available candidate image")
        self.require_candidate_images = require_candidate_images
        self.image_size, self.load_images = image_size, load_images
        self.image_transform = image_transform
        self.label_mappings = build_label_mappings(self.root / "taxonomy/hotel_absa_taxonomy.xlsx")
        self._images = OrderedDict()
        self._verified_image_ids = set()

        rows = read_jsonl(self.root / "splits" / split_view / f"{split}.jsonl")
        segments = {r["segment_id"]: r for r in read_jsonl(self.root / "metadata/segments.jsonl")}
        reviews = {r["source_review_id"]: r for r in read_jsonl(self.root / "metadata/reviews.jsonl")}
        image_meta = {r["image_id"]: r for r in read_jsonl(self.root / "metadata/images.jsonl")}
        quads = defaultdict(list)
        for quad in read_jsonl(self.root / "annotations/quads.jsonl"):
            quads[quad["segment_id"]].append(quad)
        images_by_review = defaultdict(list)
        for img in image_meta.values():
            images_by_review[img["source_review_id"]].append(img)

        self.instances, self.excluded_instances, self.unavailable_instances = [], [], []
        for row in rows:
            sid, rid = row["segment_id"], row["source_review_id"]
            segment, review = segments[sid], reviews[rid]
            targets = quads[sid]
            if len(targets) > self.max_slots:
                raise AssertionError(f"Target set exceeds {self.max_slots} slots: {sid} has {len(targets)}")
            # Candidate IDs are derived from metadata/image records, never from quad fields.
            candidates = []
            for meta in images_by_review[rid]:
                path = resolve_image_path(self.root, meta.get("local_path"), meta["image_id"])
                if not path.is_file():
                    continue
                if verify_images and meta["image_id"] not in self._verified_image_ids:
                    try:
                        with Image.open(path) as im:
                            im.verify()
                    except Exception as exc:
                        raise OSError(f"Candidate image is unreadable: {meta['image_id']} ({path})") from exc
                    self._verified_image_ids.add(meta["image_id"])
                candidates.append({"image_id": meta["image_id"], "path": path})
            candidates.sort(key=lambda x: x["image_id"])
            if not candidates:
                exclusion = {"segment_id": sid, "source_review_id": rid,
                             "reason": "review has no accessible candidate image"}
                self.unavailable_instances.append(exclusion)
                if self.require_candidate_images:
                    self.excluded_instances.append(exclusion)
                    continue
            self.instances.append({
                "segment_id": sid, "source_review_id": rid, "hotel_id": str(review["hotel_id"]),
                "text": row["text"], "language": segment["language"],
                "candidate_images": candidates, "quads": targets,
            })

        expected = len(rows) - len(self.excluded_instances)
        assert len(self.instances) == expected
        # Both primary modes have the same eligibility; mode changes only image tensor loading.

    def __len__(self):
        return len(self.instances)

    def _load_image(self, image_record):
        image_id = image_record["image_id"]
        if image_id not in self._images:
            image = load_rgb_image(image_record["path"])
            tensor = (self.image_transform(image) if self.image_transform else
                      preprocess_image(image, image_size=self.image_size))
            self._images[image_id] = tensor
            if len(self._images) > 16:
                self._images.popitem(last=False)
        else:
            self._images.move_to_end(image_id)
        return self._images[image_id]

    def __getitem__(self, index):
        row = self.instances[index]
        # Explicit input/target boundary: no gold term, label or quad image association in inputs.
        inputs = {"text": row["text"], "language": row["language"]}
        if self.mode == "multimodal" and self.load_images:
            inputs["candidate_images"] = [self._load_image(x) for x in row["candidate_images"]]
        assert set(inputs).issubset({"text", "language", "candidate_images"})
        assert not (set(inputs) & {"aspect_term", "opinion_term", "aspect_category", "taxonomy_code",
                                   "sentiment", "aspect_implicit", "opinion_implicit",
                                   "primary_image_id", "supporting_image_ids", "segment_id",
                                   "source_review_id", "hotel_id"})
        target_rows = []
        for q in row["quads"]:
            target_rows.append({
                "aspect_category": self.label_mappings["aspect_categories"][q["aspect_category"]],
                "sentiment": self.label_mappings["sentiments"][q["sentiment"]],
                "aspect_implicit": int(bool(q["aspect_implicit"])),
                "opinion_implicit": int(bool(q["opinion_implicit"])),
            })
        assert len(target_rows) <= self.max_slots
        return {"inputs": inputs, "targets": target_rows,
                "metadata": {"segment_id": row["segment_id"],
                             "source_review_id": row["source_review_id"],
                             "hotel_id": row["hotel_id"],
                             "candidate_image_ids": [x["image_id"] for x in row["candidate_images"]],
                             "quad_count": len(target_rows)}}


def collate_structured(samples, tokenizer=None, max_length=256):
    """Collate variable target/photo sets; metadata is kept outside model inputs."""
    import torch

    batch_size = len(samples)
    max_targets = max(len(s["targets"]) for s in samples)
    max_images = max((len(s["inputs"].get("candidate_images", [])) for s in samples), default=0)
    inputs = {"text": [s["inputs"]["text"] for s in samples],
              "language": [s["inputs"]["language"] for s in samples]}
    if tokenizer is not None:
        encoded = tokenizer(inputs["text"], padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt")
        inputs["input_ids"] = encoded["input_ids"]
        inputs["attention_mask"] = encoded["attention_mask"]
    if max_images:
        shape = next(tuple(im.shape) for s in samples for im in s["inputs"].get("candidate_images", []))
        image_tensor = torch.zeros((batch_size, max_images, *shape), dtype=torch.float32)
        image_mask = torch.zeros((batch_size, max_images), dtype=torch.bool)
        for i, sample in enumerate(samples):
            for j, image in enumerate(sample["inputs"].get("candidate_images", [])):
                image_tensor[i, j] = image
                image_mask[i, j] = True
        inputs["candidate_images"] = image_tensor
        inputs["image_mask"] = image_mask
    targets = {"category": torch.zeros((batch_size, max_targets), dtype=torch.long),
               "sentiment": torch.zeros((batch_size, max_targets), dtype=torch.long),
               "aspect_implicit": torch.zeros((batch_size, max_targets), dtype=torch.float32),
               "opinion_implicit": torch.zeros((batch_size, max_targets), dtype=torch.float32),
               "mask": torch.zeros((batch_size, max_targets), dtype=torch.bool)}
    for i, sample in enumerate(samples):
        for j, target in enumerate(sample["targets"]):
            targets["category"][i, j] = target["aspect_category"]
            for key in ("sentiment", "aspect_implicit", "opinion_implicit"):
                targets[key][i, j] = target[key]
            targets["mask"][i, j] = True
    return {"inputs": inputs, "targets": targets,
            "metadata": [s["metadata"] for s in samples]}
