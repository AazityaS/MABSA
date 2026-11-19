"""Selective reader and validation for train/dev rows in the structured text cache.

The cache is a torch ZIP archive containing a dictionary of tensor views. A normal
torch.load would materialize the entire dictionary, including test feature values.
This module reads only the uncompressed byte range for each requested train/dev
vector; it never loads the test split or test feature bytes.
"""

from __future__ import annotations

import collections
import io
import json
import pickle
import struct
import zipfile
from pathlib import Path

import torch

from .v3_presence_dataset import TEXT_CHECKPOINT, TEXT_REVISION

EXPECTED_DIM = 768


class _StorageRef:
    def __init__(self, key: str, size: int, storage_type):
        self.key, self.size, self.storage_type = key, size, storage_type


class _TensorRef:
    def __init__(self, storage, offset, size, stride):
        self.storage = storage
        self.offset = int(offset)
        self.size = tuple(int(x) for x in size)
        self.stride = tuple(int(x) for x in stride)


class _MetadataUnpickler(pickle.Unpickler):
    """Decode cache tensor indexes only; replace tensor rebuild with descriptors."""

    def persistent_load(self, pid):
        if len(pid) < 5 or pid[0] != "storage":
            raise pickle.UnpicklingError("Unexpected persistent object in feature cache")
        return _StorageRef(str(pid[2]), int(pid[4]), pid[1])

    def find_class(self, module, name):
        if module == "torch._utils" and name.startswith("_rebuild_tensor"):
            return lambda storage, offset, size, stride, *args: _TensorRef(storage, offset, size, stride)
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if module == "torch" and name.endswith("Storage"):
            return getattr(torch, name)
        return super().find_class(module, name)


def _official_split_ids(root: Path, split: str) -> list[str]:
    if split not in {"train", "dev"}:
        raise ValueError("V3-Frozen cache access permits only train/dev IDs")
    path = root / "splits" / "single_image" / f"{split}.jsonl"
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line)["segment_id"] for line in stream]


def _read_raw_member_slice(archive_path: Path, info: zipfile.ZipInfo, offset: int, length: int) -> bytes:
    if info.compress_type != zipfile.ZIP_STORED:
        raise ValueError("Selective feature reads require uncompressed torch storage members")
    with archive_path.open("rb") as source:
        source.seek(info.header_offset)
        header = source.read(30)
        if len(header) != 30:
            raise IOError("Incomplete ZIP local file header")
        (signature, _, _, _, _, _, _, _, _, name_len, extra_len) = struct.unpack("<IHHHHHIIIHH", header)
        if signature != 0x04034B50:
            raise IOError("Invalid ZIP local file header")
        data_offset = info.header_offset + 30 + name_len + extra_len
        if offset < 0 or length < 0 or offset + length > info.file_size:
            raise ValueError("Requested feature byte range is outside its storage")
        source.seek(data_offset + offset)
        raw = source.read(length)
        if len(raw) != length:
            raise IOError("Short read while reading a train/dev feature row")
        return raw


def load_train_dev_text_features(root: str | Path, cache_dir: str | Path, *, return_stats: bool = False):
    """Validate provenance and materialize train/dev vectors only."""
    root, cache_dir = Path(root).resolve(), Path(cache_dir).resolve()
    manifest_path = cache_dir / "manifest.json"
    tensor_path = cache_dir / "text_features.pt"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("text_checkpoint") != TEXT_CHECKPOINT:
        raise ValueError("Text cache checkpoint differs from the pinned V3 encoder")
    if manifest.get("text_revision_requested") != TEXT_REVISION or manifest.get("text_revision_resolved") != TEXT_REVISION:
        raise ValueError("Text cache revision differs from the pinned V3 revision")
    if int(manifest.get("text_dim", -1)) != EXPECTED_DIM:
        raise ValueError(f"Expected {EXPECTED_DIM}-dim text features, found {manifest.get('text_dim')}")

    train_ids = _official_split_ids(root, "train")
    dev_ids = _official_split_ids(root, "dev")
    if len(train_ids) != 8816 or len(dev_ids) != 1895:
        raise ValueError(f"Unexpected V3 split sizes: train={len(train_ids)}, dev={len(dev_ids)}")
    if len(set(train_ids)) != len(train_ids) or len(set(dev_ids)) != len(dev_ids):
        raise ValueError("Duplicate segment ID in official train/dev split")
    train_set, dev_set = set(train_ids), set(dev_ids)
    if train_set & dev_set:
        raise ValueError("Train/dev segment IDs overlap")
    selected = train_set | dev_set
    manifest_ids = manifest.get("segment_ids", [])
    if len(selected & set(manifest_ids)) != len(selected):
        raise ValueError("One or more train/dev segment IDs are absent from cache manifest")

    bytes_read = 0
    members_touched = set()
    with zipfile.ZipFile(tensor_path) as archive:
        names = archive.namelist()
        if not names or not names[0].endswith("/data.pkl"):
            raise ValueError("Unexpected torch text-feature archive structure")
        prefix = names[0].split("/", 1)[0]
        # data.pkl is tensor-index metadata (IDs, storage refs, offsets), not feature values.
        index = _MetadataUnpickler(io.BytesIO(archive.read(f"{prefix}/data.pkl"))).load()
        selected_keys = [sid for sid in index.keys() if sid in selected]
        if len(selected_keys) != len(selected) or set(selected_keys) != selected:
            raise ValueError("Cache does not contain exactly one key for each train/dev ID")
        result = {}
        for sid in selected_keys:
            ref = index[sid]
            if not isinstance(ref, _TensorRef) or ref.size != (EXPECTED_DIM,) or ref.stride != (1,):
                raise ValueError(f"Invalid cached tensor metadata for a selected segment ({sid})")
            if getattr(ref.storage.storage_type, "__name__", "") != "FloatStorage":
                raise ValueError("Expected float32 cached XLM-R embeddings")
            if ref.offset + EXPECTED_DIM > ref.storage.size:
                raise ValueError("Cached embedding view exceeds its backing storage")
            member = f"{prefix}/data/{ref.storage.key}"
            info = archive.getinfo(member)
            row_bytes = EXPECTED_DIM * 4
            raw = _read_raw_member_slice(tensor_path, info, ref.offset * 4, row_bytes)
            bytes_read += len(raw)
            members_touched.add((ref.storage.key, info.file_size))
            tensor = torch.frombuffer(bytearray(raw), dtype=torch.float32).clone()
            if tensor.shape != (EXPECTED_DIM,) or not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"Non-finite or malformed cached embedding for selected segment ({sid})")
            result[sid] = tensor

    if len(result) != len(selected):
        raise AssertionError("Selected train/dev feature count changed during load")
    train_features = {sid: result[sid] for sid in train_ids}
    dev_features = {sid: result[sid] for sid in dev_ids}
    stats = {"train_rows": len(train_features), "dev_rows": len(dev_features),
             "feature_payload_bytes_read": bytes_read,
             "expected_selected_row_bytes": len(selected) * EXPECTED_DIM * 4,
             "storage_members_touched": len(members_touched),
             "partial_storage_reads": sum(1 for key, size in members_touched
                                           if sum(1 for sid in selected_keys
                                                  if index[sid].storage.key == key) * EXPECTED_DIM * 4 < size)}
    if bytes_read != stats["expected_selected_row_bytes"]:
        raise AssertionError("Selective cache reader read a byte count inconsistent with selected rows")
    return (train_features, dev_features, stats) if return_stats else (train_features, dev_features)
