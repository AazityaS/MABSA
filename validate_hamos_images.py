"""
HAMOS image and reference validation
------------------------------------
Place this script in your HAMOS project root, alongside:
    download_report.csv
    images/
    metadata/images.jsonl              (optional but recommended)
    splits/single_image/{train,dev,test}.jsonl  (optional)
    splits/multi_image/{train,dev,test}.jsonl   (optional)

Install dependencies:
    python -m pip install pandas pillow

Run:
    python validate_hamos_images.py

Outputs:
    image_validation_report.csv
    duplicate_images.csv
    reference_validation_report.csv
    validation_summary.txt
"""

from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import json
import re
import sys

import pandas as pd
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
DOWNLOAD_REPORT = ROOT / "download_report.csv"
IMAGE_ID_PATTERN = re.compile(r"IMG\d{8}")

def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()

def resolve_report_path(raw_path):
    """Resolve paths written by the downloader, including Windows separators."""
    if pd.isna(raw_path) or not str(raw_path).strip():
        return None
    value = str(raw_path).strip().replace("\\", "/")
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()

def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[WARNING] Invalid JSON in {path} at line {line_no}: {exc}")
    return rows

def collect_image_ids(obj, found):
    """
    Collect image IDs only from fields whose names contain 'image'.
    Handles scalar IDs and lists such as candidate_image_ids.
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            if "image" in str(key).lower():
                if isinstance(value, str):
                    found.update(IMAGE_ID_PATTERN.findall(value))
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str):
                            found.update(IMAGE_ID_PATTERN.findall(item))
                        elif isinstance(item, (dict, list)):
                            collect_image_ids(item, found)
                elif isinstance(value, dict):
                    collect_image_ids(value, found)
            elif isinstance(value, (dict, list)):
                collect_image_ids(value, found)
    elif isinstance(obj, list):
        for item in obj:
            collect_image_ids(item, found)

def main():
    if not DOWNLOAD_REPORT.exists():
        sys.exit(
            f"Could not find {DOWNLOAD_REPORT.name}. "
            "Put this script in the same folder as download_report.csv."
        )

    df = pd.read_csv(DOWNLOAD_REPORT)
    required = {"image_id", "status", "path"}
    missing_columns = required - set(df.columns)
    if missing_columns:
        sys.exit(f"download_report.csv is missing columns: {sorted(missing_columns)}")

    print(f"Project folder: {ROOT}")
    print(f"Rows in download report: {len(df):,}")
    print("Checking downloaded files. This may take a little while...")

    file_rows = []
    hashes = defaultdict(list)
    status_counts = Counter()

    for _, row in df.iterrows():
        image_id = str(row["image_id"]).strip()
        status = str(row["status"]).strip().lower()
        status_counts[status] += 1

        result = {
            "image_id": image_id,
            "download_status": status,
            "path": str(row.get("path", "")),
            "file_exists": False,
            "image_readable": False,
            "width": None,
            "height": None,
            "format": None,
            "size_bytes": None,
            "sha256": None,
            "validation_status": "",
            "error": "",
        }

        if status != "downloaded":
            result["validation_status"] = "download_failed"
            result["error"] = str(row.get("error", "")) if not pd.isna(row.get("error", "")) else ""
            file_rows.append(result)
            continue

        path = resolve_report_path(row["path"])
        if path is None or not path.exists():
            result["validation_status"] = "file_missing"
            result["error"] = "File does not exist at the path recorded in download_report.csv"
            file_rows.append(result)
            continue

        result["file_exists"] = True
        result["size_bytes"] = path.stat().st_size

        try:
            # verify() checks file integrity; reopen to read dimensions/format.
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                im.load()
                result["width"], result["height"] = im.size
                result["format"] = im.format
            result["image_readable"] = True
            result["sha256"] = sha256_file(path)
            result["validation_status"] = "valid"
            hashes[result["sha256"]].append(image_id)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            result["validation_status"] = "corrupt_or_unsupported"
            result["error"] = str(exc)

        file_rows.append(result)

    image_report = pd.DataFrame(file_rows)
    image_report.to_csv(ROOT / "image_validation_report.csv", index=False)

    # Exact duplicate detection: same SHA-256 means byte-for-byte identical files.
    duplicate_rows = []
    duplicate_group = 0
    for digest, ids in hashes.items():
        if len(ids) > 1:
            duplicate_group += 1
            for image_id in ids:
                duplicate_rows.append({
                    "duplicate_group": duplicate_group,
                    "sha256": digest,
                    "image_id": image_id,
                })

    duplicates = pd.DataFrame(
        duplicate_rows,
        columns=["duplicate_group", "sha256", "image_id"]
    )
    duplicates.to_csv(ROOT / "duplicate_images.csv", index=False)

    # Compare metadata IDs and split references against the download inventory.
    report_ids = set(df["image_id"].astype(str).str.strip())
    downloaded_ids = set(
        df.loc[df["status"].astype(str).str.lower() == "downloaded", "image_id"]
          .astype(str).str.strip()
    )

    metadata_path = ROOT / "metadata" / "images.jsonl"
    metadata_ids = set()
    reference_rows = []

    if metadata_path.exists():
        for item in read_jsonl(metadata_path):
            value = item.get("image_id")
            if value is not None:
                metadata_ids.add(str(value).strip())
        reference_rows.append({
            "source": "metadata/images.jsonl",
            "records_or_unique_references": len(metadata_ids),
            "references_in_download_report": len(metadata_ids & report_ids),
            "downloaded_and_available": len(metadata_ids & downloaded_ids),
            "missing_from_download_report": len(metadata_ids - report_ids),
            "not_downloaded": len(metadata_ids - downloaded_ids),
            "status": "checked",
        })
    else:
        reference_rows.append({
            "source": "metadata/images.jsonl",
            "records_or_unique_references": "",
            "references_in_download_report": "",
            "downloaded_and_available": "",
            "missing_from_download_report": "",
            "not_downloaded": "",
            "status": "file_not_found",
        })

    split_files = [
        ROOT / "splits" / split_type / f"{split}.jsonl"
        for split_type in ("single_image", "multi_image")
        for split in ("train", "dev", "test")
    ]

    all_split_refs = set()
    for split_path in split_files:
        if not split_path.exists():
            reference_rows.append({
                "source": str(split_path.relative_to(ROOT)),
                "records_or_unique_references": "",
                "references_in_download_report": "",
                "downloaded_and_available": "",
                "missing_from_download_report": "",
                "not_downloaded": "",
                "status": "file_not_found",
            })
            continue

        refs = set()
        records = read_jsonl(split_path)
        for item in records:
            collect_image_ids(item, refs)
        all_split_refs.update(refs)

        reference_rows.append({
            "source": str(split_path.relative_to(ROOT)),
            "records_or_unique_references": len(refs),
            "references_in_download_report": len(refs & report_ids),
            "downloaded_and_available": len(refs & downloaded_ids),
            "missing_from_download_report": len(refs - report_ids),
            "not_downloaded": len(refs - downloaded_ids),
            "status": "checked",
        })

    pd.DataFrame(reference_rows).to_csv(
        ROOT / "reference_validation_report.csv", index=False
    )

    validation_counts = image_report["validation_status"].value_counts().to_dict()
    valid_count = validation_counts.get("valid", 0)
    failed_downloads = validation_counts.get("download_failed", 0)
    missing_files = validation_counts.get("file_missing", 0)
    corrupt_files = validation_counts.get("corrupt_or_unsupported", 0)
    duplicate_file_count = len(duplicates)
    duplicate_groups = duplicates["duplicate_group"].nunique() if not duplicates.empty else 0

    summary = [
        "HAMOS IMAGE VALIDATION SUMMARY",
        "==============================",
        f"Rows in download report: {len(df):,}",
        f"Download status counts: {dict(status_counts)}",
        f"Files successfully opened and verified: {valid_count:,}",
        f"Download failures recorded: {failed_downloads:,}",
        f"Downloaded files missing from disk: {missing_files:,}",
        f"Corrupt/unsupported files: {corrupt_files:,}",
        f"Exact duplicate groups: {duplicate_groups:,}",
        f"Files participating in exact duplicate groups: {duplicate_file_count:,}",
        f"Unique image IDs in download report: {len(report_ids):,}",
        f"Unique downloaded image IDs: {len(downloaded_ids):,}",
        f"Unique image IDs referenced by split files: {len(all_split_refs):,}",
        f"Split references not found in download report: {len(all_split_refs - report_ids):,}",
        f"Split references without a downloaded image: {len(all_split_refs - downloaded_ids):,}",
        "",
        "Files created:",
        "- image_validation_report.csv",
        "- duplicate_images.csv",
        "- reference_validation_report.csv",
    ]

    summary_text = "\n".join(summary)
    (ROOT / "validation_summary.txt").write_text(summary_text, encoding="utf-8")
    print("\n" + summary_text)
    print("\nValidation finished.")

if __name__ == "__main__":
    main()
