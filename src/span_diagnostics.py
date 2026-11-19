"""Create a read-only report for explicit annotations with empty token spans."""

import csv
import json
import unicodedata
from collections import Counter
from pathlib import Path

from .data_loader import read_jsonl


def diagnose_empty_explicit_spans(root):
    root = Path(root).resolve()
    segments = {r["segment_id"]: r for r in read_jsonl(root / "metadata/segments.jsonl")}
    reviews = {r["source_review_id"]: r for r in read_jsonl(root / "metadata/reviews.jsonl")}
    records = []
    for quad in read_jsonl(root / "annotations/quads.jsonl"):
        for side in ("aspect", "opinion"):
            if quad.get(f"{side}_implicit") or quad.get(f"{side}_span"):
                continue
            segment = segments[quad["segment_id"]]
            review = reviews[segment["source_review_id"]]
            term = quad.get(f"{side}_term")
            text = segment["segment_text"]
            exact = term in text if term is not None else False
            norm = lambda value: unicodedata.normalize("NFKC", value or "").casefold()
            normalized = norm(term) in norm(text) if term is not None else False
            records.append({
                "quad_id": quad["quad_id"],
                "segment_id": quad["segment_id"],
                "source_review_id": segment["source_review_id"],
                "language": segment["language"],
                "issue_side": side,
                "aspect_term": quad.get("aspect_term"),
                "aspect_span": quad.get("aspect_span"),
                "aspect_implicit": quad.get("aspect_implicit"),
                "opinion_term": quad.get("opinion_term"),
                "opinion_span": quad.get("opinion_span"),
                "opinion_implicit": quad.get("opinion_implicit"),
                "term_under_review": term,
                "original_span": quad.get(f"{side}_span"),
                "term_appears_exactly_in_segment": exact,
                "term_appears_after_nfkc_casefold": normalized,
                "segment_text": text,
                "review_text_equals_segment_text": review.get("review_text") == text,
                "segment_text_occurs_in_review_text": text in (review.get("review_text") or ""),
                "diagnostic_class": "TERM_PRESENT_BUT_SOURCE_SPAN_EMPTY" if exact else "TERM_NOT_FOUND_AND_SOURCE_SPAN_EMPTY",
            })
    records.sort(key=lambda r: (r["quad_id"], r["issue_side"]))
    return {
        "issue_count": len(records),
        "by_side": dict(Counter(r["issue_side"] for r in records)),
        "term_exactly_present": sum(r["term_appears_exactly_in_segment"] for r in records),
        "term_not_found": sum(not r["term_appears_exactly_in_segment"] for r in records),
        "review_text_equals_segment_text": sum(r["review_text_equals_segment_text"] for r in records),
        "segment_text_occurs_in_review_text": sum(r["segment_text_occurs_in_review_text"] for r in records),
        "records": records,
    }


def write_span_diagnostics(root, report=None):
    root = Path(root).resolve()
    report = report or diagnose_empty_explicit_spans(root)
    output_dir = root / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "empty_explicit_span_diagnostics.json"
    csv_path = output_dir / "empty_explicit_span_diagnostics.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = [k for k in report["records"][0]] if report["records"] else []
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in report["records"]:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})
    return json_path, csv_path

