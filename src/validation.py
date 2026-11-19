"""Read-only integrity and distribution checks for HAMOS official splits."""

import collections
import hashlib
import json
from pathlib import Path

from PIL import Image

from .data_loader import read_jsonl
from .image_utils import resolve_image_path
from .label_mapping import build_label_mappings


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_hamos(root, verify_images=True):
    root = Path(root).resolve()
    errors, warnings = [], []
    segments = read_jsonl(root / "metadata/segments.jsonl")
    segment_map = {r["segment_id"]: r for r in segments}
    reviews = read_jsonl(root / "metadata/reviews.jsonl")
    image_rows = read_jsonl(root / "metadata/images.jsonl")
    quads = read_jsonl(root / "annotations/quads.jsonl")
    segment_ids = [r["segment_id"] for r in segments]
    segment_counts = collections.Counter(segment_ids)
    review_map = {r["source_review_id"]: r for r in reviews}
    image_map = {r["image_id"]: r for r in image_rows}
    quad_by_segment = collections.defaultdict(list)
    quad_id_counts = collections.Counter(q["quad_id"] for q in quads)
    for q in quads:
        quad_by_segment[q["segment_id"]].append(q)
    for sid, count in segment_counts.items():
        if count != 1:
            errors.append(f"segment_id {sid} appears {count} times in metadata")
    if any(count != 1 for count in quad_id_counts.values()):
        errors.append("Duplicate quad_id values exist")
    unknown_sentiments = sorted({q.get("sentiment") for q in quads} - {"positive", "negative", "neutral"})
    if unknown_sentiments:
        errors.append(f"Unknown sentiments: {unknown_sentiments}")
    maps = build_label_mappings(root / "taxonomy/hotel_absa_taxonomy.xlsx")
    code_to_category = maps["taxonomy_code_to_category"]
    unknown_codes = sorted({q.get("taxonomy_code") for q in quads} - set(code_to_category))
    if unknown_codes:
        errors.append(f"Unknown taxonomy codes: {unknown_codes}")
    category_mismatches = [q["quad_id"] for q in quads if q.get("taxonomy_code") in code_to_category
                           and code_to_category[q["taxonomy_code"]] != q.get("aspect_category")]
    if category_mismatches:
        errors.append(f"Taxonomy/category mismatches: {len(category_mismatches)}")

    missing_segment_refs = sorted({q["segment_id"] for q in quads} - set(segment_counts))
    if missing_segment_refs:
        errors.append(f"Quads with missing segment references: {len(missing_segment_refs)}")
    bad_nulls, empty_explicit_spans = [], []
    for q in quads:
        for field in ("aspect_category", "taxonomy_code", "sentiment", "aspect_implicit", "opinion_implicit"):
            if q.get(field) is None:
                bad_nulls.append((q["quad_id"], field))
        if q.get("aspect_implicit") and (q.get("aspect_term") is not None or q.get("aspect_span") != []):
            bad_nulls.append((q["quad_id"], "implicit aspect term/span invariant"))
        if not q.get("aspect_implicit"):
            if q.get("aspect_term") is None:
                bad_nulls.append((q["quad_id"], "explicit aspect term missing"))
            if not q.get("aspect_span"):
                empty_explicit_spans.append((q["quad_id"], "aspect"))
        if q.get("opinion_implicit") and (q.get("opinion_term") is not None or q.get("opinion_span") != []):
            bad_nulls.append((q["quad_id"], "implicit opinion term/span invariant"))
        if not q.get("opinion_implicit"):
            if q.get("opinion_term") is None:
                bad_nulls.append((q["quad_id"], "explicit opinion term missing"))
            if not q.get("opinion_span"):
                empty_explicit_spans.append((q["quad_id"], "opinion"))
    if bad_nulls:
        warnings.append(f"Null/implicitness field discrepancies: {len(bad_nulls)} (examples: {bad_nulls[:5]})")
    if empty_explicit_spans:
        warnings.append(f"Explicit terms with empty token spans: {len(empty_explicit_spans)} "
                        f"(examples: {empty_explicit_spans[:8]})")

    image_refs_primary, image_refs_support = set(), set()
    for q in quads:
        primary = q.get("primary_image_id")
        source_review_id = segment_map.get(q.get("segment_id"), {}).get("source_review_id")
        if not primary or primary not in image_map:
            errors.append(f"Quad {q['quad_id']} primary image is absent from metadata: {primary}")
        else:
            image_refs_primary.add(primary)
            if image_map[primary].get("source_review_id") != source_review_id:
                errors.append(f"Quad {q['quad_id']} primary image belongs to a different review")
        for supporting in q.get("supporting_image_ids") or []:
            image_refs_support.add(supporting)
            if supporting not in image_map:
                errors.append(f"Quad {q['quad_id']} supporting image is absent from metadata: {supporting}")
            elif image_map[supporting].get("source_review_id") != source_review_id:
                errors.append(f"Quad {q['quad_id']} supporting image belongs to a different review")

    image_status = {}
    corrupt_paths = []
    for image_id, image in image_map.items():
        path = resolve_image_path(root, image.get("local_path"), image_id)
        exists = path.is_file()
        readable = None
        if exists and verify_images:
            try:
                with Image.open(path) as img:
                    img.verify()
                readable = True
            except Exception:
                readable = False
                corrupt_paths.append(image_id)
        image_status[image_id] = {"path": str(path), "exists": exists, "readable": readable}

    split_rows, split_hotels, split_hashes = {}, {}, {}
    split_quads_preserved = {}
    split_lang, split_sentiment, split_category, split_code = {}, {}, {}, {}
    split_missing_primary, split_missing_supporting = {}, {}
    all_split_segment_ids = []
    for split in ("train", "dev", "test"):
        path = root / "splits/single_image" / f"{split}.jsonl"
        rows = read_jsonl(path)
        split_rows[split] = rows
        split_hashes[split] = _sha256(path)
        hotels, lang_counts = set(), collections.Counter()
        sentiments, categories, codes = collections.Counter(), collections.Counter(), collections.Counter()
        expected_count, observed_count = 0, 0
        no_primary, no_support = set(), set()
        for row in rows:
            sid = row.get("segment_id")
            all_split_segment_ids.append(sid)
            segment = segment_map.get(sid)
            if segment is None:
                errors.append(f"Split {split} segment missing from metadata: {sid}")
            else:
                lang_counts[segment.get("language", "<missing>")] += 1
            review = review_map.get(row.get("source_review_id"))
            if review is None:
                errors.append(f"Split {split} review missing from metadata: {row.get('source_review_id')}")
            else:
                hotels.add(str(review["hotel_id"]))
            master = quad_by_segment.get(sid, [])
            expected_count += len(master)
            observed_count += len(row.get("quads", []))
            if len(master) != len(row.get("quads", [])):
                errors.append(f"Split {split} does not preserve quad count for {sid}")
            master_by_id = {q["quad_id"]: q for q in master}
            split_by_id = {q["quad_id"]: q for q in row.get("quads", [])}
            if set(master_by_id) != set(split_by_id):
                errors.append(f"Split {split} quad IDs differ from master annotations for {sid}")
            else:
                for qid, master_q in master_by_id.items():
                    projected = split_by_id[qid]
                    for field in ("aspect_term", "aspect_category", "taxonomy_code", "opinion_term", "sentiment",
                                  "aspect_implicit", "opinion_implicit", "aspect_span", "opinion_span", "primary_image_id"):
                        if projected.get(field) != master_q.get(field):
                            errors.append(f"Split {split} differs from master on {qid}.{field}")
                    sentiments[master_q.get("sentiment", "<missing>")] += 1
                    categories[master_q.get("aspect_category", "<missing>")] += 1
                    codes[master_q.get("taxonomy_code", "<missing>")] += 1
                    primary = master_q.get("primary_image_id")
                    if primary not in image_status or not image_status[primary]["exists"]:
                        no_primary.add(primary)
                    for support in master_q.get("supporting_image_ids") or []:
                        if support not in image_status or not image_status[support]["exists"]:
                            no_support.add(support)
        split_hotels[split] = hotels
        split_lang[split], split_sentiment[split] = dict(lang_counts), dict(sentiments)
        split_category[split], split_code[split] = dict(categories), dict(codes)
        split_quads_preserved[split] = {"split_quad_count": observed_count, "master_quad_count": expected_count,
                                        "all_preserved": observed_count == expected_count}
        split_missing_primary[split], split_missing_supporting[split] = sorted(no_primary), sorted(no_support)
        if len({r["segment_id"] for r in rows}) != len(rows):
            errors.append(f"Duplicate segment IDs in {split} split")
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        overlap = split_hotels[a] & split_hotels[b]
        if overlap:
            errors.append(f"Hotel leakage {a}/{b}: {len(overlap)} hotel IDs")
    split_segment_counts = collections.Counter(all_split_segment_ids)
    duplicate_split_segments = sum(n != 1 for n in split_segment_counts.values())
    unassigned_segments = set(segment_map) - set(split_segment_counts)
    extra_segments = set(split_segment_counts) - set(segment_map)
    if duplicate_split_segments or unassigned_segments or extra_segments:
        errors.append(f"Official split segment coverage discrepancy: duplicate={duplicate_split_segments}, "
                      f"unassigned={len(unassigned_segments)}, extra={len(extra_segments)}")

    quad_counts = collections.Counter({sid: len(qs) for sid, qs in quad_by_segment.items()})
    multi_segments = sum(n > 1 for n in quad_counts.values())
    aspect_implicit = collections.Counter("implicit" if q.get("aspect_implicit") else "explicit" for q in quads)
    opinion_implicit = collections.Counter("implicit" if q.get("opinion_implicit") else "explicit" for q in quads)
    result = {
        "dataset_counts": {"reviews": len(reviews), "segments": len(segments), "images": len(image_rows),
                           "quads": len(quads), "hotels": len({str(r['hotel_id']) for r in reviews})},
        "splits": {s: {"segments": len(split_rows[s]), "hotels": len(split_hotels[s]),
                       "sha256": split_hashes[s], "quad_preservation": split_quads_preserved[s],
                       "language_counts": split_lang[s], "sentiment_counts": split_sentiment[s],
                       "aspect_category_counts": split_category[s], "taxonomy_code_counts": split_code[s],
                       "missing_primary_image_ids": split_missing_primary[s],
                       "missing_supporting_image_ids": split_missing_supporting[s]} for s in split_rows},
        "hotel_overlap_counts": {"train_dev": len(split_hotels["train"] & split_hotels["dev"]),
                                 "train_test": len(split_hotels["train"] & split_hotels["test"]),
                                 "dev_test": len(split_hotels["dev"] & split_hotels["test"])},
        "official_split_coverage": {"segments_assigned_exactly_once": len(all_split_segment_ids) == len(segment_map)
                                     and duplicate_split_segments == 0 and not unassigned_segments and not extra_segments,
                                     "duplicate_segment_rows": duplicate_split_segments,
                                     "unassigned_segment_count": len(unassigned_segments),
                                     "extra_segment_count": len(extra_segments)},
        "quad_structure": {"segments_with_quads": len(quad_counts), "segments_with_multiple_quads": multi_segments,
                           "maximum_quads_per_segment": max(quad_counts.values(), default=0),
                           "aspect_implicit_counts": dict(aspect_implicit),
                           "opinion_implicit_counts": dict(opinion_implicit)},
        "annotation_integrity": {"unexpected_null_or_implicitness_violations": len(bad_nulls),
                                 "explicit_empty_span_count": len(empty_explicit_spans),
                                 "explicit_empty_span_examples": empty_explicit_spans[:20]},
        "taxonomy": {"broad_category_count": len(maps["aspect_categories"]),
                     "fine_grained_code_count": len(maps["taxonomy_codes"]),
                     "code_to_category": code_to_category},
        "image_validation": {"metadata_images": len(image_map),
                             "files_exist": sum(v["exists"] for v in image_status.values()),
                             "files_missing": sum(not v["exists"] for v in image_status.values()),
                             "verified_readable": sum(v["readable"] is True for v in image_status.values()),
                             "corrupt_image_ids": corrupt_paths,
                             "primary_image_references": len(image_refs_primary),
                             "supporting_image_references": len(image_refs_support),
                             "quad_count_with_unavailable_primary": sum(not image_status.get(q.get("primary_image_id"), {}).get("exists", False) for q in quads),
                             "segments_with_unavailable_primary": len({q["segment_id"] for q in quads
                                 if not image_status.get(q.get("primary_image_id"), {}).get("exists", False)})},
        "errors": errors, "warnings": warnings,
    }
    return result


def write_validation_report(root, result, report_path=None):
    root = Path(root)
    output = Path(report_path) if report_path else root / "reports/validation_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return output

