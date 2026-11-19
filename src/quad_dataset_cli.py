"""Write per-split quad-level dataset counts and target distributions."""

import argparse
import json

from .quad_dataset import summarize_quad_datasets, write_quad_dataset_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="HAMOS project directory")
    args = parser.parse_args()
    report = summarize_quad_datasets(args.root)
    path = write_quad_dataset_summary(args.root, report)
    print(json.dumps({"splits": {name: {key: value[key] for key in (
        "quad_instances", "multimodal_usable_instances", "text_only_instances",
        "missing_primary_image_excluded_from_multimodal", "instances_with_implicit_aspect")}
        for name, value in report["splits"].items()},
        "hotel_overlap_counts": report["hotel_overlap_counts"],
        "no_hotel_leakage": report["no_hotel_leakage"],
        "all_quads_preserved_in_text_view": report["all_quads_preserved_in_text_view"],
        "summary_path": str(path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
