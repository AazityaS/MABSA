"""Command-line entry point for read-only HAMOS validation."""

import argparse
import json

from .validation import validate_hamos, write_validation_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="HAMOS project directory")
    parser.add_argument("--skip-image-decode", action="store_true",
                        help="check existence but skip full Pillow decode verification")
    args = parser.parse_args()
    result = validate_hamos(args.root, verify_images=not args.skip_image_decode)
    output = write_validation_report(args.root, result)
    print(json.dumps({"counts": result["dataset_counts"], "splits": {
        split: {"segments": values["segments"], "quad_preservation": values["quad_preservation"],
                "missing_primary_image_ids": values["missing_primary_image_ids"]}
        for split, values in result["splits"].items()},
        "hotel_overlap_counts": result["hotel_overlap_counts"],
        "image_validation": result["image_validation"], "errors": result["errors"],
        "report": str(output)}, ensure_ascii=False, indent=2))
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

