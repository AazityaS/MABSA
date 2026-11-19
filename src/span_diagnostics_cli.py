"""Write the explicit-empty-span diagnostic CSV and JSON reports."""

import argparse
from .span_diagnostics import diagnose_empty_explicit_spans, write_span_diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="HAMOS project directory")
    args = parser.parse_args()
    report = diagnose_empty_explicit_spans(args.root)
    json_path, csv_path = write_span_diagnostics(args.root, report)
    print(f"Issues: {report['issue_count']}; by side: {report['by_side']}; "
          f"exact term matches: {report['term_exactly_present']}; term not found: {report['term_not_found']}")
    print(f"Wrote {json_path} and {csv_path}")


if __name__ == "__main__":
    main()
