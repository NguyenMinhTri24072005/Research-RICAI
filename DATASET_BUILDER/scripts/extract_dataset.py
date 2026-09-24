#!/usr/bin/env python3
"""CLI tool for batch dataset extraction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from rice_dataset.config import ExtractionConfig
from rice_dataset.io.image_index import validate_flat_inventory


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run batch computer vision dataset extraction pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="DATASET_BUILDER/config/extraction_config.json",
        help="Path to extraction configuration JSON file.",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Override project root directory (useful for Google Colab).",
    )
    parser.add_argument(
        "--workbook",
        type=str,
        default=None,
        help="Override path to manual records workbook.",
    )
    parser.add_argument(
        "--images",
        type=str,
        default=None,
        help="Override path to raw images directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit execution to first N samples.",
    )
    parser.add_argument(
        "--sample-id",
        type=str,
        default=None,
        help="Run pipeline on a single specific Sample_ID.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the flat M#### workbook/image inventory without loading AI models.",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Skip writing canonical audit and training dataset files.",
    )

    args = parser.parse_args()

    config_file = Path(args.config)
    if not config_file.exists():
        print(f"❌ Error: Config file not found: {config_file}", file=sys.stderr)
        return 1

    config = ExtractionConfig.from_file(config_file, project_root_override=args.project_root)

    if args.workbook:
        config.workbook_path = Path(args.workbook)
    if args.images:
        config.raw_images_dir = Path(args.images)

    if args.validate_only:
        from rice_dataset.io.manual_records import load_manual_records

        records, errors = load_manual_records(config.get_resolved_workbook_path())
        if errors:
            print(f"Input validation failed: {len(errors)} invalid workbook row(s).", file=sys.stderr)
            return 1
        summary = validate_flat_inventory(
            records,
            config.get_resolved_raw_images_dir(),
            prefix=config.id_prefix,
            digits=config.id_digits,
        )
        print(
            "Input inventory valid: "
            f"{summary['record_count']} workbook rows, {summary['image_count']} flat images."
        )
        return 0
    from rice_dataset.pipeline import DatasetExtractionPipeline

    pipeline = DatasetExtractionPipeline(config)


    if args.sample_id:
        from rice_dataset.io.manual_records import load_manual_records

        records, _ = load_manual_records(config.get_resolved_workbook_path())
        matched = [r for r in records if r.sample_id.upper() == args.sample_id.strip().upper()]
        if not matched:
            print(f"❌ Error: Sample_ID '{args.sample_id}' not found in workbook.", file=sys.stderr)
            return 1
        pipeline.validate_input_inventory([matched[0]], require_exact_match=False)


        print(f"🎯 Processing single sample: {args.sample_id}...")
        res = pipeline.process_single_sample(matched[0])
        qc_status = res.get("QC_Status")
        print(f"QC Status: {qc_status}")
        if res.get("QC_Reason"):
            print(f"QC Reason: {res.get('QC_Reason')}")
        if not args.no_export:
            pipeline.export_results([res])
        return 0 if qc_status == "PASS" else 1

    results = pipeline.run_batch(limit=args.limit, export=not args.no_export)
    if not results:
        print("❌ Error: No samples were processed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
