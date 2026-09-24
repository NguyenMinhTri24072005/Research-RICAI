#!/usr/bin/env python3
"""CLI tool for migrating legacy Sample IDs and image files to flat M#### structure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add DATASET_BUILDER/src to sys.path
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from rice_dataset.migration.migrate_ids import SampleIDMigrator


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migrate legacy Sample IDs and images to flat M#### naming.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--workbook",
        type=str,
        required=True,
        help="Path to authoritative manual records workbook (e.g. manual_data.xlsx).",
    )
    parser.add_argument(
        "--images",
        type=str,
        default="DATASET_BUILDER/1_Raw_Images",
        help="Path to raw images directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="DATASET_BUILDER/migrations",
        help="Directory to store migration reports, backups and manifests.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="M",
        help="ID prefix for target naming.",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=4,
        help="Number of digits in target ID (e.g. 4 -> M0001).",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="Starting sequential integer for target ID.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform preflight inventory and generate conflict reports without modifying data.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Execute the migration with staging, byte verification, and atomic promotion.",
    )
    parser.add_argument(
        "--rollback",
        type=str,
        default=None,
        help="Path to migration manifest JSON to roll back.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force apply even if non-critical warnings exist (will NOT bypass severe corruption).",
    )
    parser.add_argument(
        "--force-resequence",
        action="store_true",
        help="Force resequencing if dataset is already in target format.",
    )

    args = parser.parse_args()

    # Handle rollback
    if args.rollback:
        print(f"🔄 Rolling back migration from manifest: {args.rollback}")
        res = SampleIDMigrator.rollback(args.rollback)
        print("✅ Rollback successful:")
        print(f"   Restored workbook: {res['restored_workbook']}")
        print(f"   Restored images  : {res['restored_images']}")
        return 0

    # Default to dry-run if neither dry-run nor apply is specified
    is_apply = args.apply
    if not is_apply and not args.dry_run:
        print("ℹ️  Neither --dry-run nor --apply specified. Defaulting to safe --dry-run mode.")

    migrator = SampleIDMigrator(
        workbook_path=args.workbook,
        images_dir=args.images,
        output_dir=args.output_dir,
        prefix=args.prefix,
        digits=args.digits,
        start=args.start,
    )

    if not is_apply:
        print(f"🔍 Running preflight dry-run inspection on workbook: {args.workbook}...")
        report = migrator.dry_run(export_reports=True)

        print("\n" + "=" * 65)
        print("📊 MIGRATION PREFLIGHT SUMMARY (DRY-RUN)")
        print("=" * 65)
        print(f"Workbook              : {report.workbook_path}")
        print(f"Images Directory      : {report.images_dir}")
        print(f"Total Excel Rows      : {report.total_excel_rows}")
        print(f"Non-empty Excel Rows  : {report.non_empty_excel_rows}")
        print(f"Unique Normalized IDs : {report.unique_normalized_ids}")
        print(f"Duplicate Excel IDs   : {report.duplicate_excel_id_count}")
        if report.duplicate_excel_ids:
            for d in report.duplicate_excel_ids:
                print(f"   - {d['normalized_id']}: {d['count']} times (rows {d['rows']})")

        print(f"Total Images on Disk  : {report.total_images_on_disk}")
        print(f"Unique Image Stems    : {report.unique_image_stems}")
        print(f"Duplicate Image Stems : {report.duplicate_image_stem_count}")

        print(f"Excel-only (No Image) : {report.excel_without_image_count}")
        if report.excel_without_image_ids:
            print(f"   Sample list: {report.excel_without_image_ids[:10]}...")

        print(f"Image-only (No Excel) : {report.images_without_excel_count}")
        if report.images_without_excel_stems:
            print(f"   Sample list: {report.images_without_excel_stems[:10]}...")

        print(f"Proposed Mappings     : {report.proposed_mappings_count}")
        print(f"Apply Ready           : {'YES' if report.is_apply_ready else 'NO (Blocked)'}")
        print("=" * 65)

        if not report.is_apply_ready:
            print("🚫 BLOCKERS DETECTED:")
            for b in report.blockers:
                print(f"   ❌ {b}")
            print("\nReports generated in:")
            print(f"   - {args.output_dir}/id_mapping.csv")
            print(f"   - {args.output_dir}/duplicate_excel_ids.csv")
            print(f"   - {args.output_dir}/excel_without_image.csv")
            print(f"   - {args.output_dir}/images_without_excel.csv")
            print(f"   - {args.output_dir}/migration_report.json")
            return 1
        else:
            print("✅ Preflight checks passed. Dataset is apply-ready.")
            return 0

    # Apply mode
    print(f"🚀 Applying ID migration on: {args.workbook}...")
    try:
        res = migrator.apply(force=args.force, force_resequence=args.force_resequence)
        if res.get("status") == "NOOP_ALREADY_MIGRATED":
            print(f"ℹ️  {res.get('message')}")
            return 0
        print("\n" + "=" * 65)
        print("🎉 MIGRATION APPLIED SUCCESSFULLY!")
        print("=" * 65)
        print(f"Migrated Samples : {res['migrated_samples']}")
        print(f"Backup Dir       : {res['backup_dir']}")
        print(f"Manifest File    : {res['manifest_path']}")
        print("To roll back this migration, run:")
        print(f"   python DATASET_BUILDER/scripts/migrate_sample_ids.py --workbook {args.workbook} --rollback {res['manifest_path']}")
        print("=" * 65)
        return 0
    except Exception as e:
        print(f"\n❌ MIGRATION FAILED: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
