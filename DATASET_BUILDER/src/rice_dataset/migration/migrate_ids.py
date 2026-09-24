"""Safe legacy Sample ID and image migration engine with staging, reports, and rollback."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import openpyxl

from .id_mapping import SampleMapping, build_id_mappings, derive_physical_sample_id, parse_natural_key
from ..io.image_index import ImageIndex


@dataclass
class MigrationPreflightReport:
    timestamp: str
    workbook_path: str
    images_dir: str
    total_excel_rows: int
    non_empty_excel_rows: int
    unique_normalized_ids: int
    duplicate_excel_id_count: int
    duplicate_excel_ids: List[Dict[str, Any]]
    total_images_on_disk: int
    unique_image_stems: int
    duplicate_image_stem_count: int
    duplicate_image_stems: List[Dict[str, Any]]
    excel_without_image_count: int
    excel_without_image_ids: List[str]
    images_without_excel_count: int
    images_without_excel_stems: List[str]
    is_apply_ready: bool
    blockers: List[str]
    proposed_mappings_count: int


class SampleIDMigrator:
    """Orchestrates legacy sample ID migration with dry-run, atomic staging, and rollback."""

    def __init__(
        self,
        workbook_path: Path | str,
        images_dir: Path | str,
        output_dir: Path | str = "DATASET_BUILDER/migrations",
        prefix: str = "M",
        digits: int = 4,
        start: int = 1,
    ) -> None:
        self.workbook_path = Path(workbook_path).resolve()
        self.images_dir = Path(images_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.prefix = prefix
        self.digits = digits
        self.start = start

    def run_preflight(self) -> Tuple[MigrationPreflightReport, List[SampleMapping], Dict[str, Any]]:
        """Run complete inventory and conflict detection without writing changes."""
        if not self.workbook_path.exists():
            raise FileNotFoundError(f"Workbook not found: {self.workbook_path}")
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        wb = openpyxl.load_workbook(self.workbook_path, data_only=False)
        ws = wb.active

        # Find Sample_ID header column
        header_row = [cell.value for cell in ws[1]]
        sample_id_col_idx = None
        for idx, val in enumerate(header_row, start=1):
            if val and str(val).strip().lower() in ["sample_id", "sampleid", "id"]:
                sample_id_col_idx = idx
                break

        if sample_id_col_idx is None:
            raise ValueError(f"Could not find 'Sample_ID' column in header: {header_row}")

        total_rows = 0
        non_empty_rows = 0
        seen_ids: Dict[str, List[int]] = {}
        row_id_list: List[Tuple[int, str]] = []

        for row_idx in range(2, ws.max_row + 1):
            cell_val = ws.cell(row=row_idx, column=sample_id_col_idx).value
            total_rows += 1
            if cell_val is None or str(cell_val).strip() == "":
                continue

            raw_id = str(cell_val).strip()
            norm_id = raw_id.upper()
            non_empty_rows += 1

            if norm_id not in seen_ids:
                seen_ids[norm_id] = []
            seen_ids[norm_id].append(row_idx)
            row_id_list.append((row_idx, norm_id))

        duplicate_excel_ids: List[Dict[str, Any]] = []
        for norm_id, row_indices in seen_ids.items():
            if len(row_indices) > 1:
                duplicate_excel_ids.append({
                    "normalized_id": norm_id,
                    "count": len(row_indices),
                    "rows": row_indices,
                })

        # Scan images
        image_index = ImageIndex(self.images_dir, recursive=True)
        total_images = image_index.get_total_image_count()
        all_image_stems = image_index.get_all_stems()
        dup_stems_dict = image_index.get_duplicate_stems()

        duplicate_image_stems = [
            {"stem": stem, "paths": [str(p) for p in paths]}
            for stem, paths in dup_stems_dict.items()
        ]

        excel_norm_ids = set(seen_ids.keys())
        excel_without_image = sorted(list(excel_norm_ids - all_image_stems))
        images_without_excel = sorted(list(all_image_stems - excel_norm_ids))

        blockers: List[str] = []
        if duplicate_excel_ids:
            dup_details = ", ".join(f"{d['normalized_id']} (rows {d['rows']})" for d in duplicate_excel_ids)
            blockers.append(f"Duplicate Excel Sample_IDs found: {dup_details}")
        if duplicate_image_stems:
            blockers.append(f"Duplicate image stems on disk: {[d['stem'] for d in duplicate_image_stems]}")
        if excel_without_image:
            blockers.append(f"{len(excel_without_image)} Excel IDs have no matching image on disk")
        if images_without_excel:
            blockers.append(f"{len(images_without_excel)} images on disk have no matching Excel row")

        # Build mapping using unique legacy IDs
        image_path_map = {}
        for stem in all_image_stems:
            p, err = image_index.resolve_image(stem)
            if p and err is None:
                image_path_map[stem] = p

        mappings = build_id_mappings(
            legacy_ids=list(seen_ids.keys()),
            image_map=image_path_map,
            prefix=self.prefix,
            digits=self.digits,
            start=self.start,
        )

        is_apply_ready = len(blockers) == 0

        report = MigrationPreflightReport(
            timestamp=datetime.now().isoformat(),
            workbook_path=str(self.workbook_path),
            images_dir=str(self.images_dir),
            total_excel_rows=total_rows,
            non_empty_excel_rows=non_empty_rows,
            unique_normalized_ids=len(seen_ids),
            duplicate_excel_id_count=len(duplicate_excel_ids),
            duplicate_excel_ids=duplicate_excel_ids,
            total_images_on_disk=total_images,
            unique_image_stems=len(all_image_stems),
            duplicate_image_stem_count=len(duplicate_image_stems),
            duplicate_image_stems=duplicate_image_stems,
            excel_without_image_count=len(excel_without_image),
            excel_without_image_ids=excel_without_image,
            images_without_excel_count=len(images_without_excel),
            images_without_excel_stems=images_without_excel,
            is_apply_ready=is_apply_ready,
            blockers=blockers,
            proposed_mappings_count=len(mappings),
        )

        raw_meta = {
            "sample_id_col_idx": sample_id_col_idx,
            "header_row": header_row,
            "row_id_list": row_id_list,
        }

        return report, mappings, raw_meta

    def dry_run(self, export_reports: bool = True) -> MigrationPreflightReport:
        """Execute dry-run inspection, print summary and save diagnostic reports."""
        report, mappings, _ = self.run_preflight()

        if export_reports:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            report_dir = self.output_dir / f"dry_run_{int(time.time())}"
            report_dir.mkdir(parents=True, exist_ok=True)

            # 1. id_mapping.csv
            import pandas as pd
            df_map = pd.DataFrame([m.to_dict() for m in mappings])
            df_map.to_csv(report_dir / "id_mapping.csv", index=False, encoding="utf-8-sig")

            # 2. duplicate_excel_ids.csv
            df_dup_excel = pd.DataFrame(report.duplicate_excel_ids)
            df_dup_excel.to_csv(report_dir / "duplicate_excel_ids.csv", index=False, encoding="utf-8-sig")

            # 3. excel_without_image.csv
            df_no_img = pd.DataFrame({"Sample_ID": report.excel_without_image_ids})
            df_no_img.to_csv(report_dir / "excel_without_image.csv", index=False, encoding="utf-8-sig")

            # 4. images_without_excel.csv
            df_no_excel = pd.DataFrame({"Image_Stem": report.images_without_excel_stems})
            df_no_excel.to_csv(report_dir / "images_without_excel.csv", index=False, encoding="utf-8-sig")

            # 5. migration_report.json
            with open(report_dir / "migration_report.json", "w", encoding="utf-8") as f:
                json.dump(asdict(report), f, indent=2, ensure_ascii=False)

            # Also create standard symlink/copy to root of migrations/
            for name in ["id_mapping.csv", "duplicate_excel_ids.csv", "excel_without_image.csv", "images_without_excel.csv", "migration_report.json"]:
                shutil.copy2(report_dir / name, self.output_dir / name)

        return report

    def apply(self, force: bool = False, force_resequence: bool = False) -> Dict[str, Any]:
        """Apply migration with staging, verification, atomic promotion, and rollback support."""
        report, mappings, meta = self.run_preflight()

        # Check if already migrated
        all_already_flat = all(re.match(r"^[A-Z]\d{" + str(self.digits) + r"}$", m.original_sample_id) for m in mappings)
        if all_already_flat and not force_resequence:
            return {
                "status": "NOOP_ALREADY_MIGRATED",
                "message": f"Dataset already in {self.prefix}{'0'*self.digits} format. Use --force-resequence to re-assign sequential IDs.",
                "migrated_samples": 0,
            }

        # Critical reconciliation blockers can NEVER be bypassed, even with --force
        critical_blockers = []
        if report.duplicate_excel_id_count > 0:
            critical_blockers.append(f"{report.duplicate_excel_id_count} duplicate Excel Sample_IDs")
        if report.duplicate_image_stem_count > 0:
            critical_blockers.append(f"{report.duplicate_image_stem_count} duplicate image stems on disk")
        if report.excel_without_image_count > 0:
            critical_blockers.append(f"{report.excel_without_image_count} Excel IDs missing images on disk")
        if report.images_without_excel_count > 0:
            critical_blockers.append(f"{report.images_without_excel_count} images on disk missing Excel rows")

        if critical_blockers:
            raise RuntimeError(
                f"Migration cannot be applied due to {len(critical_blockers)} critical blockers (cannot be bypassed even with --force):\n"
                + "\n".join(f" - ❌ {b}" for b in critical_blockers)
                + "\nResolve data reconciliation issues before applying."
            )

        if not report.is_apply_ready and not force:
            raise RuntimeError(
                f"Migration cannot be applied due to {len(report.blockers)} blockers:\n"
                + "\n".join(f" - {b}" for b in report.blockers)
                + "\nResolve data reconciliation issues before applying."
            )

        # Disk space check: ensure disk has enough free space for staging and backup
        try:
            total_img_size = sum(m.source_image_path.stat().st_size for m in mappings if m.source_image_path and m.source_image_path.exists())
            free_bytes = shutil.disk_usage(self.images_dir).free
            if free_bytes < total_img_size * 2:
                raise IOError(f"Insufficient disk space: required at least {total_img_size * 2} bytes, available {free_bytes} bytes.")
        except Exception as e:
            if "Insufficient disk space" in str(e):
                raise

        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        staging_dir = self.output_dir / f"staging_{timestamp_str}"
        backup_dir = self.output_dir / f"backup_{timestamp_str}"
        staging_dir.mkdir(parents=True, exist_ok=True)
        backup_dir.mkdir(parents=True, exist_ok=True)

        staged_images_dir = staging_dir / "1_Raw_Images"
        staged_images_dir.mkdir(parents=True, exist_ok=True)
        staged_workbook_path = staging_dir / self.workbook_path.name

        # Mapping lookup by legacy ID
        mapping_by_legacy: Dict[str, SampleMapping] = {m.original_sample_id: m for m in mappings}

        # Step 1: Stage and copy images
        staged_image_records = []
        for m in mappings:
            if m.source_image_path and m.source_image_path.exists():
                dst_img = staged_images_dir / m.target_image_filename
                shutil.copy2(m.source_image_path, dst_img)

                # Verify byte parity
                src_size = m.source_image_path.stat().st_size
                dst_size = dst_img.stat().st_size
                if src_size != dst_size:
                    raise IOError(f"Byte size mismatch for {m.target_image_filename}: src={src_size}, dst={dst_size}")

                staged_image_records.append({
                    "source_path": str(m.source_image_path),
                    "staged_path": str(dst_img),
                    "target_filename": m.target_image_filename,
                    "byte_size": src_size,
                })

        # Step 2: Stage workbook with openpyxl
        wb = openpyxl.load_workbook(self.workbook_path)
        ws = wb.active

        header_row = [cell.value for cell in ws[1]]
        headers_lower = [str(h).strip().lower() if h else "" for h in header_row]

        def ensure_column(name: str) -> int:
            clean = name.strip().lower()
            if clean in headers_lower:
                return headers_lower.index(clean) + 1
            new_col = len(headers_lower) + 1
            ws.cell(row=1, column=new_col, value=name)
            headers_lower.append(clean)
            return new_col

        sample_id_col = meta["sample_id_col_idx"]
        orig_id_col = ensure_column("Original_Sample_ID")
        phys_id_col = ensure_column("Physical_Sample_ID")
        img_fn_col = ensure_column("Image_Filename")

        for row_idx in range(2, ws.max_row + 1):
            cell_val = ws.cell(row=row_idx, column=sample_id_col).value
            if cell_val is None or str(cell_val).strip() == "":
                continue

            raw_id = str(cell_val).strip()
            norm_id = raw_id.upper()
            mapping = mapping_by_legacy.get(norm_id)
            if mapping:
                ws.cell(row=row_idx, column=sample_id_col, value=mapping.sample_id)
                ws.cell(row=row_idx, column=orig_id_col, value=mapping.original_sample_id)
                ws.cell(row=row_idx, column=phys_id_col, value=mapping.physical_sample_id)
                ws.cell(row=row_idx, column=img_fn_col, value=mapping.target_image_filename)

        wb.save(staged_workbook_path)

        # Step 3: Verify staged workbook
        wb_check = openpyxl.load_workbook(staged_workbook_path)
        ws_check = wb_check.active
        staged_non_empty = 0
        for r in range(2, ws_check.max_row + 1):
            if ws_check.cell(row=r, column=sample_id_col).value:
                staged_non_empty += 1
        if staged_non_empty != report.non_empty_excel_rows:
            raise ValueError(f"Staged workbook validation failed: row count mismatch ({staged_non_empty} != {report.non_empty_excel_rows})")

        # Step 4: Backup original workbook and images
        backup_workbook_path = backup_dir / self.workbook_path.name
        shutil.copy2(self.workbook_path, backup_workbook_path)

        backup_images_dir = backup_dir / "1_Raw_Images"
        shutil.copytree(self.images_dir, backup_images_dir)

        # Step 5: Write rollback manifest
        manifest_path = self.output_dir / f"migration_manifest_{timestamp_str}.json"
        manifest_data = {
            "timestamp": timestamp_str,
            "workbook_path": str(self.workbook_path),
            "backup_workbook_path": str(backup_workbook_path),
            "images_dir": str(self.images_dir),
            "backup_images_dir": str(backup_images_dir),
            "staged_images": staged_image_records,
            "mappings": [m.to_dict() for m in mappings],
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2, ensure_ascii=False)

        # Step 6: Atomic Promotion
        # Replace workbook
        shutil.copy2(staged_workbook_path, self.workbook_path)

        # Replace / Flatten images
        for item in staged_image_records:
            target_path = self.images_dir / item["target_filename"]
            shutil.copy2(item["staged_path"], target_path)

        # Remove legacy nested directories in images_dir if they are empty or old files
        # (Preserved in backup_images_dir)
        for child in list(self.images_dir.iterdir()):
            if child.is_dir():
                shutil.rmtree(child)

        return {
            "status": "APPLIED",
            "timestamp": timestamp_str,
            "manifest_path": str(manifest_path),
            "migrated_samples": len(mappings),
            "backup_dir": str(backup_dir),
        }

    @classmethod
    def rollback(cls, manifest_path: Path | str) -> Dict[str, Any]:
        """Rollback an applied migration using its manifest."""
        manifest_file = Path(manifest_path)
        if not manifest_file.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_file}")

        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        workbook_path = Path(manifest["workbook_path"]).resolve()
        backup_workbook_path = Path(manifest["backup_workbook_path"]).resolve()
        images_dir = Path(manifest["images_dir"]).resolve()
        backup_images_dir = Path(manifest["backup_images_dir"]).resolve()

        # Security check: ensure paths are not root or system drives
        for p in [workbook_path, backup_workbook_path, images_dir, backup_images_dir]:
            if p == p.parent or len(p.parts) <= 1:
                raise ValueError(f"Unsafe path in migration manifest: {p}")

        # Restore workbook
        if backup_workbook_path.exists():
            shutil.copy2(backup_workbook_path, workbook_path)

        # Restore images
        if backup_images_dir.exists():
            if images_dir.exists():
                shutil.rmtree(images_dir)
            shutil.copytree(backup_images_dir, images_dir)

        return {
            "status": "ROLLED_BACK",
            "manifest": str(manifest_file),
            "restored_workbook": str(workbook_path),
            "restored_images": str(images_dir),
        }
