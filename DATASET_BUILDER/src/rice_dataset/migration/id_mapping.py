"""Deterministic legacy-to-new Sample ID mapping and natural sorting."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SampleMapping:
    sample_id: str                   # New flat ID, e.g. M0001
    original_sample_id: str          # Legacy ID, e.g. M001A
    physical_sample_id: str          # Physical group ID, e.g. M001
    source_image_path: Optional[Path]# Path to source image file
    target_image_filename: str       # e.g. M0001.jpg

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Sample_ID": self.sample_id,
            "Original_Sample_ID": self.original_sample_id,
            "Physical_Sample_ID": self.physical_sample_id,
            "Target_Image_Filename": self.target_image_filename,
            "Source_Image_Path": str(self.source_image_path) if self.source_image_path else "",
            "Source_Image_Filename": self.source_image_path.name if self.source_image_path else "",
        }


def parse_natural_key(sample_id: str) -> Tuple[str, int, str, str]:
    """Parse legacy ID into natural sortable components.

    Example:
        'M001A' -> ('M', 1, 'A', 'M001A')
        'M057F' -> ('M', 57, 'F', 'M057F')
        'M0001' -> ('M', 1, '', 'M0001')
    """
    clean = sample_id.strip().upper()
    match = re.match(r"^([A-Za-z]+)(\d+)([A-Za-z]*)$", clean)
    if match:
        prefix, num_str, suffix = match.groups()
        return (prefix, int(num_str), suffix, clean)
    return (clean, 0, "", clean)


def derive_physical_sample_id(sample_id: str) -> str:
    """Derive physical sample ID grouping key.

    Example:
        'M001A' -> 'M001'
        'M057B' -> 'M057'
        'M0001' -> 'M0001' (if already flat 4-digit)
    """
    clean = sample_id.strip().upper()
    match = re.match(r"^([A-Za-z]+\d+)[A-Za-z]*$", clean)
    if match:
        return match.group(1)
    return clean


def build_id_mappings(
    legacy_ids: List[str],
    image_map: Optional[Dict[str, Path]] = None,
    prefix: str = "M",
    digits: int = 4,
    start: int = 1,
) -> List[SampleMapping]:
    """Build deterministic sequential mappings sorted naturally by legacy ID."""
    image_map = image_map or {}

    # Unique normalized legacy IDs sorted naturally
    unique_legacy_ids = sorted(list(set(id_str.strip().upper() for id_str in legacy_ids if id_str.strip())), key=parse_natural_key)

    mappings: List[SampleMapping] = []
    current_num = start

    for legacy_id in unique_legacy_ids:
        new_id = f"{prefix}{current_num:0{digits}d}"
        phys_id = derive_physical_sample_id(legacy_id)

        source_img = image_map.get(legacy_id)
        ext = source_img.suffix.lower() if source_img else ".jpg"
        target_img_name = f"{new_id}{ext}"

        mappings.append(
            SampleMapping(
                sample_id=new_id,
                original_sample_id=legacy_id,
                physical_sample_id=phys_id,
                source_image_path=source_img,
                target_image_filename=target_img_name,
            )
        )
        current_num += 1

    return mappings
