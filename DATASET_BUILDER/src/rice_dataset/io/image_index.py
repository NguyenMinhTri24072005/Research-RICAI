"""Image indexing and path resolution with strict duplicate detection."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class ImageIndex:
    """Index raw images and validate the canonical flat ``M####`` layout."""

    def __init__(self, root_dir: Path | str, recursive: bool = False) -> None:
        self.root_dir = Path(root_dir)
        self.recursive = recursive
        self._index: Dict[str, List[Path]] = {}
        self._scan()

    def _scan(self) -> None:
        self._index.clear()
        if not self.root_dir.exists():
            return

        pattern = "**/*" if self.recursive else "*"
        for path in self.root_dir.glob(pattern):
            if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                stem = path.stem.strip().upper()
                if stem not in self._index:
                    self._index[stem] = []
                self._index[stem].append(path)

    def refresh(self) -> None:
        self._scan()

    def get_all_stems(self) -> Set[str]:
        return set(self._index.keys())

    def get_total_image_count(self) -> int:
        return sum(len(paths) for paths in self._index.values())

    def get_duplicate_stems(self) -> Dict[str, List[Path]]:
        """Return stems that map to more than one file."""
        return {stem: paths for stem, paths in self._index.items() if len(paths) > 1}

    def validate_flat_layout(self, prefix: str = "M", digits: int = 4) -> List[str]:
        """Return layout errors for the canonical one-folder raw-image contract."""
        issues: List[str] = []
        expected = re.compile(rf"^{re.escape(prefix)}\d{{{digits}}}$", re.IGNORECASE)

        nested_images = sorted(
            path
            for path in self.root_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
            and path.parent != self.root_dir
        ) if self.root_dir.exists() else []
        if nested_images:
            preview = ", ".join(str(path.relative_to(self.root_dir)) for path in nested_images[:5])
            issues.append(
                f"Found {len(nested_images)} nested raw image(s); all images must be directly under "
                f"'{self.root_dir}'. Examples: {preview}"
            )

        invalid_names = sorted(
            path.name
            for paths in self._index.values()
            for path in paths
            if expected.fullmatch(path.stem) is None
        )
        if invalid_names:
            issues.append(
                f"Found {len(invalid_names)} image filename(s) outside the {prefix}{'0' * digits} contract: "
                + ", ".join(invalid_names[:10])
            )

        duplicate_stems = self.get_duplicate_stems()
        if duplicate_stems:
            issues.append(
                "Duplicate image stems found: " + ", ".join(sorted(duplicate_stems)[:10])
            )

        return issues

    def resolve_image(self, sample_id: str) -> Tuple[Optional[Path], Optional[str]]:
        """Resolve an image path for a given sample_id.

        Returns:
            Tuple of (path, error_code)
            error_code is None on success, 'DUPLICATE_IMAGE' if multiple files match,
            or 'MISSING_IMAGE' if not found.
        """
        stem = sample_id.strip().upper()
        matches = self._index.get(stem, [])

        if len(matches) == 0:
            return None, "MISSING_IMAGE"
        if len(matches) > 1:
            return None, "DUPLICATE_IMAGE"
        return matches[0], None


def validate_flat_inventory(
    records: Sequence[Any],
    images_dir: Path | str,
    prefix: str = "M",
    digits: int = 4,
    require_exact_match: bool = True,
) -> Dict[str, int]:
    """Validate IDs and the one-row/one-image canonical flat inventory."""
    index = ImageIndex(images_dir, recursive=False)
    issues = index.validate_flat_layout(prefix=prefix, digits=digits)
    expected_id = re.compile(rf"^{re.escape(prefix)}\d{{{digits}}}$", re.IGNORECASE)

    record_ids = [str(record.sample_id).strip().upper() for record in records]
    invalid_record_ids = sorted({
        sample_id for sample_id in record_ids if expected_id.fullmatch(sample_id) is None
    })
    if invalid_record_ids:
        issues.append(
            f"Workbook contains {len(invalid_record_ids)} Sample_ID value(s) outside the "
            f"{prefix}{'0' * digits} contract: " + ", ".join(invalid_record_ids[:10])
        )

    record_counts: Dict[str, int] = {}
    for sample_id in record_ids:
        record_counts[sample_id] = record_counts.get(sample_id, 0) + 1
    duplicate_record_ids = sorted(
        sample_id for sample_id, count in record_counts.items() if count > 1
    )
    if duplicate_record_ids:
        issues.append(
            "Workbook contains duplicate Sample_ID value(s): "
            + ", ".join(duplicate_record_ids[:10])
        )

    image_ids = index.get_all_stems()
    record_id_set = set(record_ids)
    if require_exact_match:
        missing_images = sorted(record_id_set - image_ids)
        unreferenced_images = sorted(image_ids - record_id_set)
        if missing_images:
            issues.append(
                f"{len(missing_images)} workbook row(s) have no matching flat image: "
                + ", ".join(missing_images[:10])
            )
        if unreferenced_images:
            issues.append(
                f"{len(unreferenced_images)} flat image(s) have no matching workbook row: "
                + ", ".join(unreferenced_images[:10])
            )

    if issues:
        raise ValueError("Raw input inventory is invalid:\n - " + "\n - ".join(issues))

    return {
        "record_count": len(record_ids),
        "unique_record_count": len(record_id_set),
        "image_count": index.get_total_image_count(),
    }
