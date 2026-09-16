from __future__ import annotations

import argparse
import string
import sys
import uuid
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg"}


def is_sample_folder(path: Path) -> bool:
    return path.is_dir() and path.name.startswith("M") and path.name[1:].isdigit()


def collect_renames(root: Path) -> list[tuple[Path, Path]]:
    renames: list[tuple[Path, Path]] = []

    for folder in sorted((p for p in root.iterdir() if is_sample_folder(p)), key=lambda p: p.name):
        images = sorted(
            (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
            key=lambda p: p.name.lower(),
        )

        if len(images) > len(string.ascii_uppercase):
            raise ValueError(
                f"{folder.name} has {len(images)} images, but this script supports at most 26 images per folder."
            )

        for index, source in enumerate(images):
            target = folder / f"{folder.name}{string.ascii_uppercase[index]}.jpg"
            renames.append((source, target))

    return renames


def validate_targets(renames: list[tuple[Path, Path]]) -> None:
    sources = {source.resolve() for source, _ in renames}
    targets_seen: set[Path] = set()

    for source, target in renames:
        resolved_target = target.resolve()

        if resolved_target in targets_seen:
            raise ValueError(f"Duplicate target name detected: {target}")

        if target.exists() and resolved_target not in sources:
            raise FileExistsError(f"Target already exists and is not part of this rename batch: {target}")

        targets_seen.add(resolved_target)


def apply_renames(renames: list[tuple[Path, Path]]) -> None:
    temp_pairs: list[tuple[Path, Path]] = []

    for source, _ in renames:
        temp = source.with_name(f".rename_tmp_{uuid.uuid4().hex}{source.suffix}")
        source.rename(temp)
        temp_pairs.append((temp, source))

    try:
        for (temp, _), (_, target) in zip(temp_pairs, renames):
            temp.rename(target)
    except Exception:
        for temp, original in reversed(temp_pairs):
            if temp.exists() and not original.exists():
                temp.rename(original)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rename JPG images inside M### folders to M###A.jpg, M###B.jpg, ..."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned renames without changing any files.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    renames = collect_renames(root)

    if not renames:
        print("No JPG/JPEG images found in M### folders.")
        return 0

    validate_targets(renames)

    for source, target in renames:
        if source.name != target.name:
            print(f"{source.relative_to(root)} -> {target.relative_to(root)}")

    if args.dry_run:
        print(f"\nDry run complete. Planned rename count: {len(renames)}")
        return 0

    apply_renames(renames)
    print(f"\nDone. Renamed {len(renames)} image(s).")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
