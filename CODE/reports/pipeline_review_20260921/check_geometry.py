"""Bounded read-only data/geometry audit; no CNN or fitting. Outputs here only."""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT / 'CODE'))
DATA = ROOT / 'DATASET_BUILDER/4_Final_Dataset/final_linear_regression_dataset.csv'
ROWS = list(csv.DictReader(DATA.open(encoding='utf-8-sig')))
NAMES = json.loads((ROOT / 'LINEAR_REGRESSION_MODEL/models/extra_trees/feature_schema.json').read_text())['features']


def number(row, name):
    try:
        return float(row[name])
    except (ValueError, TypeError, KeyError):
        return float('nan')


def valid(row):
    return all(math.isfinite(number(row, n)) for n in NAMES + ['Actual_Count'])


def save(name, payload):
    (OUT / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def data_audit():
    good = [r for r in ROWS if valid(r)]
    ids = Counter(r['Sample_ID'].upper() for r in ROWS)
    cups = defaultdict(list)
    for r in good:
        cups[(r['Inner_Diameter_mm'], r['Container_Height_mm'])].append(number(r, 'Actual_Count'))
    suspicious = [r for r in good if number(r, 'Grain_Length_mm_Mean') > number(r, 'Inner_Diameter_mm')]
    keys = ['Sample_ID', 'Pixels_Per_mm', 'Inner_Diameter_mm', 'Container_Detected_Diam_px',
            'Grain_Length_mm_Mean', 'Grain_Width_mm_Mean', 'Grain_Volume_mm3_Mean', 'Actual_Count']
    too_few = sum(number(r, 'Whole_Grains_Count') < 8 for r in good)
    save('data_quality.json', dict(rows=len(ROWS), complete_rows=len(good),
        duplicate_ids={k: v for k, v in ids.items() if v > 1},
        duplicate_complete_ids={k: v for k, v in Counter(r['Sample_ID'].upper() for r in good).items() if v > 1},
        missing_feature_counts={n: sum(not math.isfinite(number(r, n)) for r in ROWS) for n in NAMES if any(not math.isfinite(number(r, n)) for r in ROWS)},
        incomplete_rows=[dict(sample_id=r['Sample_ID'], image_status=r['Image_Status'], whole_count=r['Whole_Grains_Count']) for r in ROWS if not valid(r)],
        cups=[dict(diameter_mm=float(k[0]), height_mm=float(k[1]), n=len(v), count_min=min(v), count_max=max(v)) for k, v in cups.items()],
        rows_with_fewer_than_8_whole_grains=too_few,
        mean_grain_length_exceeds_cup_diameter=[{k: r[k] for k in keys} for r in suspicious]))


def geometry():
    import cv2
    import numpy as np
    from modules.container_detector import detect_container_and_scale
    from modules.ellipsoid_geometry import compute_single_grain_metrics
    cv2.setNumThreads(2)
    selected = ['M014a', 'M031a', 'M056a', 'M029c', 'M037e', 'M038e']
    image_files = {p.name.upper(): p for p in (ROOT / 'DATASET_BUILDER/1_Raw_Images').rglob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png']}
    results, tiles = [], []
    for sample_id in selected:
        row = next(r for r in ROWS if r['Sample_ID'].upper() == sample_id.upper())
        r = {'sample_id': sample_id, 'stored_pixels_per_mm': number(row, 'Pixels_Per_mm')}
        start = time.perf_counter()
        try:
            img = cv2.imdecode(np.fromfile(image_files[row['Image_Filename'].upper()], dtype=np.uint8), cv2.IMREAD_COLOR)
            info = detect_container_and_scale(img, inner_diam_mm=number(row, 'Inner_Diameter_mm'),
                container_height_mm=number(row, 'Container_Height_mm'), empty_height_mm=number(row, 'Empty_Height_mm'), detect_mode='inner')
            r.update(status='returned_not_manually_verified', shape=list(img.shape),
                **{k: info[k] for k in ['pixels_per_mm', 'inner_w_px', 'outer_w_px', 'detect_type', 'outer_confidence', 'inner_confidence', 'bulk_rice_volume_mm3']})
            r['scale_ratio_current_over_stored'] = info['pixels_per_mm'] / number(row, 'Pixels_Per_mm')
            # Draw current ellipses on a compact copy only for review, not inference.
            factor = min(400 / img.shape[1], 420 / img.shape[0])
            small = cv2.resize(img, None, fx=factor, fy=factor)
            for field, color in [('outer_ellipse', (0, 160, 255)), ('inner_ellipse', (0, 255, 0))]:
                (cx, cy), (a, b), angle = info[field]
                cv2.ellipse(small, ((cx*factor, cy*factor), (a*factor, b*factor), angle), color, 2)
            tile = np.full((470, 400, 3), 240, np.uint8)
            tile[:small.shape[0], :small.shape[1]] = small
            cv2.putText(tile, f'{sample_id}: {info["pixels_per_mm"]:.2f} px/mm', (5, 440), cv2.FONT_HERSHEY_SIMPLEX, .52, (0, 0, 0), 1)
            cv2.putText(tile, f'stored: {r["stored_pixels_per_mm"]:.2f}', (5, 460), cv2.FONT_HERSHEY_SIMPLEX, .52, (0, 0, 0), 1)
            tiles.append(tile)
        except Exception as e:
            r.update(status='error', error=f'{type(e).__name__}: {e}')
        r['elapsed_seconds'] = round(time.perf_counter() - start, 3)
        results.append(r)
        save('geometry_summary.json', {'scope': 'Container stage only. NOT full image-to-count E2E.', 'results': results})
    if tiles:
        while len(tiles) % 3:
            tiles.append(np.full_like(tiles[0], 240))
        sheet = np.vstack([np.hstack(tiles[i:i+3]) for i in range(0, len(tiles), 3)])
        cv2.imencode('.jpg', sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tofile(OUT / 'container_overlays.jpg')
    # Controlled geometry: 80 px long / 24 px wide at 10 px/mm.
    mask = np.zeros((120, 120, 4), np.uint8)
    cv2.ellipse(mask, (60, 60), (40, 12), 0, 0, 360, (255, 255, 255, 255), -1)
    base = compute_single_grain_metrics(mask, 10.0)
    changed_scale = compute_single_grain_metrics(mask, 11.0)
    save('geometry_synthetic.json', {'known_ellipse_diameter_px': [80, 24], 'pixels_per_mm': 10,
         'measured': {k: base[k] for k in ['length_mm', 'width_mm', 'thickness_mm', 'volume_mm3', 'k_factor']},
         'scale_increase_10pct_volume_ratio': changed_scale['volume_mm3']/base['volume_mm3'],
         'scale_increase_10pct_count_ratio_fixed_bulk': base['volume_mm3']/changed_scale['volume_mm3'],
         'note': 'Synthetic calculation, not measured biological thickness or accuracy.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['data', 'geometry'], required=True)
    args = parser.parse_args()
    (data_audit if args.mode == 'data' else geometry)()
