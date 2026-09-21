"""
Audit dataset quality and evaluate geometry fix impact.
Outputs: dataset_audit_manifest.json, geometry_fix_evaluation.json
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATA = ROOT / 'DATASET_BUILDER/4_Final_Dataset/final_linear_regression_dataset.csv'
BUNDLE = ROOT / 'LINEAR_REGRESSION_MODEL/models/extra_trees'
OUT = ROOT / 'CODE/reports/pipeline_review_20260921'

def audit():
    df = pd.read_csv(DATA, encoding='utf-8-sig')
    names = json.loads((BUNDLE / 'feature_schema.json').read_text())['features']
    
    flags = []
    
    for idx, row in df.iterrows():
        reasons = []
        
        # 1. Missing image
        if str(row.get('Image_Status', '')).upper() == 'MISSING':
            reasons.append('MISSING_IMAGE')
        
        # 2. Scale corrupted (px/mm < 15)
        try:
            ppm = float(row['Pixels_Per_mm'])
            if math.isfinite(ppm) and ppm < 15:
                reasons.append(f'SCALE_CORRUPTED (px/mm={ppm:.2f}, expected ~70-80)')
        except (ValueError, TypeError):
            pass
        
        # 3. Grain length exceeds cup diameter (physically impossible)
        try:
            grain_len = float(row['Grain_Length_mm_Mean'])
            cup_diam = float(row['Inner_Diameter_mm'])
            if math.isfinite(grain_len) and math.isfinite(cup_diam) and grain_len > cup_diam:
                reasons.append(f'GRAIN_EXCEEDS_CUP (grain={grain_len:.1f}mm > cup={cup_diam}mm)')
        except (ValueError, TypeError):
            pass
        
        # 4. Duplicate ID
        sid = str(row['Sample_ID']).upper()
        dup_count = df['Sample_ID'].str.upper().eq(sid).sum()
        if dup_count > 1:
            reasons.append(f'DUPLICATE_ID (appears {dup_count}x)')
        
        # 5. Incomplete features
        valid = True
        for n in names + ['Actual_Count']:
            try:
                v = float(row[n])
                if not math.isfinite(v):
                    valid = False
                    break
            except (ValueError, TypeError):
                valid = False
                break
        if not valid:
            reasons.append('INCOMPLETE_FEATURES')
        
        if reasons:
            flags.append({
                'sample_id': row['Sample_ID'],
                'row_index': int(idx),
                'reasons': reasons,
                'action': 'REMOVE' if any(r.startswith(('SCALE_CORRUPTED', 'DUPLICATE')) for r in reasons) else 'FLAG'
            })
    
    # Summary
    remove_count = sum(1 for f in flags if f['action'] == 'REMOVE')
    flag_count = sum(1 for f in flags if f['action'] == 'FLAG')
    
    manifest = {
        'dataset': str(DATA.relative_to(ROOT)),
        'total_rows': len(df),
        'flagged_rows': len(flags),
        'rows_to_remove': remove_count,
        'rows_to_flag': flag_count,
        'clean_rows': len(df) - len(flags),
        'details': flags
    }
    
    out_path = OUT / 'dataset_audit_manifest.json'
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    
    print(f"Dataset Audit Complete:")
    print(f"  Total rows: {len(df)}")
    print(f"  Rows to REMOVE: {remove_count}")
    print(f"    - Scale corrupted: {sum(1 for f in flags if any('SCALE_CORRUPTED' in r for r in f['reasons']))}")
    print(f"    - Duplicate: {sum(1 for f in flags if any('DUPLICATE' in r for r in f['reasons']))}")
    print(f"  Rows to FLAG: {flag_count}")
    print(f"    - Missing image: {sum(1 for f in flags if any('MISSING' in r for r in f['reasons']))}")
    print(f"  Clean rows remaining: {len(df) - remove_count}")
    print(f"  Manifest saved to: {out_path}")
    
    return manifest


def evaluate_geometry_fix():
    """Compare physical formula MAE before/after thickness fix."""
    df = pd.read_csv(DATA, encoding='utf-8-sig')
    names = json.loads((BUNDLE / 'feature_schema.json').read_text())['features']
    
    # Filter valid rows with good scale
    valid_mask = np.isfinite(df[names + ['Actual_Count']].to_numpy(dtype=float)).all(axis=1)
    df_valid = df[valid_mask & (df['Pixels_Per_mm'] >= 15)].copy()
    y = df_valid['Actual_Count'].to_numpy()
    
    print(f"\nGeometry Fix Evaluation ({len(df_valid)} clean rows):")
    print("=" * 60)
    
    # OLD formula: c = 0.85 * a (length-based)
    # thickness_old = 0.85 * length, volume_old = 4/3 * pi * (L/2) * (W/2) * (0.85*L/2)
    # In stored data: Grain_Volume_mm3_Mean already uses old formula
    old_vol = df_valid['Grain_Volume_mm3_Mean'].to_numpy()
    
    # NEW formula: c = 0.80 * b (width-based)
    # volume_new = 4/3 * pi * (L/2) * (W/2) * (0.80*W/2) 
    # = old_volume * (0.80 * W) / (0.85 * L)
    L = df_valid['Grain_Length_mm_Mean'].to_numpy()
    W = df_valid['Grain_Width_mm_Mean'].to_numpy()
    T_old = df_valid['Grain_Thickness_mm_Mean'].to_numpy()
    
    # Recalculate with new formula
    T_new = 0.80 * W  # thickness = 0.80 * width
    new_vol = old_vol * (T_new / T_old)  # Volume scales linearly with c
    
    bulk_vol = df_valid['Bulk_Rice_Volume_mm3'].to_numpy()
    
    results = {}
    for phi_label, phi in [('phi_082', 0.82), ('phi_062', 0.62)]:
        old_count = np.rint(bulk_vol * phi / old_vol)
        new_count = np.rint(bulk_vol * phi / new_vol)
        
        old_mae = float(np.mean(np.abs(old_count - y)))
        new_mae = float(np.mean(np.abs(new_count - y)))
        old_bias = float(np.mean(old_count - y))
        new_bias = float(np.mean(new_count - y))
        
        results[phi_label] = {
            'old_mae': round(old_mae, 2),
            'new_mae': round(new_mae, 2),
            'improvement_pct': round((old_mae - new_mae) / old_mae * 100, 1),
            'old_bias': round(old_bias, 2),
            'new_bias': round(new_bias, 2),
        }
        
        print(f"\n  Packing fraction = {phi}:")
        print(f"    OLD (c=0.85*L): MAE={old_mae:.2f}, Bias={old_bias:+.2f}")
        print(f"    NEW (c=0.80*W): MAE={new_mae:.2f}, Bias={new_bias:+.2f}")
        print(f"    Improvement: {(old_mae - new_mae) / old_mae * 100:+.1f}%")
    
    # Sanity check: thickness < width in new formula
    print(f"\n  Sanity check (new formula):")
    print(f"    Thickness < Width: {(T_new < W).all()} (100% of rows)")
    print(f"    Thickness/Width ratio: {(T_new / W).mean():.2f} (should be 0.80)")
    print(f"    Thickness/Length ratio: {(T_new / L).mean():.4f} (should be << 1)")
    
    # Volume comparison
    print(f"\n  Volume change:")
    ratio = new_vol / old_vol
    print(f"    New/Old volume ratio: {ratio.mean():.4f} (median: {np.median(ratio):.4f})")
    print(f"    Old mean volume: {old_vol.mean():.2f} mm³")
    print(f"    New mean volume: {new_vol.mean():.2f} mm³")
    
    eval_result = {
        'n_clean_rows': len(df_valid),
        'formula_change': 'c = 0.85 * a_px (length) -> c = 0.80 * b_px (width)',
        'scores': results,
        'volume_ratio_mean': round(float(ratio.mean()), 4),
        'sanity_thickness_lt_width': bool((T_new < W).all()),
    }
    
    out_path = OUT / 'geometry_fix_evaluation.json'
    out_path.write_text(json.dumps(eval_result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\n  Evaluation saved to: {out_path}")
    
    return eval_result


if __name__ == '__main__':
    audit()
    evaluate_geometry_fix()
