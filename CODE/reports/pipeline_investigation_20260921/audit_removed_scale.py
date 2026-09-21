"""Recheck current main detector against rows retained only in the 285-row XLSX."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import cv2
import pandas as pd

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
sys.path.insert(0,str(ROOT/'CODE'))
from modules.container_detector import detect_container_and_scale
rows=pd.read_csv(OUT/'historical_removed_rows.csv')
rows=rows[rows.Sample_ID.str.lower().isin(['m029c','m037e','m038e'])]
images={p.name.upper():p for p in (ROOT/'DATASET_BUILDER/1_Raw_Images').rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png'}}
results=[]
for src in rows.itertuples(index=False):
    path=images[str(src.Image_Filename).upper()]
    t0=time.perf_counter()
    info=detect_container_and_scale(path,float(src.Inner_Diameter_mm),float(src.Container_Height_mm),float(src.Empty_Height_mm),detect_mode='inner')
    ratio=float(info['pixels_per_mm']/src.Pixels_Per_mm)
    rec={'sample_id':src.Sample_ID,'historical_pixels_per_mm':float(src.Pixels_Per_mm),'current_pixels_per_mm':float(info['pixels_per_mm']),'ratio_current_over_historical':ratio,'ratio_cubed':ratio**3,'outer_confidence':info.get('outer_confidence'),'inner_confidence':info.get('inner_confidence'),'detect_type':info.get('detect_type'),'elapsed_seconds':time.perf_counter()-t0}
    results.append(rec)
    overlay=info['overlay_bgr']
    if max(overlay.shape[:2]) > 1600:
        factor=1600.0/max(overlay.shape[:2])
        overlay=cv2.resize(overlay,None,fx=factor,fy=factor,interpolation=cv2.INTER_AREA)
    ok,encoded=cv2.imencode('.jpg',overlay,[cv2.IMWRITE_JPEG_QUALITY,82])
    if ok: encoded.tofile(OUT/f"removed_scale_overlay_{src.Sample_ID}.jpg")
    print(json.dumps(rec,ensure_ascii=False),flush=True)
summary={'source':'DATASET_BUILDER/4_Final_Dataset/final_linear_regression_dataset.xlsx via historical_removed_rows.csv','implementation':'CODE/modules/container_detector.py','results':results}
text=json.dumps(summary,ensure_ascii=False,indent=2)
(OUT/'removed_scale_audit.json').write_text(text,encoding='utf-8')
(OUT/'removed_scale_audit.log').write_text(text,encoding='utf-8')