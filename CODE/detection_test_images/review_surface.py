"""Reproducible, detector-only visual review on the four local fixtures."""
import sys
import json
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'CODE'))
from modules import container_detector as detector

def save(path, img):
    cv2.imencode('.jpg', img)[1].tofile(str(path))

if __name__ == '__main__':
    cv2.setNumThreads(1)
    out = Path(__file__).parent / 'surface_review'
    out.mkdir(exist_ok=True)
    summary = []
    for path in sorted(Path(__file__).parent.glob('M*.jpg')):
        img = detector._read_image(path)
        img = cv2.resize(img, (round(img.shape[1]*1600/max(img.shape[:2])), round(img.shape[0]*1600/max(img.shape[:2]))))
        rice = detector._segment_rice(img)
        x,y = rice['center']; r=rice['radius']
        roi=img[max(0,int(y-r*1.3)):int(y+r*1.3),max(0,int(x-r*1.3)):int(x+r*1.3)]
        save(out / (path.stem+'_original.jpg'),roi)
        hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        sat=cv2.GaussianBlur(hsv[:,:,1],(9,9),0)
        yy,xx=np.indices(sat.shape)
        local=(xx-x)**2+(yy-y)**2<(r*1.15)**2
        threshold,_=cv2.threshold(sat[local],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        panels=[]
        for t in [threshold,threshold+15,threshold+30]:
            mask=((sat>t)&local).astype('uint8')*255
            mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((11,11),np.uint8))
            contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            contour=max(contours,key=cv2.contourArea)
            hull=cv2.convexHull(contour)
            ell=cv2.fitEllipse(hull)
            show=img.copy()
            cv2.drawContours(show,[contour],-1,(0,0,255),2)
            cv2.ellipse(show,ell,(0,255,0),2)
            show=show[max(0,int(y-r*1.3)):int(y+r*1.3),max(0,int(x-r*1.3)):int(x+r*1.3)]
            show=cv2.resize(show,(450,450))
            cv2.putText(show,f'S > {t:.0f}',(10,25),0,.7,(0,255,0),2)
            panels.append(show)
            print('threshold',t,'ellipse',ell)
        save(out/(path.stem+'_thresholds.jpg'),np.hstack(panels))
        print(path.stem, rice['center'],r)
        full = detector._read_image(path)
        results = {}
        panels = []
        for method in ('radial', 'adaptive'):
            result = detector.detect_container_and_scale(
                full, 22.8, 37.2, 9.7, surface_method=method, on_failure='dict')
            if result.get('status') != 'ok':
                raise AssertionError((path.name, method, result))
            results[method] = result
            overlay = cv2.resize(result['overlay_bgr'], (img.shape[1], img.shape[0]))
            view = overlay[max(0,int(y-r*1.3)):int(y+r*1.3),max(0,int(x-r*1.3)):int(x+r*1.3)]
            view = cv2.resize(view, (520,520))
            crop = result['cropped_bgr']
            ch,cw = crop.shape[:2]
            resized = cv2.resize(crop,(round(cw*480/max(ch,cw)),round(ch*480/max(ch,cw))))
            canvas = np.zeros((520,520,3),np.uint8)
            hh,ww=resized.shape[:2]
            canvas[(520-hh)//2:(520-hh)//2+hh,(520-ww)//2:(520-ww)//2+ww]=resized
            cv2.putText(view,method,(15,30),0,0.85,(0,255,0),2)
            panels.append(np.vstack([view,canvas]))
        save(out/(path.stem+'_comparison.jpg'),np.hstack(panels))
        result=results['adaptive']
        save(out/(path.stem+'_crop.jpg'),result['cropped_bgr'])
        x1,y1,x2,y2=result['crop_bbox']
        assert np.array_equal(result['cropped_bgr'],result['surface_isolated_bgr'][y1:y2,x1:x2])
        assert not np.any(result['surface_isolated_bgr'][result['surface_mask']==0])
        row={'sample':path.stem,'old_inner_px':results['radial']['inner_w_px'],
             'new_inner_px':result['inner_w_px'],'new_center':result['center'],
             'pixels_per_mm':result['pixels_per_mm'],'diagnostics':result['surface_diagnostics']}
        print(json.dumps(row))
        summary.append(row)
        del full, results, result
    (out/'measurements.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
