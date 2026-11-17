
"""Experimental board-size-independent PCB preprocessing.

Requires: pip install opencv-python numpy
Example: python pcb_flexible_preprocess.py --input "new.jpg" --output "processed" --debug

Does not modify inputs. Does not perform reference-to-reference defect registration.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import cv2
import numpy as np

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def ordered_quad(pts):
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def detect_board(img, diagnostics=None):
    """Try several edge thresholds; reject weak or ambiguous contours."""
    h, w = img.shape[:2]
    scale = min(1., 900 / max(h, w))
    sm = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1 else img.copy()
    gray = cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (5, 5), 0)
    area = gray.size
    k = max(9, int(min(gray.shape) * .025) | 1)
    for low, high, kernel in [(35,100,k),(20,60,k),(15,45,k),(20,60,2*k+1)]:
        edges = cv2.Canny(smooth, low, high)
        binary = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((kernel,kernel),np.uint8))
        if diagnostics is not None:
            diagnostics[f'edges_{low}_{high}_{kernel}'] = edges
            diagnostics[f'closed_{low}_{high}_{kernel}'] = binary
        contours,_ = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        candidates=[]
        for cnt in contours:
            ca=cv2.contourArea(cnt)
            if not .07*area <= ca <= .88*area: continue
            rw,rh=cv2.minAreaRect(cnt)[1]
            if min(rw,rh)<.12*min(gray.shape): continue
            rectangularity=ca/max(rw*rh,1)
            if rectangularity < .53: continue
            hull=cv2.convexHull(cnt)
            for eps in (.02,.025,.035,.045):
                poly=cv2.approxPolyDP(hull,eps*cv2.arcLength(hull,True),True)
                if len(poly)==4 and cv2.isContourConvex(poly): break
            else: continue
            quad=ordered_quad(poly.reshape(4,2).astype(np.float32)/scale)
            if len(np.unique(quad,axis=0))!=4: continue
            if np.min(np.linalg.norm(np.roll(quad,-1,axis=0)-quad,axis=1))<40: continue
            if cv2.contourArea(quad)*scale**2 < .07*area: continue
            candidates.append((ca*rectangularity,quad,rectangularity))
        if candidates:
            _,quad,rectangularity=max(candidates,key=lambda x:x[0])
            return quad,binary,{'rectangularity':round(float(rectangularity),3),'detection_method':f'Canny {low}/{high}; kernel {kernel}'}
    raise ValueError('No reliable PCB quadrilateral found; inspect diagnostic masks')


def preprocess(img, margin=.20, output_size=1024, diagnostics=None):
    quad, binary, metrics = detect_board(img, diagnostics)
    tl,tr,br,bl = quad
    bw = int(round(max(np.linalg.norm(tr-tl), np.linalg.norm(br-bl))))
    bh = int(round(max(np.linalg.norm(bl-tl), np.linalg.norm(br-tr))))
    if bw < 50 or bh < 50 or bw * bh > 40_000_000:
        raise ValueError('Implausible detected board size')
    pad = int(round(max(bw,bh)*margin))
    dst = np.float32([[pad,pad],[pad+bw-1,pad],
                      [pad+bw-1,pad+bh-1],[pad,pad+bh-1]])
    H = cv2.getPerspectiveTransform(quad,dst)
    canvas_size=(bw+2*pad,bh+2*pad)
    warped=cv2.warpPerspective(img,H,canvas_size,flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0))
    valid=cv2.warpPerspective(np.full(img.shape[:2],255,np.uint8),H,canvas_size,
                              flags=cv2.INTER_NEAREST,borderValue=0)
    # Letterbox rather than stretch to avoid distorting board geometry.
    cw,ch=canvas_size
    factor=min(output_size/cw,output_size/ch)
    rw,rh=max(1,round(cw*factor)),max(1,round(ch*factor))
    interp=cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR
    resized=cv2.resize(warped,(rw,rh),interpolation=interp)
    mask=cv2.resize(valid,(rw,rh),interpolation=cv2.INTER_NEAREST)
    result=np.zeros((output_size,output_size,3),np.uint8)
    result_mask=np.zeros((output_size,output_size),np.uint8)
    ox,oy=(output_size-rw)//2,(output_size-rh)//2
    result[oy:oy+rh,ox:ox+rw]=resized
    result_mask[oy:oy+rh,ox:ox+rw]=mask
    overlay=img.copy()
    cv2.polylines(overlay,[quad.astype(np.int32)],True,(0,255,0),5)
    for idx,(x,y) in enumerate(quad.astype(int)):
        cv2.circle(overlay,(x,y),9,(0,0,255),-1)
        cv2.putText(overlay,str(idx+1),(x+10,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    metrics.update({'board_width_px':bw,'board_height_px':bh,
                    'valid_fraction_inside_letterbox':round(float((mask>0).mean()),4),
                    'note':'Black regions may be missing camera pixels or letterbox padding'})
    return result,result_mask,overlay,metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,help='Image file or directory of images')
    p.add_argument('--output',required=True,help='NEW output directory; must not exist')
    p.add_argument('--size',type=int,default=1024)
    p.add_argument('--margin',type=float,default=.10)
    p.add_argument('--debug',action='store_true')
    args=p.parse_args()
    src=Path(args.input).resolve(); dest=Path(args.output).resolve()
    if not src.exists(): p.error('Input does not exist')
    if dest.exists(): p.error('Output directory already exists; choose a NEW directory')
    if src.is_dir() and (src==dest or src in dest.parents): p.error('Output cannot be inside input directory')
    if not (0 <= args.margin <= .35): p.error('margin must be between 0 and .35')
    if args.size < 128: p.error('size must be >=128')
    files=[src] if src.is_file() else sorted(x for x in src.rglob('*') if x.suffix.lower() in EXTENSIONS)
    dest.mkdir(parents=True)
    records=[]
    for path in files:
        relative=Path(path.name) if src.is_file() else path.relative_to(src)
        diagnostics = {} if args.debug else None
        base=dest/relative.parent/relative.stem
        try:
            img=cv2.imread(str(path),cv2.IMREAD_COLOR)
            if img is None: raise ValueError('Unable to decode image')
            result,mask,overlay,metrics=preprocess(img,args.margin,args.size,diagnostics)
            base=dest/relative.parent/relative.stem
            base.parent.mkdir(parents=True,exist_ok=True)
            outputs=[(base.with_name(base.name+'_processed.png'),result),
                     (base.with_name(base.name+'_valid_mask.png'),mask)]
            if args.debug: outputs.append((base.with_name(base.name+'_detection.png'),overlay))
            if any(q.exists() for q,_ in outputs): raise FileExistsError('Output filename collision')
            for q,array in outputs:
                if not cv2.imwrite(str(q),array): raise IOError(f'Unable to save {q}')
            records.append({'input':str(relative),'status':'OK','reason':'',**metrics})
            print('OK',relative,metrics['board_width_px'],metrics['board_height_px'])
        except Exception as exc:
            records.append({'input':str(relative),'status':'FAILED','reason':str(exc)})
            print('FAILED',relative,exc)
        finally:
            if diagnostics:
                base.parent.mkdir(parents=True,exist_ok=True)
                for name,arr in diagnostics.items():
                    q=base.with_name(base.name+"_"+name+".png")
                    if not q.exists() and not cv2.imwrite(str(q),arr):
                        print("WARNING: unable to save",q)
    columns=['input','status','reason','board_width_px','board_height_px','detection_method','rectangularity','valid_fraction_inside_letterbox','note']
    with (dest/'report.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(records)
    print(f'Processed {sum(r["status"]=="OK" for r in records)}/{len(records)}; report: {dest/"report.csv"}')

if __name__=='__main__': main()
