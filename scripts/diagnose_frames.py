"""Feature evidence using already decoded frames (no video backend required)."""
import json
from pathlib import Path
import cv2
import numpy as np

root=Path('outputs/inspection')
files=sorted(root.glob('frame_[0-9][0-9][0-9][0-9].jpg'))
sift=cv2.SIFT_create(nfeatures=4000)
features=[sift.detectAndCompute(cv2.imread(str(p),0),None) for p in files]
pairs=[]
for i in range(len(files)-1):
    ka,da=features[i];kb,db=features[i+1]
    good=[a for a,b in cv2.BFMatcher().knnMatch(da,db,k=2) if a.distance<.72*b.distance]
    a=np.float32([ka[m.queryIdx].pt for m in good]);b=np.float32([kb[m.trainIdx].pt for m in good])
    h,mask=cv2.findHomography(a,b,cv2.USAC_MAGSAC,2.)
    err=np.linalg.norm(cv2.perspectiveTransform(a[None],h)[0]-b,axis=1)
    pairs.append(dict(a=i+1,b=i+2,matches=len(good),homography_fraction=float(mask.mean()),median_residual_px=float(np.median(err))))
report=dict(upright_resolution=[320,568],duration_seconds=17.17,fps=30,sampled_fps=2,pairs=pairs,
    metric_scale='unobserved',note='Homography support is consistent with rotation or planar surfaces and is not a depth validation.')
(root/'video_analysis.json').write_text(json.dumps(report,indent=2))
print('median homography support',np.median([p['homography_fraction'] for p in pairs]))
