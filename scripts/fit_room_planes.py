"""Estimate relative room envelope planes from recovered source surfaces.

All meter values inherit the unmeasured tabletop scale assumption.
"""
import argparse,json
from pathlib import Path
import numpy as np
from process_geometry import plane_ransac
p=argparse.ArgumentParser();p.add_argument('--geometry',type=Path,default=Path('outputs/geometry_full'));p.add_argument('--report',type=Path,default=Path('reports/room_plane_fit.json'));a=p.parse_args()
d=np.load(a.geometry/'aligned_geometry.npz');points=d['points'];rgb=d['colors'].astype(float);valid=d['valid'];dx=points[:,1:-1,2:]-points[:,1:-1,:-2];dy=points[:,2:,1:-1]-points[:,:-2,1:-1];normal=np.cross(dx,dy);normal/=np.maximum(np.linalg.norm(normal,axis=-1,keepdims=True),1e-9);nx=np.zeros(valid.shape);ny=np.zeros(valid.shape);nz=np.zeros(valid.shape);nx[:,1:-1,1:-1]=abs(normal[...,0]);ny[:,1:-1,1:-1]=abs(normal[...,1]);nz[:,1:-1,1:-1]=abs(normal[...,2]);white=(rgb.mean(-1)>150)&(rgb.max(-1)-rgb.min(-1)<50)
regions=dict(ceiling=valid&white&(points[...,2]>2.3)&(points[...,2]<3.)&(points[...,1]<4.7)&(nz>.90),right_wall=valid&white&(points[...,0]>5.5)&(points[...,2]>1.3)&(points[...,2]<2.5)&(nx>.90),left_wall=valid&white&(points[...,0]<.3)&(points[...,2]>1.3)&(points[...,2]<2.5)&(nx>.90),window_wall=valid&white&(points[...,1]>4.6)&(points[...,1]<5.3)&(points[...,2]>1.2)&(points[...,2]<2.5)&(ny>.90))
report=dict(scale_status='Assumed 0.74 m tabletop anchor; no real measurement',planes={})
for name,mask in regions.items():
 axis=2 if name=='ceiling' else 1 if name=='window_wall' else 0;hint=np.eye(3)[axis]
 try:
  n,offset,quality=plane_ransac(points[mask],hint,.015);reference=np.array([3.,2.5,1.5]);coordinate=float(-(offset+sum(n[j]*reference[j] for j in range(3) if j!=axis))/n[axis]);report['planes'][name]=dict(normal=n.tolist(),offset=offset,axis_coordinate_at_reference_m=coordinate,reference_xyz=reference.tolist(),quality=quality)
 except Exception as error:report['planes'][name]=dict(error=str(error))
window_region=valid&(points[...,1]>4.6)&(points[...,1]<5.3)&(points[...,2]>.8)&(points[...,2]<2.6)&(rgb.mean(-1)>140)&(rgb.max(-1)-rgb.min(-1)<60)
window_max=float(np.percentile(points[...,1][window_region],95));right=report['planes']['right_wall'];ceiling=report['planes']['ceiling']
if right.get('quality',{}).get('inlier_fraction',0)>.6 and ceiling.get('quality',{}).get('inlier_fraction',0)>.5:
    shift=np.array([.20,.60,0.]);dimensions=np.ceil((np.array([right['axis_coordinate_at_reference_m']+.02,window_max,ceiling['axis_coordinate_at_reference_m']+.04])+shift)*100)/100
    report['recommended_rectangular_envelope']=dict(dimensions_m=dimensions.tolist(),source_geometry_shift_m=shift.tolist(),window_structural_y_95_percentile_dense_m=window_max,derivation='Representative right-wall plane plus 2 cm, observed window structural depth 95th percentile, and ceiling plane plus 4 cm; rounded outward to centimeters',limitations=['Axis-aligned envelope approximates recovered wall slopes/recesses.','Window/left-wall individual plane inlier fractions are below 50%; this is an observed extent bound, not a surveyed rectangular room.','All dimensions inherit the assumed table-height scale.'])
a.report.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
