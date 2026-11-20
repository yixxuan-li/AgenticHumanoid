"""Fit a native Atlas floor and scale from explicitly supplied correspondences.

Input JSON: floor_points (N x 3 native XYZ), anchor.points (2 x 3 native XYZ),
anchor.meters, anchor.measured (boolean), anchor.description. No scene priors.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
config=json.loads(a.config.read_text());floor=np.asarray(config['floor_points'],float);anchor=config['anchor'];pair=np.asarray(anchor['points'],float);meters=float(anchor['meters'])
if floor.ndim!=2 or floor.shape[1]!=3 or len(floor)<3 or pair.shape!=(2,3) or not np.isfinite(floor).all() or not np.isfinite(pair).all() or not np.isfinite(meters) or meters<=0:raise ValueError('Supply finite floor points and a positive metric anchor')
if not isinstance(anchor['measured'],bool):raise ValueError('State explicitly whether the anchor was measured')
center=floor.mean(0);_,singular,vh=np.linalg.svd(floor-center)
if singular[1]<1e-8:raise ValueError('Floor points must span a plane')
normal=vh[-1]
if normal[1]<0:normal=-normal
rotation=Rotation.align_vectors([[0,1,0]],[normal])[0].as_matrix()
floor_y=float((rotation@center)[1]);length=float(np.linalg.norm(pair[1]-pair[0]))
if length<1e-8:raise ValueError('Anchor endpoints must be distinct')
scale=meters/length;residual=(floor-center)@normal
report=dict(scale=scale,floor_y=floor_y,floor_rotation=rotation.tolist(),measured=anchor['measured'],anchor=anchor,floor_point_count=len(floor),floor_rms_native=float(np.sqrt(np.mean(residual**2))),scale_status='measured_anchor' if anchor['measured'] else 'provisional_assumed_anchor',limitations=['A fitted plane and one anchor do not certify metric accuracy across the room.'])
if a.output.exists():raise FileExistsError('Preserve existing calibration; choose a new output path')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
