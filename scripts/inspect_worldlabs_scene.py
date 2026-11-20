"""Inspect Atlas RGBD in its native Y-up frame and inspect source-camera coverage."""
import os,sys,json,argparse
from pathlib import Path
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1');os.environ.setdefault('MPLCONFIGDIR','/tmp/agentic-humanoid-matplotlib')
import cv2,numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
parser=argparse.ArgumentParser();parser.add_argument('--workspace',type=Path,default=Path('outputs/worldlabs'));args=parser.parse_args()
out=args.workspace;op=json.loads((out/'pose/operation.json').read_text());frames=op['response']['frames'];selected=json.loads((out/'selected_frames.json').read_text());points=[];colors=[];rows=[]
def local_path(a):
 p=Path(a['path']);return p if p.exists() else out/'pose'/p.name
for i,f in enumerate(frames):
 depth=cv2.imread(str(local_path(f['depth']['depthAsset'])),cv2.IMREAD_UNCHANGED)
 if depth.ndim==3:depth=depth[...,0]
 rgb=cv2.cvtColor(cv2.imread(str(local_path(f['imageAsset']))),cv2.COLOR_BGR2RGB);rgb=cv2.resize(rgb,(depth.shape[1],depth.shape[0]))
 confidence=cv2.imread(str(local_path(f['depth']['confidenceAsset'])),0);k=f['camera']['intrinsics'];e=f['camera']['extrinsics'];R=Rotation.from_quat(e['quaternion']).as_matrix();t=np.array(e['position']);y,x=np.mgrid[:depth.shape[0]:6,:depth.shape[1]:6];z=depth[::6,::6];valid=np.isfinite(z)&(z>0)&(confidence[::6,::6]>64)
 xyz=np.stack([(x-k['cx'])/k['fx']*z,-(y-k['cy'])/k['fy']*z,-z],-1);world=xyz@R.T+t;points.append(world[valid]);colors.append(rgb[::6,::6][valid]);rows.append(dict(index=i,source=selected[i]['path'],valid_depth_fraction=float(np.mean(np.isfinite(depth)&(depth>0))),confident_fraction=float(np.mean(valid))))
points=np.concatenate(points);colors=np.concatenate(colors);np.savez_compressed(out/'posed_points.npz',points=points,colors=colors)
with (out/'posed_points.ply').open('w') as f:
 f.write(f'ply\nformat ascii 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
 for p,c in zip(points,colors):f.write(' '.join(map(str,[*p,*c]))+'\n')
fig,ax=plt.subplots(figsize=(10,9));mask=(points[:,1]>.05)&(points[:,1]<2.3);p=points[mask];c=colors[mask];ax.scatter(p[::2,0],p[::2,2],c=c[::2]/255,s=.3,alpha=.5)
for i,f in enumerate(frames):
 e=f['camera']['extrinsics'];t=np.array(e['position']);forward=Rotation.from_quat(e['quaternion']).apply([0,0,-1]);ax.plot(t[0],t[2],'ro',ms=3);ax.arrow(t[0],t[2],forward[0]*.35,forward[2]*.35,color='r',head_width=.05);ax.text(t[0],t[2],str(i),fontsize=7)
lo,hi=np.percentile(points[:,[0,2]],[1,99],axis=0);ax.set(xlim=[lo[0]-.2,hi[0]+.2],ylim=[lo[1]-.2,hi[1]+.2],xlabel='Atlas X [estimated units]',ylabel='Atlas Z [estimated units]',title='Atlas posed capture: top view, red source cameras; Y is up');ax.set_aspect('equal');fig.savefig(out/'posed_top_view.png',dpi=160);plt.close(fig)
report=dict(operation_id=op['id'],pose_refinement_accepted=op['response']['poseRefinementAccepted'],frames=rows,points=len(points),limitations=['Unseen surfaces are not recovered.','Atlas estimates scale; calibration still needs a measurement.']);(out/'pose_review.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
