"""Prepare Atlas depth correspondences for camera-only held-out PnP review."""
import argparse,json,os
from pathlib import Path
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2,numpy as np
from scipy.spatial.transform import Rotation
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,default=Path('outputs/worldlabs'));p.add_argument('--scene',type=Path,default=Path('outputs/worldlabs_scene'));p.add_argument('--output',type=Path,default=Path('outputs/worldlabs_geometry'));a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
root=a.workspace;op=json.loads((root/'pose/operation.json').read_text());mf=json.loads((a.scene/'manifest.json').read_text());conversion=mf['worldlabs_native_to_sim'];C=np.array(conversion['rotation']);scale=conversion['scale'];shift=np.r_[conversion['translation_xy'],0];floor=conversion['floor_y'];selected=json.loads((root/'selected_frames.json').read_text());points=[];valid=[];intrinsic=[];bounds=[];frames=[]
for i,f in enumerate(op['response']['frames']):
 depth=cv2.imread(str(root/'pose'/f['depth']['depthAsset']['path']),-1);confidence=cv2.imread(str(root/'pose'/f['depth']['confidenceAsset']['path']),0)
 if depth.ndim==3:depth=depth[...,0]
 height,width=depth.shape;w=max(1,width//2);h=max(1,height//2);depth=cv2.resize(depth,(w,h),interpolation=cv2.INTER_NEAREST);confidence=cv2.resize(confidence,(w,h),interpolation=cv2.INTER_NEAREST);k=f['camera']['intrinsics'];K=np.array([[k['fx']*w/k['width'],0,k['cx']*w/k['width']],[0,k['fy']*h/k['height'],k['cy']*h/k['height']],[0,0,1.]])
 y,x=np.mgrid[:h,:w];q=np.stack([(x-K[0,2])/K[0,0]*depth,-(y-K[1,2])/K[1,1]*depth,-depth],-1);e=f['camera']['extrinsics'];world=q@Rotation.from_quat(e['quaternion']).as_matrix().T+e['position'];world=scale*(world@C.T-[0,0,floor])+shift;mask=np.isfinite(world).all(-1)&(depth>0)&(confidence>64);points.append(np.nan_to_num(world).astype('float32'));valid.append(mask);intrinsic.append(K);frames.append(selected[i]['path'])
 source=cv2.imread(selected[i]['path']);source_h,source_w=source.shape[:2];bounds.append([0,0,w,h,source_w,source_h])

np.savez_compressed(a.output/'aligned_geometry.npz',points=np.array(points),valid=np.array(valid),intrinsic=np.array(intrinsic));np.savez_compressed(a.output/'predictions.npz',frames=np.array(frames),image_bounds=np.array(bounds),backend=np.array('World Labs Atlas images2PosedRGBD'));(a.output/'provenance.json').write_text(json.dumps(dict(source_operation_id=op['id'],purpose='Held-out review camera estimation only; no review RGB used to reconstruct the scene',image_grid=[w,h],metric_accuracy_verified=False),indent=2));print('Prepared Atlas depth for held-out camera review')
