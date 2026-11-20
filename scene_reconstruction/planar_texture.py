"""Rectify observed RGB onto a fitted plane with depth visibility checks."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


def make_table_texture(manifest,geometry,output):
    geometry=Path(geometry);output=Path(output);aligned=np.load(geometry/'aligned_geometry.npz');pred=np.load(geometry/'predictions.npz')
    b=next(b for b in manifest['bodies'] if b['name']=='central_table_0');top=next(p for p in b['parts'] if p['name']=='top');w,h=1024,512
    xx,yy=np.meshgrid(np.linspace(-top['size'][0]/2,top['size'][0]/2,w),np.linspace(top['size'][1]/2,-top['size'][1]/2,h));local=np.stack([xx,yy,np.full_like(xx,.74)],-1);world=local@Rotation.from_euler('z',b['yaw']).as_matrix().T+np.array(b['pos'])
    raw=(world-np.array(manifest['source_geometry_shift_m'])-aligned['translation'])@aligned['rotation']/float(aligned['scale'])
    best=np.full((h,w),-np.inf);result=np.full((h,w,3),220,np.uint8);chosen=np.full((h,w),-1,np.int16)
    for i in range(len(pred['colors'])):
        e=aligned['extrinsic'][i];K=aligned['intrinsic'][i];cam=raw@e[:,:3].T+e[:,3];q=cam@K.T;uv=(q[...,:2]/q[...,2:]).astype('float32');u,v=uv[...,0],uv[...,1]
        depth=cv2.remap(pred['depth'][i].astype('float32'),u,v,cv2.INTER_LINEAR)
        valid=cv2.remap(pred['pixel_mask'][i].astype('uint8'),u,v,cv2.INTER_NEAREST)>0
        residual=np.abs(depth-cam[...,2])*float(aligned['scale']);valid&=(cam[...,2]>0)&(residual<.04)
        color=cv2.remap(pred['colors'][i],u,v,cv2.INTER_LINEAR)
        # Prefer one central source view; alternatives fill its occlusion/FOV gaps.
        score=-abs(i-18)*.03-residual*3-np.maximum(0,np.abs(u-K[0,2])-100)*.002
        replace=valid&(score>best);best[replace]=score[replace];result[replace]=color[replace];chosen[replace]=i
    # Do not bake the dynamic small black object into its supporting plane.
    prop=next(b for b in manifest['bodies'] if b['name']=='pick_box');p=(world-np.array(prop['pos']))@Rotation.from_euler('z',prop['yaw']).as_matrix()
    remove=(np.abs(p[...,0])<.075)&(np.abs(p[...,1])<.055)
    holes=(chosen<0)|remove;result=cv2.inpaint(result,holes.astype('uint8'),7,cv2.INPAINT_TELEA)
    (output/'textures').mkdir(exist_ok=True);Image.fromarray(result).save(output/'textures/captured_table.png')
    report=dict(source='Depth-tested planar projection of source RGB; preference for frame 19',coverage_fraction=float(np.mean(chosen>=0)),plane_height_m=.74,pose_yaw_rad=b['yaw'],inpainting='Occluded pixels and footprint of dynamic small black object',scale_status=manifest['scale_status'])
    (output/'table_texture_provenance.json').write_text(json.dumps(report,indent=2));return report
