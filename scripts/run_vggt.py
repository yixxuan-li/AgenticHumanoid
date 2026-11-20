"""Estimate camera/depth geometry from video frames using pinned official VGGT.

Outputs remain in arbitrary scale until a measured reference is applied.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--frames',type=Path,default=Path('outputs/inspection'))
    p.add_argument('--checkpoint',type=Path,default=Path('checkpoints/vggt_model_ranges.pt'))
    p.add_argument('--source',type=Path,default=Path('vendor/vggt'))
    p.add_argument('--stride',type=int,default=2);p.add_argument('--output',type=Path,default=Path('outputs/geometry'))
    p.add_argument('--max-frames',type=int,default=24)
    args=p.parse_args();sys.path.insert(0,str(args.source.resolve()))
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    if not torch.cuda.is_available():raise RuntimeError('VGGT run requires a CUDA runtime; do not fabricate depth on failure')
    files=sorted(args.frames.glob('frame_[0-9][0-9][0-9][0-9].jpg'))[::args.stride][:args.max_frames]
    args.output.mkdir(parents=True,exist_ok=True)
    model=VGGT(enable_track=False,enable_point=False).eval()
    print('Loading official checkpoint',flush=True)
    with args.checkpoint.open('rb') as file:
        digest=hashlib.file_digest(file,'sha256').hexdigest()
    expected='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'
    if digest!=expected:raise RuntimeError(f'Official VGGT checkpoint SHA-256 mismatch: {digest}')
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    # Camera/depth weights must match exactly. Tracking and the separate point
    # head are unused; points are unprojected from the depth/camera predictions.
    state={k:v for k,v in state.items() if not k.startswith(('track_head.','point_head.'))}
    model.load_state_dict(state,strict=True);del state
    model=model.to('cuda')
    images=load_and_preprocess_images([str(p) for p in files],mode='pad').to('cuda')
    print(f'Inference: {len(files)} frames, {tuple(images.shape)}, {torch.cuda.get_device_name()}',flush=True)
    begin=time.time()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        predictions=model(images)
        extrinsic,intrinsic=pose_encoding_to_extri_intri(predictions['pose_enc'],images.shape[-2:])
    depth=predictions['depth'][0].float().cpu().numpy();conf=predictions['depth_conf'][0].float().cpu().numpy()
    extrinsic=extrinsic[0].float().cpu().numpy();intrinsic=intrinsic[0].float().cpu().numpy()
    points=unproject_depth_map_to_point_map(depth,extrinsic,intrinsic)
    rgb=images.permute(0,2,3,1).float().cpu().numpy();colors=np.uint8(np.clip(rgb*255,0,255))
    # Official pad mode uses WHITE padding. Track the exact source footprint;
    # RGB thresholds would include artificial padding and discard real black objects.
    from PIL import Image
    pixel_mask=np.zeros(conf.shape,bool);bounds=[]
    for i,path in enumerate(files):
        original_w,original_h=Image.open(path).size
        if original_w>=original_h:
            nw=518;nh=round(original_h*(nw/original_w)/14)*14
        else:
            nh=518;nw=round(original_w*(nh/original_h)/14)*14
        left,top=(518-nw)//2,(518-nh)//2;pixel_mask[i,top:top+nh,left:left+nw]=True
        bounds.append([left,top,nw,nh,original_w,original_h])
    np.savez_compressed(args.output/'predictions.npz',depth=depth,confidence=conf,extrinsic=extrinsic,intrinsic=intrinsic,points=points,colors=colors,pixel_mask=pixel_mask,image_bounds=np.array(bounds),frames=np.array([str(p) for p in files]))
    thresholds=np.array([np.percentile(conf[i][pixel_mask[i]],40) for i in range(len(files))])[:,None,None]
    valid=(conf>thresholds)&pixel_mask&np.isfinite(points).all(-1)
    xyz=points[valid];color=colors[valid]
    if len(xyz)>1500000:
        idx=np.random.default_rng(0).choice(len(xyz),1500000,replace=False);xyz=xyz[idx];color=color[idx]
    write_ply(args.output/'raw_cloud.ply',xyz,color)
    centers=-np.einsum('nji,nj->ni',extrinsic[:,:,:3],extrinsic[:,:,3])
    report=dict(model='facebook/VGGT-1B',source_revision=json.loads((args.source/'UPSTREAM_REVISION.json').read_text()),
        checkpoint_sha256=digest,checkpoint_digest_verified=True,
        frames=[str(p) for p in files],inference_seconds=time.time()-begin,gpu=torch.cuda.get_device_name(),
        max_allocated_gpu_gb=torch.cuda.max_memory_allocated()/1024**3,point_count=len(xyz),
        scale_status='arbitrary model units; no metric measurement supplied',camera_centers=centers.tolist(),
        camera_path_length_model_units=float(np.linalg.norm(np.diff(centers,axis=0),axis=1).sum()),
        caveats=['Learned depth is a hypothesis, not metric ground truth.','Short rotation-heavy video has weak geometric depth constraints.'])
    (args.output/'inference_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)


def write_ply(path,xyz,rgb):
    array=np.empty(len(xyz),dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
    for i,k in enumerate(['x','y','z']):array[k]=xyz[:,i]
    for i,k in enumerate(['red','green','blue']):array[k]=rgb[:,i]
    header=f'ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n'
    with path.open('wb') as f:f.write(header.encode());array.tofile(f)


if __name__=='__main__':main()
