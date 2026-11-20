"""Texture one fused surface from depth-tested actual source views.

Source view selection and mild adjacency regularization are visual hypotheses.
No synthesized pixels or collision surfaces are introduced by this script.
"""
import argparse,json,os,shutil,time
from pathlib import Path
import cv2,numpy as np
import open3d as o3d
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--geometry',type=Path,default=Path('outputs/geometry_full'));p.add_argument('--fusion',type=Path,default=Path('outputs/fusion'));p.add_argument('--max-triangles',type=int,default=250000);p.add_argument('--bounds',nargs=6,type=float,metavar=('XMIN','YMIN','ZMIN','XMAX','YMAX','ZMAX'));p.add_argument('--overwrite',action='store_true');a=p.parse_args();begin=time.time()
if a.fusion.resolve()==a.geometry.resolve():p.error('Use a separate fusion directory to preserve the original surfaces')
data=np.load(a.fusion/'fused_geometry.npz');mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(data['vertices']),o3d.utility.Vector3iVector(data['faces']))
if a.bounds:
 vertices=np.asarray(mesh.vertices);inside=np.all(vertices>np.array(a.bounds[:3]),axis=1)&np.all(vertices<np.array(a.bounds[3:]),axis=1)
 mesh.remove_triangles_by_mask(~inside[np.asarray(mesh.triangles)].all(axis=1));mesh.remove_unreferenced_vertices()
if len(mesh.triangles)>a.max_triangles:mesh=mesh.simplify_quadric_decimation(a.max_triangles)
mesh.remove_degenerate_triangles();mesh.remove_unreferenced_vertices();mesh.compute_triangle_normals();v=np.asarray(mesh.vertices);f=np.asarray(mesh.triangles);normal=np.asarray(mesh.triangle_normals);center=v[f].mean(axis=1)
raw=np.load(a.geometry/'predictions.npz');align=np.load(a.geometry/'aligned_geometry.npz');depth=raw['depth'][...,0];rgb=raw['colors'];mask=align['visual_valid'];e=raw['extrinsic'];K=raw['intrinsic'];Q=align['rotation'];tr=align['translation'];scale=float(align['scale']);n,h,w=depth.shape
model=(center-tr)@Q/scale;model_vertices=(v-tr)@Q/scale;scores=np.full((n,len(f)),-np.inf,np.float32);all_uv=[]
for i in range(n):
 cam=model@e[i,:,:3].T+e[i,:,3];q=cam@K[i].T;uv=(q[:,:2]/q[:,2:]).astype('float32');u,y=uv[:,0],uv[:,1]
 ix=np.clip(np.rint(np.nan_to_num(u,nan=-100,posinf=-100,neginf=-100)).astype(int),0,w-1);iy=np.clip(np.rint(np.nan_to_num(y,nan=-100,posinf=-100,neginf=-100)).astype(int),0,h-1)
 observed=depth[i,iy,ix];valid=mask[i,iy,ix].copy()
 cv=model_vertices@e[i,:,:3].T+e[i,:,3];qv=cv@K[i].T;uvv=qv[:,:2]/qv[:,2:];left,top,nw,nh,_,_=raw['image_bounds'][i]
 in_view=(uvv[:,0]>left+1)&(uvv[:,0]<left+nw-2)&(uvv[:,1]>top+1)&(uvv[:,1]<top+nh-2)&(cv[:,2]>0)
 camera=(-e[i,:,:3].T@e[i,:,3])@Q.T*scale+tr;direction=camera-center;distance=np.linalg.norm(direction,axis=1);cos=np.abs(np.sum(normal*direction,axis=1))/np.maximum(distance,1e-8)
 residual=np.abs(observed-cam[:,2])*scale;valid&=(cam[:,2]>0)&(residual<.06)&in_view[f].all(axis=1)
 score=cos/np.maximum(distance**2,.1)*np.exp(-residual/.05);scores[i,valid]=score[valid];all_uv.append(np.c_[uvv[:,0]/(w-1),1-uvv[:,1]/(h-1)])
 print(f'View assignment {i+1}/{n}',flush=True)
labels=scores.argmax(axis=0);best=scores.max(axis=0);labels[~np.isfinite(best)]=-1
# Shared-edge neighbors permit local camera label regularization without moving geometry.
edges=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]);edges.sort(axis=1);face_ids=np.tile(np.arange(len(f)),3);order=np.lexsort((edges[:,1],edges[:,0]));same=np.all(edges[order[1:]]==edges[order[:-1]],axis=1);aa=face_ids[order[:-1][same]];bb=face_ids[order[1:][same]]
neighbors=np.full((len(f),3),-1,int);counts=np.zeros(len(f),int)
for x,y in zip(np.r_[aa,bb],np.r_[bb,aa]):
 if counts[x]<3:neighbors[x,counts[x]]=y;counts[x]+=1
for _ in range(5):
 nl=np.where(neighbors>=0,labels[np.maximum(neighbors,0)],-2);candidate=np.where(nl[:,0]==nl[:,1],nl[:,0],np.where(nl[:,0]==nl[:,2],nl[:,0],np.where(nl[:,1]==nl[:,2],nl[:,1],-1)));eligible=(candidate>=0)&(labels>=0);ids=np.where(eligible)[0];good=scores[candidate[ids],ids]>.8*best[ids];labels[ids[good]]=candidate[ids[good]]
folder=a.fusion/'photo_mesh'
if folder.exists():
 if not a.overwrite:p.error('Fusion photo_mesh exists; use --overwrite to replace generated output')
 shutil.rmtree(folder)
folder.mkdir();written=0;views=[]
for i in range(n):
 triangles=f[labels==i]
 if not len(triangles):continue
 used=np.unique(triangles);remap=np.full(len(v),-1,int);remap[used]=np.arange(len(used));triangles=remap[triangles];vertices=v[used];uv=all_uv[i][used]
 lines=['v '+' '.join(f'{x:.8g}' for x in p) for p in vertices]+['vt '+' '.join(f'{x:.8g}' for x in p) for p in uv]+['f '+' '.join(f'{x+1}/{x+1}' for x in tri) for tri in triangles]
 (folder/f'view_{i:03d}.obj').write_text('\n'.join(lines)+'\n');Image.fromarray(rgb[i]).save(folder/f'view_{i:03d}.png');written+=len(triangles)
 views.append(dict(index=i,vertices=vertices,faces=triangles,uv=uv))
for name in ['predictions.npz','aligned_geometry.npz']:
 target=a.fusion/name
 if not target.exists():os.link(a.geometry/name,target)
for name in ['scene_features.json','geometry_audit.json']:
 if (a.geometry/name).exists():shutil.copy2(a.geometry/name,a.fusion/name)
from process_geometry import export_visual_assets
audit=json.loads((a.geometry/'geometry_audit.json').read_text());export_visual_assets(a.fusion,views,e,K,Q,tr,scale,audit['scale_status'],raw['image_bounds'])
report=dict(input_triangles=int(len(data['faces'])),crop_bounds_dense_coordinates=a.bounds,decimated_triangles=len(f),textured_triangles=written,untextured_triangles_removed=int((labels<0).sum()),source='actual RGB, depth-tested per triangle, camera scoring by incidence/distance and five mild adjacency passes',seconds=time.time()-begin,collision=False)
(a.fusion/'texture_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
