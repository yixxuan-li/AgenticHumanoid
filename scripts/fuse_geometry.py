"""Fuse actual learned depth views into one TSDF surface; no collision inference."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import open3d as o3d
p=argparse.ArgumentParser();p.add_argument('--geometry',type=Path,default=Path('outputs/geometry_full'));p.add_argument('--output',type=Path,default=Path('outputs/fusion'));p.add_argument('--voxel',type=float,default=.012);p.add_argument('--truncation',type=float,default=.05);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
raw=np.load(a.geometry/'predictions.npz');aligned=np.load(a.geometry/'aligned_geometry.npz');depth=raw['depth'][...,0];colors=raw['colors'];valid=aligned['visual_valid'];Q=aligned['rotation'];translation=aligned['translation'];scale=float(aligned['scale']);extr=raw['extrinsic'];intr=raw['intrinsic'];n,h,w=depth.shape
audit=json.loads((a.geometry/'geometry_audit.json').read_text())
if not audit.get('candidate_for_visual_review'):raise ValueError('Geometry alignment quality gate failed; review the audit before surface fusion')
volume=o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=a.voxel,sdf_trunc=a.truncation,color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8);begin=time.time()
for i in range(n):
 metric=np.where(valid[i],depth[i]*scale,0).astype('float32');rgb=o3d.geometry.Image(np.ascontiguousarray(colors[i]));dep=o3d.geometry.Image(np.ascontiguousarray(metric));rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(rgb,dep,depth_scale=1.,depth_trunc=12.,convert_rgb_to_intensity=False)
 e=np.eye(4);e[:3,:3]=extr[i,:,:3]@Q.T;e[:3,3]=extr[i,:,3]*scale-e[:3,:3]@translation
 K=intr[i];camera=o3d.camera.PinholeCameraIntrinsic(w,h,float(K[0,0]),float(K[1,1]),float(K[0,2]),float(K[1,2]));volume.integrate(rgbd,camera,e)
 print(f'Integrated {i+1}/{n}',flush=True)
mesh=volume.extract_triangle_mesh();mesh.remove_duplicated_vertices();mesh.remove_degenerate_triangles();mesh.remove_unreferenced_vertices();mesh.compute_vertex_normals();o3d.io.write_triangle_mesh(str(a.output/'fused.ply'),mesh)
np.savez_compressed(a.output/'fused_geometry.npz',vertices=np.asarray(mesh.vertices),faces=np.asarray(mesh.triangles),colors=np.asarray(mesh.vertex_colors),normals=np.asarray(mesh.vertex_normals))
report=dict(method='Open3D scalable TSDF integration of VGGT depth and source RGB',frames=n,voxel_m=a.voxel,truncation_m=a.truncation,vertices=len(mesh.vertices),triangles=len(mesh.triangles),seconds=time.time()-begin,scale_status=audit['scale_status'],collision=False,limitations=['Fusion smooths agreement between learned depth maps; it does not establish true depth.','Reflective/occluded surfaces may remain absent or distorted.'])
(a.output/'fusion_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
