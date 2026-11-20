"""Constrain fitted floor/table surfaces; preserve the unmodified Atlas mesh.

The table outline is fitted to the completed splat, not a measured CAD model.
Its texture is projected from the curated RGBD inputs with depth visibility.
"""
import argparse,json,os
from pathlib import Path
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2,numpy as np,trimesh
from PIL import Image
from scipy.spatial.transform import Rotation
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=Path('outputs/worldlabs/mesh/mesh.glb'));p.add_argument('--output',type=Path,default=Path('outputs/worldlabs/constrained_mesh.glb'));a=p.parse_args();root=Path('outputs/worldlabs');cal=json.loads((root/'calibration.json').read_text());s=cal['scale'];floor=cal['floor_y'];C=np.array([[1,0,0],[0,0,-1],[0,1,0]])@cal['floor_rotation']
splats=np.array(json.loads((root/'table_splats.json').read_text()))[:,:3];q=s*(splats@C.T-[0,0,floor]);q=q[np.abs(q[:,2]-.74)<.04];lo,hi=np.percentile(q[:,:2],[.5,99.5],axis=0);q=q[np.all((q[:,:2]>=lo)&(q[:,:2]<=hi),axis=1)];center,size,degrees=cv2.minAreaRect(q[:,:2].astype('float32'));center=np.array(center);size=np.array(size);yaw=np.deg2rad(degrees);R=Rotation.from_euler('z',yaw).as_matrix()
def native(points):return (points/s+[0,0,floor])@C
width,height=512,1024;xx,yy=np.meshgrid(np.linspace(-size[0]/2,size[0]/2,width),np.linspace(size[1]/2,-size[1]/2,height));world=np.stack([xx,yy,np.full_like(xx,.7405)],-1)@R.T+[*center,0.];raw=native(world);best=np.full((height,width),-np.inf);image=np.full((height,width,3),210,np.uint8);chosen=np.full((height,width),-1,np.int16);inputs=[]
for stage,indices in [('pose',list(range(12))),('round1',[0,2]),('round2',[0])]:
 frames=json.loads((root/stage/'operation.json').read_text())['response']['frames']
 for index in indices:inputs.append((stage,index,frames[index]))
for n,(stage,index,f) in enumerate(inputs):
 k=f['camera']['intrinsics'];e=f['camera']['extrinsics'];cam=(raw-e['position'])@Rotation.from_quat(e['quaternion']).as_matrix();z=-cam[...,2];u=(k['fx']*cam[...,0]/z+k['cx']).astype('float32');v=(-k['fy']*cam[...,1]/z+k['cy']).astype('float32');D=cv2.imread(str(root/stage/f['depth']['depthAsset']['path']),-1);rgb=cv2.cvtColor(cv2.imread(str(root/stage/f['imageAsset']['path'])),cv2.COLOR_BGR2RGB);sampled=cv2.remap(D,u,v,cv2.INTER_NEAREST);color=cv2.remap(rgb,u,v,cv2.INTER_LINEAR);residual=np.abs(sampled-z)*s;valid=np.isfinite(sampled)&(sampled>0)&(z>0)&(u>=0)&(u<k['width']-1)&(v>=0)&(v<k['height']-1)&(residual<.075);green=(color[...,1]>1.2*color[...,0])&(color[...,1]>1.3*color[...,2]);valid&=~green
 if stage=='round2':
  mask=cv2.imread(str(root/'round2/view0_mask.png'),0);valid&=cv2.remap(mask,u,v,cv2.INTER_NEAREST)>0
 score=-(0 if stage=='pose' else 1.)-.03*abs(index-6)-3*residual;replace=valid&(score>best);image[replace]=color[replace];chosen[replace]=n;best[replace]=score[replace]
holes=chosen<0;image=cv2.inpaint(image,holes.astype('uint8'),7,cv2.INPAINT_TELEA);Image.fromarray(image).save(root/'table_atlas.png')
scene=trimesh.load(a.input,force='scene');result=trimesh.Scene();removed=0;flattened=0;floor_flattened=0
for i,node in enumerate(scene.graph.nodes_geometry):
 T,key=scene.graph[node];m=scene.geometry[key].copy();m.apply_transform(T);m.vertices=s*(m.vertices@C.T-[0,0,floor]);vertices=m.vertices.copy();normal=m.vertex_normals.copy();floor_mask=(np.abs(vertices[:,2])<.12)&(np.abs(normal[:,2])>.7);vertices[floor_mask,2]=.0015;floor_flattened+=int(floor_mask.sum());m.vertices=vertices;centers=m.triangles_center;local=(centers-[*center,0])@R;inside=(np.abs(local[:,0])<size[0]/2+.10)&(np.abs(local[:,1])<size[1]/2+.10);uv=m.visual.uv;mat=m.visual.material;texture=getattr(mat,'baseColorTexture',None)
 if texture is None:texture=getattr(mat,'image',None)
 tex=np.asarray(texture.convert('RGB'));coords=uv[m.faces].mean(1);rgb=tex[np.clip(((1-coords[:,1])*(tex.shape[0]-1)).astype(int),0,tex.shape[0]-1),np.clip((coords[:,0]*(tex.shape[1]-1)).astype(int),0,tex.shape[1]-1)].astype(float);green=(rgb[:,1]>1.2*rgb[:,0])&(rgb[:,1]>1.3*rgb[:,2]);vertex_rgb=tex[np.clip(((1-uv[:,1])*(tex.shape[0]-1)).astype(int),0,tex.shape[0]-1),np.clip((uv[:,0]*(tex.shape[1]-1)).astype(int),0,tex.shape[1]-1)].astype(float);vertex_green=(vertex_rgb[:,1]>1.2*vertex_rgb[:,0])&(vertex_rgb[:,1]>1.3*vertex_rgb[:,2]);green|=np.any(vertex_green[m.faces],axis=1);top=inside&(centers[:,2]>.64)&(centers[:,2]<.87)&(np.abs(m.face_normals[:,2])>.25)&~green;sheet=inside&(centers[:,2]>.12)&(centers[:,2]<.72)&(rgb.mean(1)>70)&~green;remove=top|sheet;removed+=int(remove.sum());m.update_faces(~remove);m.remove_unreferenced_vertices();m.vertices=native(m.vertices);result.add_geometry(m,node_name=f'atlas_{i}')
verts=np.array([[-size[0]/2,-size[1]/2,.7405],[size[0]/2,-size[1]/2,.7405],[size[0]/2,size[1]/2,.7405],[-size[0]/2,size[1]/2,.7405]])@R.T+[*center,0];plane=trimesh.Trimesh(vertices=native(verts),faces=[[0,1,2],[0,2,3]],process=False);plane.visual=trimesh.visual.texture.TextureVisuals(uv=[[0,0],[1,0],[1,1],[0,1]],image=Image.fromarray(image));result.add_geometry(plane,node_name='constrained_tabletop');result.export(a.output)
report=dict(input_mesh=str(a.input),output_mesh=str(a.output),table_center_unshifted_sim_xy=center.tolist(),table_size_m=size.tolist(),table_yaw_rad=float(yaw),table_height_assumed_m=.74,outline_source='0.5 to 99.5 percentile support of the refined Atlas splat near the table plane',table_texture_coverage_fraction=float(np.mean(chosen>=0)),inpainted_texture_fraction=float(np.mean(holes)),removed_sheet_and_top_triangles=removed,floor_vertices_constrained=floor_flattened,scale_status='Assumed table height; not measured',limitations=['Constraints do not recover hidden mechanisms or exact object dimensions.','Texture inpainting fills occlusions.','Unchanged collision boxes remain approximate; this operation changes visual geometry.']);(root/'mesh_constraints.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
