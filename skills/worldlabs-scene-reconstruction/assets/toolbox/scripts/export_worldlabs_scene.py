"""Export an Atlas mesh and its upright collider boxes to MJCF and USD.

The room is static. Object articulation and measured physical parameters are
not inferred by this exporter. One explicitly added cube supports physics tests.
"""
import argparse,json,sys,xml.etree.ElementTree as ET
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,trimesh,mujoco
from scipy.spatial.transform import Rotation
from scipy.ndimage import label
from PIL import Image
from pxr import Usd,UsdGeom,UsdShade,Sdf,Gf
from scene_reconstruction.export import export_all,values
from scene_reconstruction.layout import box
from scene_reconstruction.navigation import collision_grid,plan

p=argparse.ArgumentParser();p.add_argument('--mesh',type=Path,required=True);p.add_argument('--colliders',type=Path,required=True);p.add_argument('--output',type=Path,default=Path('outputs/worldlabs_scene'));p.add_argument('--calibration',type=Path,required=True);p.add_argument('--workspace',type=Path,default=Path('outputs/worldlabs'));a=p.parse_args()
cal=json.loads(a.calibration.read_text());scale=float(cal['scale']);floor_y=float(cal['floor_y']);C=np.array([[1.,0,0],[0,0,-1],[0,1,0]])@np.array(cal.get('floor_rotation',np.eye(3)))
if not np.isfinite(scale) or scale<=0 or not np.isfinite(floor_y) or C.shape!=(3,3) or not np.allclose(C@C.T,np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(C),1,atol=1e-5):raise ValueError('Calibration requires a positive scale, finite floor offset and proper orthonormal rotation')
scene=trimesh.load(a.mesh,force='scene');colliders=trimesh.load(a.colliders,force='scene');meshes=[];bounds=[]
for node in scene.graph.nodes_geometry:
 T,key=scene.graph[node];m=scene.geometry[key].copy();m.apply_transform(T);m.vertices=scale*(m.vertices@C.T-[0,0,floor_y]);bounds.append(m.bounds);meshes.append(m)
lo=np.min([b[0] for b in bounds],axis=0);hi=np.max([b[1] for b in bounds],axis=0);shift=np.r_[-lo[:2]+.15,0.];dims=np.r_[hi[:2]-lo[:2]+.30,max(hi[2]+.05,2.7)]
out=a.output;out.mkdir(parents=True,exist_ok=True);(out/'visual').mkdir(exist_ok=True)
parts=[box('slab',[0,0,0],[*dims[:2],.12],'wood')];parts[0]['visible']=False
bodies=[dict(name='floor_main',kind='fixed',pos=[dims[0]/2,dims[1]/2,-.06],yaw=0,mass=0,evidence='Inferred continuous floor support spanning the extracted scene; inspect boundaries',parts=parts)]
for i,node in enumerate(colliders.graph.nodes_geometry):
 T,key=colliders.graph[node];m=colliders.geometry[key];center=trimesh.transform_points([m.bounds.mean(0)],T)[0];basis=C@T[:3,:3];length=np.linalg.norm(basis,axis=0);rot=basis/np.maximum(length,1e-12);size=m.extents*length*scale
 if np.min(size)<1e-5:continue
 if np.linalg.det(rot)<0:rot[:,0]*=-1
 part=box('solid',[0,0,0],size,'wall',rotation=Rotation.from_matrix(rot).as_euler('xyz').tolist());part['visible']=False
 pos=scale*(C@center-[0,0,floor_y])+shift
 # Consolidate low floor patches into the explicitly flat support slab.
 if pos[2]+np.sum(np.abs(rot[2])*size/2)<.10:continue
 bodies.append(dict(name=f'atlas_collider_{i:04d}',kind='fixed',pos=pos.tolist(),yaw=0,mass=0,evidence='World Labs splats2Mesh upright collision fit; estimated geometry',parts=[part]))
raw=json.loads((a.workspace/'pose_operation.json').read_text())['response']['frames'];start=scale*(C@np.array(raw[0]['camera']['extrinsics']['position'])-[0,0,floor_y])+shift
mf=dict(dimensions=dims.tolist(),desk_height=cal.get('table_height_m'),scale_status='measured_anchor' if cal.get('measured',False) else 'provisional_assumed_anchor',metric_anchor=cal,global_scale=1.,bodies=bodies,cameras=[['overview',[dims[0]/2,dims[1]/2-4.,4.8],[dims[0]/2,dims[1]/2,.8]]],navigation=dict(start=start[:2].tolist(),goals=[],radius=.32),source='World Labs Atlas: images2PosedRGBD -> atlasGenerate -> posedImages2Splats -> splats2Mesh',physical_parameters='Static room, assumed friction; no recovered articulated furniture or measured masses',worldlabs_native_to_sim=dict(rotation=C.tolist(),scale=scale,floor_y=floor_y,translation_xy=shift[:2].tolist()),limitations=['Generated unseen areas are hypotheses.','Continuous rectangular floor support replaces collider patches below 10 cm; static collider boxes are approximate.','Metric accuracy beyond the documented anchor has not been independently verified.'])
export_all(mf,out)
# Pick navigation locations from the same collision model used by the simulator.
model=mujoco.MjModel.from_xml_path(str(out/'scene.xml'));data=mujoco.MjData(model);mujoco.mj_forward(model,data);grid=collision_grid(model,data,mf['dimensions'],radius=.32);free=~grid['occupied'];components,_=label(free);yx=np.argwhere(free)
if not len(yx):raise ValueError('No clear floor cell: inspect collider boxes before using the scene')
xy=yx[:,::-1]*grid['resolution'];nearest=np.argmin(np.linalg.norm(xy-start[:2],axis=1));start_xy=xy[nearest];component=components[tuple(yx[nearest])];yx=np.argwhere(components==component);xy=yx[:,::-1]*grid['resolution'];distance=np.linalg.norm(xy-start_xy,axis=1);order=np.argsort(distance)[::-1];goals=[];routes=[]
for index in order:
 if distance[index]>2.9 or distance[index]<.6:continue
 if all(np.linalg.norm(xy[index]-g)>.6 for g in goals):
  try:route=plan(grid,start_xy,xy[index])
  except ValueError:continue
  if np.linalg.norm(np.diff(route,axis=0),axis=1).sum()<=3.4:goals.append(xy[index]);routes.append(route)
 if len(goals)==3:break
mf['navigation'].update(start=start_xy.tolist(),goals=[g.tolist() for g in goals])
# A small validation cube must not close a narrow robot-clearance corridor.
candidate_yx=np.argwhere(grid['clearance']>.18);choices=candidate_yx[:,::-1]*grid['resolution'];separated=np.linalg.norm(choices-start_xy,axis=1)>.7
for route in routes:
 for point in route:separated&=np.linalg.norm(choices-point,axis=1)>.45
if not separated.any():raise ValueError('No safe off-route floor location for the validation cube')
choices=choices[separated];cube_xy=choices[np.argmin(np.linalg.norm(choices-(start_xy+[.7,0]),axis=1))]
mf['bodies'].append(dict(name='test_block',kind='free',pos=[*cube_xy,.026],yaw=0,mass=.08,evidence='Added 5 cm benchmark cube; absent from the recording',parts=[box('block',[0,0,0],[.05,.05,.05],'red')]))
export_all(mf,out);tree=ET.parse(out/'scene.xml');root=tree.getroot();asset=root.find('asset');world=root.find('worldbody');stage=Usd.Stage.Open(str(out/'scene.usda'));visual=UsdGeom.Xform.Define(stage,'/World/AtlasVisual')
# Review cutaways only affect visual mesh groups, never collision boxes.
cut_bodies={n:ET.SubElement(world,'body',name=n) for n in ['photo_cutaway_ceiling','photo_cutaway_front']}
grouped=[]
for m in meshes:
 center=m.triangles_center+shift;ceiling=center[:,2]>min(2.5,dims[2]-.2);front=(center[:,1]<.50)&(center[:,2]>.20)&~ceiling
 for group,mask in [('background',~(ceiling|front)),('photo_cutaway_ceiling',ceiling),('photo_cutaway_front',front)]:
  if mask.any():grouped.append((group,m.submesh([np.flatnonzero(mask)],append=True,repair=False)))
meshes=[m for _,m in grouped]
head=root.find('visual/headlight')
if head is not None:
 for k in ['ambient','diffuse','specular']:head.set(k,'0 0 0')
for light in list(world.findall('light')):world.remove(light)
for material in asset.findall('material'):material.set('emission','1')
for i,(group,m) in enumerate(grouped):
 m.vertices+=shift;name=f'atlas_{i:03d}';uv=getattr(m.visual,'uv',None);material=getattr(m.visual,'material',None);image=getattr(material,'baseColorTexture',None)
 if image is None:image=getattr(material,'image',None)
 if uv is None or image is None:raise ValueError('Expected an image-textured GLB; request textureMode=image_texture')
 image.convert('RGB').save(out/'visual'/f'{name}.png')
 with (out/'visual'/f'{name}.obj').open('w') as f:
  for v in m.vertices:f.write('v '+values(v)+'\n')
  for v in uv:f.write('vt '+values(v)+'\n')
  for face in m.faces:f.write('f '+' '.join(f'{j+1}/{j+1}' for j in face)+'\n')
 ET.SubElement(asset,'texture',name=name+'_tex',type='2d',file=f'visual/{name}.png');ET.SubElement(asset,'material',name=name+'_mat',texture=name+'_tex',emission='1',specular='0',shininess='0');ET.SubElement(asset,'mesh',name=name,file=f'visual/{name}.obj',inertia='shell');ET.SubElement(cut_bodies.get(group,world),'geom',name=name,type='mesh',mesh=name,material=name+'_mat',mass='0',contype='0',conaffinity='0',group='1')
 mat=UsdShade.Material.Define(stage,f'/World/Looks/{name}');shader=UsdShade.Shader.Define(stage,f'/World/Looks/{name}/Shader');shader.CreateIdAttr('UsdPreviewSurface');shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0));reader=UsdShade.Shader.Define(stage,f'/World/Looks/{name}/UV');reader.CreateIdAttr('UsdPrimvarReader_float2');reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st');tex=UsdShade.Shader.Define(stage,f'/World/Looks/{name}/Texture');tex.CreateIdAttr('UsdUVTexture');tex.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(f'visual/{name}.png'));tex.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('sRGB');tex.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(),'result');tex.CreateOutput('rgb',Sdf.ValueTypeNames.Float3);shader.CreateInput('emissiveColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(),'rgb');mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
 UsdGeom.Xform.Define(stage,f'/World/AtlasVisual/{group}')
 mesh=UsdGeom.Mesh.Define(stage,f'/World/AtlasVisual/{group}/{name}');mesh.CreatePointsAttr(m.vertices.tolist());mesh.CreateFaceVertexCountsAttr([3]*len(m.faces));mesh.CreateFaceVertexIndicesAttr(m.faces.reshape(-1).tolist());mesh.CreateSubdivisionSchemeAttr('none');mesh.CreateDoubleSidedAttr(True);st=UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.vertex);st.Set(uv.tolist());UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
for i,f in enumerate(raw):
 e=f['camera']['extrinsics'];k=f['camera']['intrinsics'];R=C@Rotation.from_quat(e['quaternion']).as_matrix();pos=scale*(C@np.array(e['position'])-[0,0,floor_y])+shift;ET.SubElement(world,'camera',name=f'source_{i:03d}',pos=values(pos),xyaxes=values(np.r_[R[:,0],R[:,1]]),fovy=str(np.rad2deg(2*np.arctan(k['height']/(2*k['fy'])))))
ET.indent(tree);tree.write(out/'scene.xml',encoding='unicode');stage.GetRootLayer().Save();mujoco.MjModel.from_xml_path(str(out/'scene.xml'))
count=sum(b['name'].startswith('atlas_collider_') for b in mf['bodies'])
(out/'atlas_export.json').write_text(json.dumps(dict(mesh_source=str(a.mesh),collider_source=str(a.colliders),mesh_parts=len(meshes),triangles=sum(len(m.faces) for m in meshes),collider_count=count,scene_type='Static reconstructed room plus one added free benchmark cube',calibration=cal),indent=2))
(out/'photo_ownership.json').write_text(json.dumps(dict(source='World Labs Atlas textured mesh',ownership='Static visual geometry; original furniture is not segmented or articulated',collision=False),indent=2));print(f'Exported {out}: {len(meshes)} visual meshes, {count} collider boxes')
