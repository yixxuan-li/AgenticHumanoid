"""Attach captured surfaces to semantic bodies; photographs never collide.

Ownership is a geometric/color hypothesis, saved for inspection. Occluded faces
retain primitive fallbacks. This does not recover material or lighting properties.
"""
import json
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from .export import values


def signed_distance(points,body):
    local=(points-np.array(body['pos']))@Rotation.from_euler('z',body['yaw']).as_matrix()
    result=np.full(len(points),np.inf)
    for p in body['parts']:
        if not p['collision']:continue
        q=(local-np.array(p['pos']))@Rotation.from_euler('xyz',p['rotation']).as_matrix()
        if p['shape']=='box':d=np.abs(q)-np.array(p['size'])/2
        else:d=np.c_[np.linalg.norm(q[:,:2],axis=1)-p['size'][0],np.abs(q[:,2])-p['size'][1]/2]
        sdf=np.linalg.norm(np.maximum(d,0),axis=1)+np.minimum(d.max(axis=1),0)
        result=np.minimum(result,sdf)
    return result


def read_obj(path):
    v=[];uv=[];f=[]
    for line in path.read_text().splitlines():
        fields=line.split()
        if not fields:continue
        if fields[0]=='v':v.append([float(x) for x in fields[1:4]])
        elif fields[0]=='vt':uv.append([float(x) for x in fields[1:3]])
        elif fields[0]=='f':f.append([int(x.split('/')[0])-1 for x in fields[1:]])
    return np.array(v),np.array(uv),np.array(f)


def attach_photo_skin(manifest,geometry,output,dynamic_geometry=None):
    from pxr import Gf,Sdf,Usd,UsdGeom,UsdShade
    geometry=Path(geometry);output=Path(output);folder=output/'photo'
    if folder.exists():shutil.rmtree(folder)
    folder.mkdir()
    tree=ET.parse(output/'scene.xml');root=tree.getroot();asset=root.find('asset');world=root.find('worldbody')
    # Source pixels already contain illumination. Avoid lighting them twice.
    headlight=root.find('visual/headlight');headlight.set('ambient','0 0 0');headlight.set('diffuse','0 0 0');headlight.set('specular','0 0 0')
    for light in world.findall('light'):world.remove(light)
    for material in asset.findall('material'):material.set('emission','1')
    stage=Usd.Stage.Open(str(output/'scene.usda'))
    shift=np.array(manifest['source_geometry_shift_m']);dynamic=[b for b in manifest['bodies'] if b['kind']!='fixed' and b['name']!='test_block']
    names=['background']+[b['name'] for b in dynamic]+['cutaway_ceiling','cutaway_front'];counts={name:0 for name in names};mesh_count=0
    for name in names[-2:]:ET.SubElement(world,'body',name='photo_'+name)
    sources=[(source,'static' if dynamic_geometry else 'all') for source in sorted((geometry/'photo_mesh').glob('view_*.obj'))]
    if dynamic_geometry:sources += [(source,'dynamic') for source in sorted((Path(dynamic_geometry)/'photo_mesh').glob('view_*.obj'))]
    for source,role in sources:
        view=source.stem if role=='all' else role+'_'+source.stem
        vertices,uv,faces=read_obj(source);vertices+=shift
        image=np.asarray(Image.open(source.with_suffix('.png')).convert('RGB'));h,w=image.shape[:2]
        normals=np.zeros_like(vertices);face_normals=np.cross(vertices[faces[:,1]]-vertices[faces[:,0]],vertices[faces[:,2]]-vertices[faces[:,0]])
        for corner in range(3):np.add.at(normals,faces[:,corner],face_normals)
        normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
        rgb=image[np.clip(np.round((1-uv[:,1])*(h-1)).astype(int),0,h-1),np.clip(np.round(uv[:,0]*(w-1)).astype(int),0,w-1)].astype(float)
        labels=np.zeros(len(vertices),int);best=np.full(len(vertices),.15)
        for i,b in enumerate(dynamic,1):
            distance=np.abs(signed_distance(vertices,b));name=b['name'];z=vertices[:,2]
            if 'green_chair' in name:
                green=(rgb[:,1]>1.2*rgb[:,0])&(rgb[:,1]>1.3*rgb[:,2])&(rgb[:,1]>65)
                permitted=green|((z<.56)&(rgb.max(1)-rgb.min(1)<65))
                # Associate the complete colored instance, including curved edges
                # whose predicted depth departs from the simplified solid proxy.
                instance=green&(np.linalg.norm(vertices[:,:2]-np.array(b['pos'][:2]),axis=1)<.52)&(z>.30)&(z<.90)
                distance[instance]=np.minimum(distance[instance],.015)
            elif name.startswith('office_'):permitted=(rgb.mean(1)<155)&(rgb.max(1)-rgb.min(1)<70)
            elif name=='gray_chair':permitted=(rgb.max(1)-rgb.min(1)<60)&(z<1.10)&((z<.58)|(np.abs(normals[:,2])<.65))
            elif name=='cabinet_door':permitted=(rgb.max(1)-rgb.min(1)<80)&(rgb.mean(1)>45)&(rgb.mean(1)<245)
            else:permitted=np.ones(len(z),bool)
            if 'chair' not in name and not name.startswith('office_'):
                permitted &= distance<(.05 if name=='storage_bin' else .12 if name=='cabinet_door' else .025)
            selected=(distance<best)&permitted&(z>.045)
            labels[selected]=i;best[selected]=distance[selected]
        # Triangle ownership by majority. Remove mixed boundaries rather than
        # stretch a photographed floor or desk onto a moving object.
        tri_labels=labels[faces];owners=np.where((tri_labels[:,0]==tri_labels[:,1])|(tri_labels[:,0]==tri_labels[:,2]),tri_labels[:,0],np.where(tri_labels[:,1]==tri_labels[:,2],tri_labels[:,1],0))
        central=next(b for b in manifest['bodies'] if b['name']=='central_table_0');top=next(p for p in central['parts'] if p['name']=='top')
        local=(vertices-np.array(central['pos']))@Rotation.from_euler('z',central['yaw']).as_matrix()
        table_surface=(np.abs(local[:,0])<top['size'][0]/2+.03)&(np.abs(local[:,1])<top['size'][1]/2+.03)&(np.abs(vertices[:,2]-.74)<.065)&(np.abs(normals[:,2])>.75)&(labels==0)
        owners[np.sum(table_surface[faces],axis=1)>=2]=-1
        centers=vertices[faces].mean(axis=1)
        owners[(owners==0)&(centers[:,2]>2.50)]=len(dynamic)+1
        owners[(owners==0)&(centers[:,1]<.78)&(centers[:,2]>.30)]=len(dynamic)+2
        # Constrain photographed floor to its fitted support plane.
        floor=(vertices[:,2]>-.035)&(vertices[:,2]<.025)&(labels==0);vertices[floor,2]=.0015+int(view.split('_')[-1])*.00003
        for b in manifest['bodies']:
            if b['kind']!='fixed':continue
            for p in b['parts']:
                if p['name']!='top' or p['shape']!='box':continue
                local=(vertices-np.array(b['pos']))@Rotation.from_euler('z',b['yaw']).as_matrix()-p['pos']
                on_top=(np.abs(local[:,0])<p['size'][0]/2+.035)&(np.abs(local[:,1])<p['size'][1]/2+.035)&(np.abs(vertices[:,2]-.74)<.07)&(np.abs(normals[:,2])>.75)&(labels==0)
                vertices[on_top,2]=.743+int(view.split('_')[-1])*.00003
        texpath=folder/(view+'.png');shutil.copy2(source.with_suffix('.png'),texpath)
        ET.SubElement(asset,'texture',name='photo_tex_'+view,type='2d',file=f'photo/{view}.png')
        ET.SubElement(asset,'material',name='photo_mat_'+view,texture='photo_tex_'+view,rgba='1 1 1 1',emission='1',specular='0',shininess='0')
        material=UsdShade.Material.Define(stage,f'/World/Looks/photo_{view}');shader=UsdShade.Shader.Define(stage,f'/World/Looks/photo_{view}/Shader');shader.CreateIdAttr('UsdPreviewSurface');shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0))
        reader=UsdShade.Shader.Define(stage,f'/World/Looks/photo_{view}/UV');reader.CreateIdAttr('UsdPrimvarReader_float2');reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st')
        texture=UsdShade.Shader.Define(stage,f'/World/Looks/photo_{view}/Texture');texture.CreateIdAttr('UsdUVTexture');texture.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(f'photo/{view}.png'));texture.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('sRGB');texture.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(),'result');texture.CreateOutput('rgb',Sdf.ValueTypeNames.Float3)
        shader.CreateInput('emissiveColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(),'rgb');material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
        for i,name in enumerate(names):
            if role=='static' and 0<i<=len(dynamic):continue
            if role=='dynamic' and (i==0 or i>len(dynamic)):continue
            triangles=faces[owners==i]
            if len(triangles)<4:continue
            used=np.unique(triangles);remap=np.full(len(vertices),-1,int);remap[used]=np.arange(len(used));triangles=remap[triangles];points=vertices[used].copy();uvs=uv[used]
            if 0<i<=len(dynamic):
                b=dynamic[i-1];points=(points-np.array(b['pos']))@Rotation.from_euler('z',b['yaw']).as_matrix();parent=world.find(f"body[@name='{name}']");usdpath=f'/World/Scene/{name}'
                if 'green_chair' in name:
                    # Fit captured panel pixels onto the corresponding solid
                    # surface so primitive fallbacks cannot occlude their skin.
                    source_camera=ET.parse(geometry/'photo_visual.xml').getroot().find(f"worldbody/camera[@name='source_{int(view.split('_')[-1]):03d}']")
                    center=np.fromstring(source_camera.get('pos'),sep=' ')+shift
                    camera_local=(center-np.array(b['pos']))@Rotation.from_euler('z',b['yaw']).as_matrix()
                    for part in b['parts']:
                        if part['name'] not in ['seat','back']:continue
                        R=Rotation.from_euler('xyz',part['rotation']).as_matrix();q=(points-part['pos'])@R;half=np.array(part['size'])/2
                        axis=2 if part['name']=='seat' else 1
                        other=[a for a in range(3) if a!=axis]
                        selected=np.all(np.abs(q[:,other])<half[other]+.025,axis=1)&(np.abs(q[:,axis])<half[axis]+.055)
                        side=np.sign(((camera_local-part['pos'])@R)[axis]) or 1
                        q[selected,axis]=side*(half[axis]+.001+int(view.split('_')[-1])*.00001)
                        points[selected]=q[selected]@R.T+part['pos']
            elif i>len(dynamic):parent=world.find(f"body[@name='photo_{name}']");usdpath=f'/World/PhotoBackground/photo_{name}'
            else:parent=world;usdpath='/World/PhotoBackground'
            meshname=f'photo_{view}_{name}';obj=folder/(meshname+'.obj')
            lines=['v '+values(p) for p in points]+['vt '+values(p) for p in uvs]+['f '+' '.join(f'{x+1}/{x+1}' for x in tri) for tri in triangles]
            obj.write_text('\n'.join(lines)+'\n')
            # Captured panels are often open or exactly planar. Shell inertia
            # permits those visual-only meshes without fabricated thickness.
            ET.SubElement(asset,'mesh',name=meshname,file=f'photo/{meshname}.obj',inertia='shell')
            ET.SubElement(parent,'geom',name=meshname,type='mesh',mesh=meshname,material='photo_mat_'+view,contype='0',conaffinity='0',mass='0',group='1')
            mesh=UsdGeom.Mesh.Define(stage,usdpath+'/'+meshname);mesh.CreatePointsAttr(points.tolist());mesh.CreateFaceVertexCountsAttr([3]*len(triangles));mesh.CreateFaceVertexIndicesAttr(triangles.flatten().tolist());mesh.CreateSubdivisionSchemeAttr('none');mesh.CreateDoubleSidedAttr(True)
            UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.vertex).Set(uvs.tolist());UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
            counts[name]+=len(triangles);mesh_count+=1
    # Reuse independently audited source cameras, shifted into the scene frame.
    source_world=ET.parse(geometry/'photo_visual.xml').getroot().find('worldbody')
    for camera in source_world.findall('camera'):
        camera.set('pos',values(np.fromstring(camera.get('pos'),sep=' ')+shift));world.append(camera)
    ET.indent(root);tree.write(output/'scene.xml',encoding='unicode');stage.GetRootLayer().Save()
    report=dict(meshes=mesh_count,triangles_by_body=counts,static_geometry=str(geometry),dynamic_geometry=str(dynamic_geometry) if dynamic_geometry else str(geometry),collision=False,ownership='proximity to fitted collision bodies with class color gates; majority per triangle',floor_projection_tolerance_m=.035,chair_panel_projection_tolerance_m=.055,limitations=['Occluded faces retain estimated primitive fallbacks.','Lighting and shadows are baked into source pixels.','Glass and occlusion boundaries contain holes.','Ownership near object boundaries is approximate.','Cabinet door joint is inferred, with primitive appearance.'])
    (output/'photo_ownership.json').write_text(json.dumps(report,indent=2));return report
