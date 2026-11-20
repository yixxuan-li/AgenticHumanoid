"""Consistent MJCF/USD export from one semantic scene manifest."""
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from .layout import COLORS


def values(x):
    return ' '.join(f'{float(v):.8g}' for v in x)


def volume(part):
    s=part['size']
    return np.prod(s) if part['shape']=='box' else math.pi*s[0]**2*s[1]


def export_mjcf(manifest,output: Path):
    scale=manifest.get('global_scale',1.)
    root=ET.Element('mujoco',model='video_office_provisional')
    ET.SubElement(root,'compiler',angle='radian',autolimits='true')
    ET.SubElement(root,'option',timestep='0.002',gravity='0 0 -9.81',integrator='implicitfast',cone='elliptic',iterations='100')
    size=manifest['dimensions']
    ET.SubElement(root,'statistic',center=values([size[0]/2,size[1]/2,1.0]),extent=str(max(size)))
    visual=ET.SubElement(root,'visual')
    ET.SubElement(visual,'global',offwidth='1280',offheight='960')
    ET.SubElement(visual,'quality',shadowsize='2048')
    ET.SubElement(visual,'headlight',ambient='.32 .32 .32',diffuse='.65 .65 .65',specular='.08 .08 .08')
    defaults=ET.SubElement(root,'default')
    ET.SubElement(defaults,'geom',friction='.65 .008 .0002',solref='.015 1',solimp='.95 .99 .001',condim='4',margin='.0005')
    ET.SubElement(defaults,'joint',damping='.02')
    asset=ET.SubElement(root,'asset')
    ET.SubElement(asset,'texture',name='sky',type='skybox',builtin='gradient',rgb1='.77 .79 .81',rgb2='.77 .79 .81',width='256',height='1536')
    for name,color in COLORS.items():
        attr=dict(name=name,rgba=values(color),roughness='.8',specular='.12',shininess='.1')
        if name=='lamp':attr['emission']='.8'
        ET.SubElement(asset,'material',**attr)
    for name in sorted({p['texture'] for b in manifest['bodies'] for p in b['parts'] if p['texture']}):
        ET.SubElement(asset,'texture',name=f'tex_{name}',type='2d',file=f'textures/{name}.png')
        ET.SubElement(asset,'material',name=f'mat_{name}',texture=f'tex_{name}',rgba='1 1 1 1',texuniform='false',specular='.13',shininess='.1')
    # Explicit per-face UVs prevent the box texture projection collapsing wall
    # and monitor images into stripes on vertical faces.
    for body in manifest['bodies']:
        for part in body['parts']:
            if (part['texture'] or part.get('visual_style')) and part['shape']=='box':
                if part.get('visual_style'):
                    vertices,faces=rounded_box(part['size']);uvs=[(0,0)]*len(vertices)
                else:
                    vertices,_,uvs=cube_mesh(part['size']);faces=[]
                    for face in range(6):
                        a=face*4;faces += [[a,a+1,a+2],[a,a+2,a+3]]
                ET.SubElement(asset,'mesh',name=f"uv_{body['name']}__{part['name']}",vertex=values(np.array(vertices).flatten()),face=values(np.array(faces).flatten()),texcoord=values(np.array(uvs).flatten()))
    world=ET.SubElement(root,'worldbody')
    for i,x in enumerate([1.2,3.2,5.2]):
        ET.SubElement(world,'light',name=f'ceiling_light_{i}',pos=values(np.array([x,2.5,2.7])*scale),dir='0 0 -1',diffuse='.55 .53 .48',ambient='.18 .18 .18',castshadow='true')
    cameras=[('overview',[3.2,-2.0,5.8],[3.2,2.5,.7]),('entry',[1.85,.45,1.55],[3.2,3.0,1.0]),('window_view',[3.1,.85,1.65],[3.2,4.6,1.2]),('right_view',[3.1,.85,1.65],[6.3,1.5,1.3])]
    cameras=manifest.get('cameras',cameras)
    for name,pos,target in cameras:
        z=np.array(pos)-target;z/=np.linalg.norm(z);x=np.cross([0,0,1],z);x/=np.linalg.norm(x);y=np.cross(z,x)
        ET.SubElement(world,'camera',name=name,pos=values(np.array(pos)*scale),xyaxes=values(np.r_[x,y]),fovy='64')
    # A transparent ceiling closes the physical volume without hiding interior previews.
    ET.SubElement(world,'geom',name='ceiling_collision',type='box',pos=values([size[0]/2,size[1]/2,size[2]+.06*scale]),size=values([size[0]/2,size[1]/2,.06*scale]),rgba='1 1 1 0',group='3')
    for body in manifest['bodies']:
        b=ET.SubElement(world,'body',name=body['name'],pos=values(body['pos']),euler=values([0,0,body['yaw']]))
        if body['kind']=='free':ET.SubElement(b,'freejoint',name=body['name']+'_free')
        elif body['kind']=='hinge':
            j=body['joint'];ET.SubElement(b,'joint',name=body['name']+'_hinge',type='hinge',axis=values(j['axis']),range=values(j['range']),damping=str(j['damping']),frictionloss=str(j['friction']))
        parts=body['parts'];vol=sum(volume(p) for p in parts)
        for part in parts:
            s=part['size']
            sz=np.array(s)/2 if part['shape']=='box' else [s[0],s[1]/2]
            attr=dict(name=body['name']+'__'+part['name'],type=part['shape'],pos=values(part['pos']),size=values(sz),euler=values(part['rotation']),material=f"mat_{part['texture']}" if part['texture'] else part['material'])
            if (part['texture'] or part.get('visual_style')) and part['shape']=='box':
                vis=attr.copy();vis.pop('size');vis.update(type='mesh',mesh=f"uv_{body['name']}__{part['name']}",contype='0',conaffinity='0',mass='0',group='1')
                vis['name']+='__visual'
                if part.get('visible') is not False:ET.SubElement(b,'geom',**vis)
                attr.pop('material');attr['rgba']='1 1 1 0';attr['group']='3'
            if part.get('visible') is False:
                attr.pop('material',None);attr['rgba']='1 1 1 0';attr['group']='3'
            if not part['collision']:attr.update(contype='0',conaffinity='0',group='1')
            if body['mass']>0:attr['mass']=str(float(body['mass']*volume(part)/vol))
            ET.SubElement(b,'geom',**attr)
        if body.get('affordances'):
            ET.SubElement(b,'site',name=body['name']+'_grasp',pos='0 0 0',size='.008',rgba='0 0 0 0',group='4')
    ET.indent(root)
    ET.ElementTree(root).write(output/'scene.xml',encoding='unicode')


def cube_mesh(size):
    x,y,z=np.array(size)/2
    vertices=[(-x,-y,z),(x,-y,z),(x,y,z),(-x,y,z), # top
              (-x,y,-z),(x,y,-z),(x,-y,-z),(-x,-y,-z), # bottom
              (-x,-y,-z),(x,-y,-z),(x,-y,z),(-x,-y,z),
              (x,-y,-z),(x,y,-z),(x,y,z),(x,-y,z),
              (x,y,-z),(-x,y,-z),(-x,y,z),(x,y,z),
              (-x,y,-z),(-x,-y,-z),(-x,-y,z),(-x,y,z)]
    return vertices,list(range(24)),[(0,0),(1,0),(1,1),(0,1)]*6


def rounded_box(size):
    """Tessellated rounded visual shell; the collision proxy remains a box."""
    half=np.array(size)/2;radius=min(.035,float(min(half))*.8)
    vertices=[];faces=[]
    grid=np.array([-1.,-.92,-.75,-.4,0,.4,.75,.92,1.]);k=len(grid)
    for axis in range(3):
        other=[a for a in range(3) if a!=axis]
        for sign in [-1,1]:
            start=len(vertices)
            for b in grid:
                for a in grid:
                    p=np.zeros(3);p[axis]=sign*half[axis];p[other]=[a*half[other[0]],b*half[other[1]]]
                    q=np.clip(p,-half+radius,half-radius);normal=p-q;p=q+radius*normal/np.linalg.norm(normal);vertices.append(p.tolist())
            for y in range(k-1):
                for x in range(k-1):
                    a=start+y*k+x;b=a+1;c=a+k+1;d=a+k
                    tris=[[a,b,c],[a,c,d]]
                    test=np.cross(np.array(vertices[b])-vertices[a],np.array(vertices[c])-vertices[a])
                    if test[axis]*sign<0:tris=[t[::-1] for t in tris]
                    faces+=tris
    return vertices,faces


def export_usd(manifest,output: Path):
    from pxr import Gf,Sdf,Tf,Usd,UsdGeom,UsdPhysics,UsdShade,Vt
    stage=Usd.Stage.CreateNew(str(output/'scene.usda'))
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z);UsdGeom.SetStageMetersPerUnit(stage,1.)
    UsdPhysics.SetStageKilogramsPerUnit(stage,1.)
    world=UsdGeom.Xform.Define(stage,'/World');stage.SetDefaultPrim(world.GetPrim())
    scene=UsdPhysics.Scene.Define(stage,'/World/PhysicsScene');scene.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1));scene.CreateGravityMagnitudeAttr(9.81)
    materials={}
    for name,color in list(COLORS.items())+[(f'mat_{k}',[1,1,1,1]) for k in sorted({p['texture'] for b in manifest['bodies'] for p in b['parts'] if p['texture']})]:
        path=f'/World/Looks/{name}';material=UsdShade.Material.Define(stage,path);shader=UsdShade.Shader.Define(stage,path+'/Shader')
        shader.CreateIdAttr('UsdPreviewSurface');shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.78)
        shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color[:3]))
        if name=='lamp':shader.CreateInput('emissiveColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.5,1.5,1.4))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
        if name.startswith('mat_'):
            uv=UsdShade.Shader.Define(stage,path+'/UV');uv.CreateIdAttr('UsdPrimvarReader_float2');uv.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st')
            tex=UsdShade.Shader.Define(stage,path+'/Texture');tex.CreateIdAttr('UsdUVTexture')
            tex.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(f'textures/{name[4:]}.png'))
            tex.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('sRGB')
            tex.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(),'result')
            tex.CreateOutput('rgb',Sdf.ValueTypeNames.Float3)
            shader.GetInput('diffuseColor').ConnectToSource(tex.ConnectableAPI(),'rgb')
        physics=UsdPhysics.MaterialAPI.Apply(material.GetPrim());physics.CreateStaticFrictionAttr(.65);physics.CreateDynamicFrictionAttr(.60);physics.CreateRestitutionAttr(.02)
        materials[name]=material
    for b in manifest['bodies']:
        path='/World/Scene/'+Tf.MakeValidIdentifier(b['name']);xf=UsdGeom.Xform.Define(stage,path)
        xf.AddTranslateOp().Set(Gf.Vec3d(*b['pos']));xf.AddRotateZOp().Set(math.degrees(b['yaw']))
        prim=xf.GetPrim();prim.SetCustomDataByKey('evidence',b['evidence']);prim.SetCustomDataByKey('scaleStatus',manifest['scale_status'])
        if b['mass']>0:
            UsdPhysics.RigidBodyAPI.Apply(prim);UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(b['mass'])
        for p in b['parts']:
            partpath=path+'/'+Tf.MakeValidIdentifier(p['name'])
            part=UsdGeom.Xform.Define(stage,partpath)
            part.AddTranslateOp().Set(Gf.Vec3d(*p['pos']));part.AddRotateXYZOp().Set(Gf.Vec3f(*np.degrees(p['rotation'])))
            if p['shape']=='box':
                if p.get('visual_style'):
                    v,faces=rounded_box(p['size']);indices=np.array(faces).flatten().tolist();uvs=[(0,0)]*len(indices);counts=[3]*len(faces)
                else:v,indices,uvs=cube_mesh(p['size']);counts=[4]*6
                mesh=UsdGeom.Mesh.Define(stage,partpath+'/Visual')
                mesh.CreatePointsAttr(v);mesh.CreateFaceVertexCountsAttr(counts);mesh.CreateFaceVertexIndicesAttr(indices)
                mesh.CreateSubdivisionSchemeAttr('none');mesh.CreateDoubleSidedAttr(False)
                st=UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.faceVarying);st.Set(uvs)
                if p['collision']:
                    c=UsdGeom.Cube.Define(stage,partpath+'/Collision');c.CreateSizeAttr(1.);c.AddScaleOp().Set(Gf.Vec3f(*p['size']));c.CreateVisibilityAttr('invisible');UsdPhysics.CollisionAPI.Apply(c.GetPrim())
            else:
                mesh=UsdGeom.Cylinder.Define(stage,partpath+'/Visual');mesh.CreateRadiusAttr(p['size'][0]);mesh.CreateHeightAttr(p['size'][1]);mesh.CreateAxisAttr('Z')
                if p['collision']:UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            mat=materials['mat_'+p['texture'] if p['texture'] else p['material']]
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
            if p.get('visible') is False:mesh.CreateVisibilityAttr('invisible')
            if p['collision']:
                cp=stage.GetPrimAtPath(partpath+'/Collision') if p['shape']=='box' else mesh.GetPrim()
                UsdShade.MaterialBindingAPI.Apply(cp).Bind(mat,materialPurpose='physics')
        if b['kind']=='hinge':
            # A world joint ignores the placement of a referenced room. A
            # collider-free kinematic anchor follows the room's parent transform.
            anchorpath='/World/Anchors/'+Tf.MakeValidIdentifier(b['name'])
            anchor=UsdGeom.Xform.Define(stage,anchorpath)
            anchor.AddTranslateOp().Set(Gf.Vec3d(*b['pos']))
            anchor.AddRotateZOp().Set(math.degrees(b['yaw']))
            anchor.GetPrim().SetCustomDataByKey('sceneJointAnchor',True)
            UsdPhysics.RigidBodyAPI.Apply(anchor.GetPrim()).CreateKinematicEnabledAttr(True)
            mass=UsdPhysics.MassAPI.Apply(anchor.GetPrim());mass.CreateMassAttr(1.)
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(.001,.001,.001))
            joint=UsdPhysics.RevoluteJoint.Define(stage,'/World/Joints/'+b['name'])
            joint.CreateBody0Rel().SetTargets([Sdf.Path(anchorpath)])
            joint.CreateBody1Rel().SetTargets([Sdf.Path(path)]);joint.CreateAxisAttr('Z')
            joint.CreateLocalPos0Attr(Gf.Vec3f(0));joint.CreateLocalPos1Attr(Gf.Vec3f(0))
            joint.CreateLocalRot0Attr(Gf.Quatf(1.,Gf.Vec3f(0)))
            joint.CreateLocalRot1Attr(Gf.Quatf(1.,Gf.Vec3f(0)))
            joint.CreateLowerLimitAttr(math.degrees(b['joint']['range'][0]));joint.CreateUpperLimitAttr(math.degrees(b['joint']['range'][1]))
    # Physical ceiling matches MuJoCo, hidden in preview.
    dims=manifest['dimensions'];scale=manifest.get('global_scale',1.);c=UsdGeom.Cube.Define(stage,'/World/Scene/ceiling_collision');c.CreateSizeAttr(1)
    c.AddTranslateOp().Set(Gf.Vec3d(dims[0]/2,dims[1]/2,dims[2]+.06*scale));c.AddScaleOp().Set(Gf.Vec3f(dims[0],dims[1],.12*scale));c.CreateVisibilityAttr('invisible');UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    stage.GetRootLayer().Save()


def export_all(manifest,output):
    output.mkdir(parents=True,exist_ok=True)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    export_mjcf(manifest,output);export_usd(manifest,output)
