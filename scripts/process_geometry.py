"""Audit learned multiview geometry and export photo-textured surface hypotheses.

This script never treats learned scale as measured scale. A source-specific
table/floor anchor is explicitly reported as assumed, and poor fits are rejected.
"""
import argparse
import json
import math
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from run_vggt import write_ply


def plane_ransac(points,hint,threshold):
    rng=np.random.default_rng(119)
    if len(points)>15000:points=points[rng.choice(len(points),15000,replace=False)]
    if len(points)<20:raise ValueError('Too few floor candidates')
    best=np.zeros(len(points),bool)
    for _ in range(1500):
        a,b,c=points[rng.choice(len(points),3,replace=False)];n=np.cross(b-a,c-a);norm=np.linalg.norm(n)
        if norm<1e-8:continue
        n/=norm
        if abs(n@hint)<math.cos(math.radians(25)):continue
        mask=np.abs((points-a)@n)<threshold
        if mask.sum()>best.sum():best=mask
    if best.sum()<10:raise ValueError('No plausible floor plane')
    center=points[best].mean(0);_,_,v=np.linalg.svd(points[best]-center);n=v[-1]
    if n@hint<0:n=-n
    d=-n@center
    return n,d,dict(candidates=len(points),inliers=int(best.sum()),inlier_fraction=float(best.mean()),median_residual_model_units=float(np.median(np.abs(points[best]@n+d))))


def cross_view(points,depth,conf,extrinsics,intrinsics,valid):
    reports=[];rng=np.random.default_rng(0)
    for i in range(len(points)-1):
        j=i+1;indices=np.flatnonzero(valid[i]);indices=rng.choice(indices,min(5000,len(indices)),replace=False)
        xyz=points[i].reshape(-1,3)[indices];camera=xyz@extrinsics[j,:,:3].T+extrinsics[j,:,3]
        z=camera[:,2];pixels=camera@intrinsics[j].T;uv=pixels[:,:2]/pixels[:,2:]
        h,w=depth[j].shape;inside=(uv[:,0]>=0)&(uv[:,0]<w-1)&(uv[:,1]>=0)&(uv[:,1]<h-1)&(z>0)
        uv=uv[inside];z=z[inside]
        if len(z)<30:reports.append(dict(a=i,b=j,status='insufficient overlap'));continue
        reference=cv2.remap(depth[j],uv[:,0].astype('f4')[:,None],uv[:,1].astype('f4')[:,None],cv2.INTER_LINEAR).ravel()
        target_mask=cv2.remap(valid[j].astype('u1'),uv[:,0].astype('f4')[:,None],uv[:,1].astype('f4')[:,None],cv2.INTER_NEAREST).ravel()>0
        good=(reference>0)&target_mask;rel=np.abs(z[good]-reference[good])/reference[good]
        reports.append(dict(a=i,b=j,samples=len(rel),median_relative_depth_error=float(np.median(rel)),fraction_within_10_percent=float((rel<.10).mean())))
    return reports


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=Path('outputs/geometry/predictions.npz'));p.add_argument('--alignment-config',type=Path,help='Source-specific floor/anchor pixel ROIs, anchor height, and forward camera');p.add_argument('--sfm',type=Path,help='Independent COLMAP model for this exact video');p.add_argument('--mesh-stride',type=int,default=3);p.add_argument('--mesh-confidence-percentile',type=float,default=5.);p.add_argument('--report',type=Path,default=Path('reports/dense_geometry.json'));args=p.parse_args()
    folder=args.input.parent;data=np.load(args.input);points=data['points'];colors=data['colors'];conf=data['confidence'];depth=data['depth'][...,0];extr=data['extrinsic'];intr=data['intrinsic'];frames=data['frames']
    n,h,w,_=points.shape
    cfg=json.loads(args.alignment_config.read_text()) if args.alignment_config else None
    def roi_mask(samples):
        mask=np.zeros((n,h,w),bool)
        for sample in samples:
            matches=[i for i,frame in enumerate(frames) if Path(frame).name==sample['frame']]
            if len(matches)!=1:raise ValueError(f"ROI frame not found exactly once: {sample['frame']}")
            i=matches[0];left,top,nw,nh,ow,oh=data['image_bounds'][i];x0,y0,x1,y1=sample['bbox']
            if not (0<=x0<x1<=ow and 0<=y0<y1<=oh):raise ValueError('ROI bbox must use original upright image pixels')
            x0,x1=np.rint(left+np.array([x0,x1])*nw/ow).astype(int);y0,y1=np.rint(top+np.array([y0,y1])*nh/oh).astype(int)
            mask[i,y0:y1,x0:x1]=True
        return mask
    pixel_mask=data['pixel_mask'] if 'pixel_mask' in data else np.ones(conf.shape,bool)
    thresholds=np.array([np.percentile(conf[i][pixel_mask[i]],35) for i in range(n)])[:,None,None]
    valid=np.isfinite(points).all(-1)&pixel_mask&(depth>0)&(conf>thresholds)
    visual_thresholds=np.array([np.percentile(conf[i][pixel_mask[i]],args.mesh_confidence_percentile) for i in range(n)])[:,None,None]
    visual_valid=np.isfinite(points).all(-1)&pixel_mask&(depth>0)&(conf>visual_thresholds)
    consistency=cross_view(points,depth,conf,extr,intr,valid)
    hints=np.mean([-e[:,:3].T[:,1] for e in extr],axis=0);hints/=np.linalg.norm(hints)
    yy=np.broadcast_to(np.arange(h)[None,:,None],valid.shape)
    rgb=colors.astype(float)
    floormask=valid&(yy>h*.80)&(rgb[...,0]>rgb[...,1]*1.07)&(rgb[...,1]>rgb[...,2]*1.08)&(rgb.min(-1)>35)
    if cfg:floormask=valid&roi_mask(cfg['floor_samples'])
    report=dict(source='official VGGT learned depth/cameras',cross_view=consistency,scale_status='unmeasured',alignment_status='not attempted',
        caveats=['Depth consistency measures agreement between learned predictions, not ground-truth accuracy.','No view can reconstruct an unseen backside uniquely.'])
    sfmpath=args.sfm
    if sfmpath and sfmpath.exists():
        try:
            import pycolmap
            rec=pycolmap.Reconstruction(sfmpath);lookup={im.name:im for im in rec.images.values()};gauges=[]
            for i,path in enumerate(frames):
                if Path(path).name in lookup:gauges.append(lookup[Path(path).name].cam_from_world().rotation.matrix().T@extr[i,:,:3])
            rotations=Rotation.from_matrix(np.array(gauges));mean=rotations.mean();angles=np.degrees((mean.inv()*rotations).magnitude())
            report['independent_sparse_camera_orientation_agreement']=dict(common_frames=len(gauges),median_rotation_residual_deg=float(np.median(angles)),max_rotation_residual_deg=float(np.max(angles)),caveat='Camera orientation agreement does not certify either method\'s depth or scale.')
        except Exception as exc:report['sparse_comparison_error']=str(exc)
    aligned=points.copy();rotation=np.eye(3);translation=np.zeros(3);scale=1.
    try:
        spread=np.linalg.norm(np.percentile(points[valid],95,axis=0)-np.percentile(points[valid],5,axis=0))
        normal,offset,floor=plane_ransac(points[floormask],hints,spread*.003)
        floor_distance=points@normal+offset
        tablemask=np.zeros(valid.shape,bool)
        for i,frame in enumerate(frames):
            frameid=int(Path(frame).stem.split('_')[-1])
            if frameid not in [15,17,19,21]:continue
            # Original 320x568 image has been scaled/padded by official pad loader.
            y0,y1=int(h*360/568),int(h*400/568);pad=(w-h*320/568)/2
            x0,x1=int(pad+h*60/568),int(pad+h*260/568)
            tablemask[i,y0:y1,x0:x1]=True
        tablemask &= valid&(rgb.mean(-1)>135)&(rgb.max(-1)-rgb.min(-1)<60)
        if cfg:tablemask=valid&roi_mask(cfg['anchor']['samples'])
        heights=floor_distance[tablemask];heights=heights[(heights>spread*.02)&(heights<spread*.5)]
        if len(heights)<50:raise ValueError('Insufficient source tabletop pixels')
        bins=np.linspace(np.percentile(heights,5),np.percentile(heights,95),100);count,edges=np.histogram(heights,bins);k=np.argmax(count)
        tableheight=float(np.median(heights[(heights>=edges[max(k-2,0)])&(heights<=edges[min(k+3,len(edges)-1)])]))
        anchor_height=float(cfg['anchor']['height_m']) if cfg else .74
        if not .01<anchor_height<10.:raise ValueError('Anchor height must be positive meters above the fitted floor')
        scale=anchor_height/tableheight
        floorrot,_=Rotation.align_vectors([[0,0,1]],[normal]);rotation=floorrot.as_matrix()
        # Align the mid-window view direction to +Y; this is an approximate semantic
        # registration, not an independent scene measurement.
        i=min(range(n),key=lambda i:abs(int(Path(frames[i]).stem.split('_')[-1])-17))
        if cfg:i=next(i for i,path in enumerate(frames) if Path(path).name==cfg['forward_frame'])
        forward=rotation@extr[i,:,:3].T@np.array([0,0,1.]);angle=math.pi/2-math.atan2(forward[1],forward[0])
        rotation=Rotation.from_euler('z',angle).as_matrix()@rotation
        aligned=points@rotation.T*scale;aligned[...,2]+=offset*scale
        if cfg:
            origin=(-extr[0,:,:3].T@extr[0,:,3])@rotation.T*scale
            translation=np.r_[-origin[:2],offset*scale]
        else:
            mid=aligned[i,int(h*200/568),w//2]
            translation=np.array([3.2-mid[0],4.74-mid[1],offset*scale])
        aligned[...,:2]+=translation[:2]
        floor['median_residual_provisional_m']=floor['median_residual_model_units']*scale
        report.update(scale_status='assumed 0.74 m tabletop anchor; not measured',provisional_m_per_model_unit=scale,floor=floor,tabletop_height_model_units=tableheight,
            alignment_status='source-specific approximate semantic alignment',rotation=rotation.tolist(),translation_m=translation.tolist(),
            quality_checks=dict(floor_majority=floor['inlier_fraction']>.50,floor_residual_under_3cm=floor['median_residual_provisional_m']<.03,
                cross_view_majority=bool(np.median([r.get('fraction_within_10_percent',0) for r in consistency])>.50)))
        report['accepted_as_metric_ground_truth']=False
        report['candidate_for_visual_review']=all(report['quality_checks'].values())
        if cfg:
            report.update(scale_status='single measured height anchor' if cfg['anchor'].get('measured') is True else 'assumed height anchor; not measured',anchor=cfg['anchor'],alignment_status='ROI-configured floor and forward-camera alignment',alignment_config=str(args.alignment_config))
    except Exception as exc:report['alignment_error']=str(exc);report['candidate_for_visual_review']=False
    np.savez_compressed(folder/'aligned_geometry.npz',points=aligned,colors=colors,valid=valid,visual_valid=visual_valid,extrinsic=extr,intrinsic=intr,rotation=rotation,translation=translation,scale=scale)
    write_ply(folder/'aligned_cloud.ply',aligned[valid],colors[valid])
    # Textured grid surfaces preserve actual source pixels, excluding depth jumps.
    meshdir=folder/'photo_mesh';meshdir.mkdir(exist_ok=True);material_lines=[];obj=[];base=0
    views=[]
    stride=args.mesh_stride
    for i in range(n):
        xyz=aligned[i,::stride,::stride];mask=visual_valid[i,::stride,::stride];zh=depth[i,::stride,::stride]
        rows,cols=mask.shape;texture=f'view_{i:03d}.png';Image.fromarray(colors[i]).save(meshdir/texture)
        material_lines += [f'newmtl view_{i}',f'map_Kd {texture}','Kd 1 1 1','Ka 1 1 1','d 1','']
        triangles=[]
        for y in range(rows-1):
            for x in range(cols-1):
                ids=[y*cols+x,y*cols+x+1,(y+1)*cols+x+1,(y+1)*cols+x]
                if not mask[y:y+2,x:x+2].all():continue
                ds=zh[y:y+2,x:x+2]
                if ds.max()-ds.min()>.08*ds.mean():continue
                # Camera-facing winding, and identical per-vertex UV indexes.
                for tri in [[ids[0],ids[2],ids[1]],[ids[0],ids[3],ids[2]]]:
                    triangles.append(tri)
        if not triangles:continue
        triangles=np.array(triangles,dtype=int);used=np.unique(triangles);remap=np.full(rows*cols,-1,int);remap[used]=np.arange(len(used));triangles=remap[triangles]
        vertices=xyz.reshape(-1,3)[used]
        uvs=np.array([(x*stride/(w-1),1-y*stride/(h-1)) for y in range(rows) for x in range(cols)])[used]
        vertex_lines=['v '+' '.join(f'{x:.7g}' for x in p) for p in vertices]
        uv_lines=[f'vt {uv[0]:.7g} {uv[1]:.7g}' for uv in uvs]
        obj += [f'o view_{i}\nusemtl view_{i}']+vertex_lines+uv_lines
        obj += ['f '+' '.join(f'{base+a+1}/{base+a+1}' for a in tri) for tri in triangles]
        local=['mtllib scene.mtl',f'o view_{i}',f'usemtl view_{i}']+vertex_lines+uv_lines
        local += ['f '+' '.join(f'{a+1}/{a+1}' for a in tri) for tri in triangles]
        (meshdir/f'view_{i:03d}.obj').write_text('\n'.join(local)+'\n')
        views.append(dict(index=i,vertices=vertices,faces=triangles,uv=uvs))
        base+=len(used)
    (meshdir/'scene.obj').write_text('mtllib scene.mtl\n'+'\n'.join(obj)+'\n');(meshdir/'scene.mtl').write_text('\n'.join(material_lines))
    report['photo_mesh']=dict(path=str(meshdir/'scene.obj'),vertices=base,kind='overlapping per-view depth surfaces; visual hypothesis, collision disabled',contains_occlusion_holes=True,
        confidence_percentile_for_visuals=args.mesh_confidence_percentile,confidence_percentile_for_geometry_audit=35.,note='Visual review retains more low-confidence source surfaces than the geometric audit; these are not certified collision surfaces.')
    export_visual_assets(folder,views,extr,intr,rotation,translation,scale,report['scale_status'],data['image_bounds'] if 'image_bounds' in data else None)
    (folder/'geometry_audit.json').write_text(json.dumps(report,indent=2));args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


def export_visual_assets(folder,views,extr,intr,rotation,translation,scale,scale_status,image_bounds=None):
    """Standalone review assets, deliberately without physical collision shapes."""
    import xml.etree.ElementTree as ET
    from pxr import Gf,Sdf,Usd,UsdGeom,UsdShade
    root=ET.Element('mujoco',model='photo_surface_hypothesis_visual_only')
    ET.SubElement(root,'compiler',angle='radian')
    visual=ET.SubElement(root,'visual');ET.SubElement(visual,'global',offwidth='1280',offheight='960');ET.SubElement(visual,'headlight',ambient='0 0 0',diffuse='0 0 0',specular='0 0 0')
    asset=ET.SubElement(root,'asset');ET.SubElement(asset,'texture',type='skybox',builtin='gradient',rgb1='.12 .14 .16',rgb2='.12 .14 .16',width='128',height='768')
    world=ET.SubElement(root,'worldbody')
    stage=Usd.Stage.CreateNew(str(folder/'photo_visual.usda'));worldprim=UsdGeom.Xform.Define(stage,'/World');stage.SetDefaultPrim(worldprim.GetPrim());UsdGeom.SetStageUpAxis(stage,'Z');UsdGeom.SetStageMetersPerUnit(stage,1.)
    worldprim.GetPrim().SetCustomDataByKey('scaleStatus',scale_status);worldprim.GetPrim().SetCustomDataByKey('purpose','Visual depth hypothesis; no physical collision scene')
    for view in views:
        i=view['index'];name=f'view_{i:03d}'
        ET.SubElement(asset,'mesh',name=name,file=f'photo_mesh/{name}.obj',inertia='shell')
        ET.SubElement(asset,'texture',name='tex_'+name,type='2d',file=f'photo_mesh/{name}.png')
        ET.SubElement(asset,'material',name='mat_'+name,texture='tex_'+name,rgba='1 1 1 1',emission='1',specular='0',shininess='0')
        ET.SubElement(world,'geom',name=name,type='mesh',mesh=name,material='mat_'+name,contype='0',conaffinity='0',group='1')
        path='/World/'+name;mesh=UsdGeom.Mesh.Define(stage,path)
        mesh.CreatePointsAttr(view['vertices'].tolist());mesh.CreateFaceVertexCountsAttr([3]*len(view['faces']));mesh.CreateFaceVertexIndicesAttr(view['faces'].flatten().tolist());mesh.CreateSubdivisionSchemeAttr('none');mesh.CreateDoubleSidedAttr(True)
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.vertex).Set(view['uv'].tolist())
        material=UsdShade.Material.Define(stage,f'/World/Looks/{name}');shader=UsdShade.Shader.Define(stage,f'/World/Looks/{name}/Shader');shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0))
        reader=UsdShade.Shader.Define(stage,f'/World/Looks/{name}/UV');reader.CreateIdAttr('UsdPrimvarReader_float2');reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st')
        texture=UsdShade.Shader.Define(stage,f'/World/Looks/{name}/Texture');texture.CreateIdAttr('UsdUVTexture');texture.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(f'photo_mesh/{name}.png'));texture.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(),'result');texture.CreateOutput('rgb',Sdf.ValueTypeNames.Float3)
        shader.CreateInput('emissiveColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(),'rgb');material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface');UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    for i,e in enumerate(extr):
        center=(-e[:,:3].T@e[:,3])@rotation.T*scale+translation
        right=rotation@e[:,:3].T@np.array([1,0,0]);up=rotation@e[:,:3].T@np.array([0,-1,0])
        image_height=float(image_bounds[i,3]) if image_bounds is not None else 2*float(intr[i,1,2])
        fovy=math.degrees(2*math.atan(image_height/(2*float(intr[i,1,1]))))
        ET.SubElement(world,'camera',name=f'source_{i:03d}',pos=' '.join(map(str,center)),xyaxes=' '.join(map(str,np.r_[right,up])),fovy=str(fovy))
    ET.indent(root);ET.ElementTree(root).write(folder/'photo_visual.xml',encoding='unicode');stage.GetRootLayer().Save()


if __name__=='__main__':main()
