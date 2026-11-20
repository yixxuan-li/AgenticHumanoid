"""Extract source-grounded furniture clusters and planes from dense video geometry."""
import json
import os
import argparse
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/agentichumanoid-matplotlib')
import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def voxelize(points,colors=None,size=.018):
    cells=np.floor(points/size).astype(int);_,inverse=np.unique(cells,axis=0,return_inverse=True);n=inverse.max()+1;count=np.bincount(inverse,minlength=n)
    center=np.column_stack([np.bincount(inverse,weights=points[:,j],minlength=n)/count for j in range(3)])
    rgb=None if colors is None else np.column_stack([np.bincount(inverse,weights=colors[:,j],minlength=n)/count for j in range(3)])
    return center,rgb


def clusters(points,eps=.07,min_points=40):
    if len(points)<min_points:return []
    pairs=cKDTree(points).query_pairs(eps,output_type='ndarray');graph=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(len(points),len(points)))
    _,labels=connected_components(graph,directed=False);counts=np.bincount(labels)
    return [points[labels==i] for i in np.argsort(counts)[::-1] if counts[i]>=min_points]


def describe(points,kind):
    bounds=np.percentile(points,[1,99],axis=0);center=np.mean(bounds,axis=0);extent=bounds[1]-bounds[0]
    result=dict(kind=kind,points=len(points),center=center.tolist(),bounds=bounds.tolist(),extent=extent.tolist())
    if kind=='horizontal white surface':
        center2,size,angle=cv2.minAreaRect(points[:,:2].astype('float32'))
        if size[0]<size[1]:size=(size[1],size[0]);angle+=90
        result['oriented_rectangle']=dict(center=list(center2),size=list(size),yaw_rad=float(np.deg2rad(angle)))
    seat=points[(points[:,2]>.36)&(points[:,2]<.57)];back=points[(points[:,2]>.60)&(points[:,2]<1.05)]
    if len(seat)>20 and len(back)>20:
        s=np.median(seat[:,:2],axis=0);b=np.median(back[:,:2],axis=0);v=b-s
        result.update(seat_xy=s.tolist(),back_xy=b.tolist(),back_vector=v.tolist(),suggested_yaw_rad=float(np.arctan2(v[1],v[0])-np.pi/2))
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--geometry',type=Path,default=Path('outputs/geometry'));parser.add_argument('--room-envelope',type=Path);args=parser.parse_args();folder=args.geometry
    data=np.load(folder/'aligned_geometry.npz');raw=np.load(folder/'predictions.npz');p=data['points'];rgb=data['colors'].astype(float);valid=data['visual_valid']
    green=valid&(p[...,2]>.30)&(p[...,2]<1.05)&(rgb[...,1]>rgb[...,0]*1.25)&(rgb[...,1]>rgb[...,2]*1.35)&(rgb[...,1]>70)
    gp,_=voxelize(p[green],size=.015);gc=clusters(gp,eps=.065,min_points=60)
    dx=p[:,1:-1,2:]-p[:,1:-1,:-2];dy=p[:,2:,1:-1]-p[:,:-2,1:-1];normal=np.cross(dx,dy);normal/=np.maximum(np.linalg.norm(normal,axis=-1,keepdims=True),1e-9)
    normal_z=np.zeros(valid.shape);normal_z[:,1:-1,1:-1]=abs(normal[...,2])
    horizontal=normal_z>.90
    white=valid&horizontal&(abs(p[...,2]-.74)<.035)&(rgb.mean(-1)>140)&(rgb.max(-1)-rgb.min(-1)<65)
    wp,_=voxelize(p[white],size=.02);wc=clusters(wp,eps=.065,min_points=90)
    report=dict(frame='dense aligned coordinates; assumed 0.74 m table anchor',green_chairs=[describe(c,'green chair candidate') for c in gc],white_tabletop_components=[describe(c,'horizontal white surface') for c in wc])
    if args.room_envelope:report['room_envelope']=json.loads(args.room_envelope.read_text())['recommended_rectangular_envelope']
    monitor_region=(p[...,0]<.55)|(p[...,0]>5.30)|((p[...,1]>4.15)&(p[...,1]<4.80))
    dark=valid&monitor_region&(p[...,2]>.79)&(p[...,2]<1.35)&(normal_z<.35)&(rgb.max(-1)<110)&(rgb.max(-1)-rgb.min(-1)<65)
    dp,_=voxelize(p[dark],size=.018);dc=clusters(dp,eps=.055,min_points=35)
    report['black_vertical_components']=[]
    for component in dc:
        item=describe(component,'possible monitor face; requires source-image review');_,_,vectors=np.linalg.svd(component-component.mean(0),full_matrices=False);item['plane_normal']=vectors[-1].tolist();report['black_vertical_components'].append(item)
    yellow=valid&(p[...,0]<.8)&(p[...,1]>2.6)&(p[...,2]>1.4)&(p[...,2]<2.5)&(rgb[...,0]>140)&(rgb[...,1]>120)&(rgb[...,2]<190)&(rgb[...,0]>1.05*rgb[...,1])&(rgb[...,1]>1.12*rgb[...,2])
    yp,_=voxelize(p[yellow],size=.018);yc=clusters(yp,eps=.07,min_points=50)
    report['yellow_wall_cabinet_components']=[describe(c,'observed yellow wall cabinet face') for c in yc]
    for chair in report['green_chairs']:
        lo,hi=np.array(chair['bounds']);region=green&(normal_z<.4)&(p[...,2]>.60)&np.all(p>lo-.01,axis=-1)&np.all(p<hi+.01,axis=-1)
        back=p[region]
        if len(back)>50:
            chair['back_surface_center']=np.median(back,axis=0).tolist()
            _,_,vectors=np.linalg.svd(back-back.mean(0),full_matrices=False);chair['back_surface_normal']=vectors[-1].tolist()
    def region_points(frame,bbox,kind,zmin=.05,zmax=1.2,surface=None):
        i=list(raw['frames']).index(f'outputs/inspection/frame_{frame:04d}.jpg');left,top,nw,nh,ow,oh=raw['image_bounds'][i]
        x0,y0,x1,y1=bbox;x0=round(left+x0*nw/ow);x1=round(left+x1*nw/ow);y0=round(top+y0*nh/oh);y1=round(top+y1*nh/oh)
        region=np.zeros(valid[i].shape,bool);region[y0:y1,x0:x1]=True;region&=valid[i]&(p[i,...,2]>zmin)&(p[i,...,2]<zmax)
        if kind=='dark':region&=(rgb[i].max(-1)<120)&(rgb[i].max(-1)-rgb[i].min(-1)<45)
        elif kind=='gray':region&=(rgb[i].mean(-1)>95)&(rgb[i].mean(-1)<195)&(rgb[i].max(-1)-rgb[i].min(-1)<45)
        elif kind=='white':region&=(rgb[i].mean(-1)>150)&(rgb[i].max(-1)-rgb[i].min(-1)<60)
        if surface=='horizontal':region&=normal_z[i]>.65
        elif surface=='vertical':region&=normal_z[i]<.5
        return p[i][region]
    report['monitor_faces']={}
    monitors=[('left_monitor_0',27,[84,290,138,330],'dark'),('left_monitor_1',27,[205,275,273,310],'dark'),
        ('right_monitor_0',5,[66,275,112,301],'any'),('right_monitor_1',9,[239,265,275,296],'any'),('right_monitor_2',9,[128,241,164,270],'dark'),
        ('window_monitor_0',17,[96,206,140,235],'dark'),('window_monitor_1',17,[150,198,204,226],'dark'),('window_monitor_2',17,[224,202,273,233],'any')]
    from scipy.spatial.transform import Rotation
    for name,frame,bbox,kind in monitors:
        points=region_points(frame,bbox,kind,.80,1.35,'vertical')
        if len(points)<25:continue
        _,_,vectors=np.linalg.svd(points-points.mean(0),full_matrices=False);normal=vectors[-1]
        yaw=float(np.arctan2(normal[1],normal[0])-np.pi/2);tilt=float(np.arcsin(np.clip(normal[2],-1,1)));R=Rotation.from_euler('z',yaw)*Rotation.from_euler('x',tilt)
        local=R.inv().apply(points);bounds=np.percentile(local,[1,99],axis=0);middle=np.mean(bounds,axis=0);middle[1]=np.median(local[:,1]);center=R.apply(middle)
        width,height=float(bounds[1,0]-bounds[0,0]+.035),float(bounds[1,2]-bounds[0,2]+.035)
        if not (.15<width<1. and .12<height<.65):continue
        report['monitor_faces'][name]=dict(center=center.tolist(),yaw_rad=yaw,tilt_rad=tilt,width_m=width,height_m=height,source_frame=frame,source_bbox=bbox,points=len(points),median_plane_residual_m=float(np.median(np.abs(local[:,1]-middle[1]))))
    rois=[('office_window',9,[160,310,285,505],'dark'),('office_table_end',25,[165,300,305,550],'dark'),('gray_chair',27,[170,350,255,425],'gray')]
    roi_points={}
    for name,frame,bbox,kind in rois:
        pts,_=voxelize(region_points(frame,bbox,kind,.08,1.15),size=.015);cs=clusters(pts,eps=.08,min_points=25)
        roi_points[name]=cs[0] if cs else pts
        report[name]=describe(roi_points[name],kind)
    focused=[('office_window',9,[175,378,250,422],[175,315,235,375],'dark'),('office_table_end',25,[215,431,290,480],[178,320,260,408],'dark'),('gray_chair',27,[175,396,243,419],[180,355,240,395],'gray')]
    for name,frame,seatbox,backbox,kind in focused:
        broad=next(r[2] for r in rois if r[0]==name)
        seat=region_points(frame,broad,kind,.37,.56,'horizontal');back=region_points(frame,backbox,kind,.61,1.05,'vertical')
        if len(seat)>15 and len(back)>15:
            s=np.median(seat[:,:2],axis=0);b=np.median(back[:,:2],axis=0);v=b-s
            report[name]['focused_pose']=dict(seat_xy=s.tolist(),back_xy=b.tolist(),yaw_rad=float(np.arctan2(v[1],v[0])-np.pi/2),seat_points=len(seat),back_points=len(back))
    report['source_landmarks']={}
    for name,frame,bbox,kind,zmin,zmax in [('cabinet_front',1,[174,181,262,510],'gray',.1,2.1),('whiteboard',33,[80,135,280,310],'any',.8,2.2),('storage_bin',23,[185,303,248,331],'white',.2,.7),('observed_small_box',21,[107,315,143,338],'dark',.72,.87),('window_bottle',17,[74,197,98,247],'gray',.75,1.4)]:
        points=region_points(frame,bbox,kind,zmin,zmax)
        if len(points)>15:
            report['source_landmarks'][name]=describe(points,name)
            _,_,v=np.linalg.svd(points-points.mean(0),full_matrices=False);n=v[-1]
            report['source_landmarks'][name]['plane_normal']=n.tolist()
    cloud,colors=voxelize(p[valid&(p[...,2]>.05)&(p[...,2]<1.4)],rgb[valid&(p[...,2]>.05)&(p[...,2]<1.4)]/255,size=.025)
    fig,axes=plt.subplots(1,2,figsize=(14,7),layout='constrained')
    for ax in axes:ax.scatter(cloud[:,0],cloud[:,1],s=2,c=np.clip(colors,0,1));ax.set_aspect('equal');ax.set(xlabel='X [provisional m]',ylabel='Y [provisional m]');ax.grid(alpha=.15)
    for i,c in enumerate(report['green_chairs']):
        ax=axes[0];xy=c['center'];ax.text(xy[0],xy[1],f'G{i}',color='red',fontsize=12,bbox=dict(facecolor='white',alpha=.7,edgecolor='none'))
    for i,c in enumerate(report['white_tabletop_components']):
        xy=c['center'];axes[0].text(xy[0],xy[1],f'T{i}',color='blue',fontsize=12)
    for name,pts in roi_points.items():
        center=np.mean(pts,axis=0);axes[0].text(center[0],center[1],name,color='purple')
    manifest=json.loads(Path('outputs/scene/manifest.json').read_text())
    for b in manifest['bodies']:
        if 'chair' not in b['name'] and 'table' not in b['name']:continue
        x,y=b['pos'][:2];axes[1].plot(x,y,'rx');axes[1].text(x,y,b['name'],fontsize=7,color='red')
    axes[0].set_title('Dense source-color clusters and tabletop components');axes[1].set_title('Prototype body origins over recovered geometry')
    fig.savefig(folder/'furniture_alignment.png',dpi=160);plt.close(fig)
    (folder/'scene_features.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    # Project cluster centroids back to source frames for semantic review.
    from PIL import Image,ImageDraw
    Q=data['rotation'];translation=data['translation'];scale=float(data['scale']);tiles=[]
    for ci,c in enumerate(report['green_chairs']):
        center=np.array(c['center']);modelpoint=(center-translation)@Q/scale
        for frameid in [5,9,13,17,21,25,29,33]:
            index=list(raw['frames']).index(f'outputs/inspection/frame_{frameid:04d}.jpg');e=raw['extrinsic'][index];K=raw['intrinsic'][index]
            cam=e[:,:3]@modelpoint+e[:,3]
            if cam[2]<=0:continue
            pixel=K@cam;pixel=pixel[:2]/pixel[2];left,top,nw,nh,ow,oh=raw['image_bounds'][index]
            u=(pixel[0]-left)*ow/nw;v=(pixel[1]-top)*oh/nh
            if 0<=u<ow and 0<=v<oh:
                image=Image.open(raw['frames'][index]);crop=image.crop((max(0,u-45),max(0,v-60),min(ow,u+45),min(oh,v+60))).resize((144,192))
                tile=Image.new('RGB',(144,218),'#172326');tile.paste(crop,(0,0));ImageDraw.Draw(tile).text((4,196),f'G{ci} / frame {frameid}',fill='white');tiles.append(tile)
    sheet=Image.new('RGB',(144*8,218*((len(tiles)+7)//8)),'#172326')
    for i,tile in enumerate(tiles):sheet.paste(tile,((i%8)*144,(i//8)*218))
    sheet.save(folder/'green_cluster_review.jpg')


if __name__=='__main__':main()
