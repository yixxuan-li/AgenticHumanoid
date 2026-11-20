"""Compare real video frames excluded from depth inference against scene renders.

PnP uses held-out SIFT correspondences only to estimate review camera poses.
Appearance agreement does not establish metric scale, dynamics, or unseen detail.
"""
import argparse,json,os,sys
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('EGL_PLATFORM','surfaceless')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np,mujoco,imageio_ffmpeg
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation
from scene_reconstruction.provenance import scene_fingerprint

p=argparse.ArgumentParser();p.add_argument('--video',type=Path,default=Path('data/scene/scene.mp4'));p.add_argument('--scene',type=Path,default=Path('outputs/worldlabs_scene'));p.add_argument('--geometry',type=Path,default=Path('outputs/worldlabs_geometry'));p.add_argument('--camera',default='source_000');p.add_argument('--times',nargs='+',type=float,default=[2.,5.,8.,11.,14.]);p.add_argument('--report',type=Path,default=Path('reports/heldout_appearance_validation.json'));a=p.parse_args()
out=a.scene/'heldout';out.mkdir(exist_ok=True);archive=np.load(a.geometry/'aligned_geometry.npz');g={key:archive[key] for key in ['points','valid','intrinsic']};archive.close();raw=np.load(a.geometry/'predictions.npz');manifest=json.loads((a.scene/'manifest.json').read_text());shift=np.array(manifest.get('source_geometry_shift_m',[0,0,0]))
training=[np.asarray(Image.open(str(path)).convert('RGB')) for path in raw['frames']];small=np.stack([cv2.resize(im,(32,57)).astype(float) for im in training]);best=np.full(len(training),np.inf);best_index=np.zeros(len(training),int)
reader=imageio_ffmpeg.read_frames(str(a.video),pix_fmt='rgb24');metadata=next(reader);w,h=metadata['size'];fps=metadata['fps'];wanted=[round(t*fps) for t in a.times];heldout={}
for index,buffer in enumerate(reader):
 im=np.frombuffer(buffer,np.uint8).reshape(h,w,3);s=cv2.resize(im,(32,57)).astype(float);errors=np.mean((small-s)**2,axis=(1,2,3));improved=errors<best;best[improved]=errors[improved];best_index[improved]=index
 if index in wanted:heldout[index]=im.copy()
reader.close()
if set(wanted)&set(best_index.tolist()):raise RuntimeError('A selected review frame coincides with a model input frame')
sift=cv2.SIFT_create(nfeatures=5000);features=[sift.detectAndCompute(cv2.cvtColor(im,cv2.COLOR_RGB2GRAY),None) for im in training]
m=mujoco.MjModel.from_xml_path(str(a.scene/'scene.xml'));d=mujoco.MjData(m);cid=m.camera(a.camera).id;results=[];tiles=[]
def ssim(x,y):
 x=cv2.cvtColor(x,cv2.COLOR_RGB2GRAY).astype(float);y=cv2.cvtColor(y,cv2.COLOR_RGB2GRAY).astype(float)
 blur=lambda z:cv2.GaussianBlur(z,(11,11),1.5)
 ux,uy=blur(x),blur(y);vx,vy=blur(x*x)-ux*ux,blur(y*y)-uy*uy;cov=blur(x*y)-ux*uy
 return float(np.mean(((2*ux*uy+6.5025)*(2*cov+58.5225))/((ux*ux+uy*uy+6.5025)*(vx+vy+58.5225))))
with mujoco.Renderer(m,height=h,width=w) as renderer:
 for index in wanted:
  target=heldout[index];kt,dt=sift.detectAndCompute(cv2.cvtColor(target,cv2.COLOR_RGB2GRAY),None);candidates=[]
  for ti in np.argsort(abs(best_index-index))[:3]:
   k0,d0=features[ti];pairs=cv2.BFMatcher().knnMatch(d0,dt,k=2);matches=[x for x,y in pairs if x.distance<.72*y.distance];xyz=[];uv=[]
   left,top,nw,nh,ow,oh=raw['image_bounds'][ti]
   for match in matches:
    x,y=k0[match.queryIdx].pt;u=int(round(left+x*nw/ow));v=int(round(top+y*nh/oh))
    if 0<=u<g['points'].shape[2] and 0<=v<g['points'].shape[1] and g['valid'][ti,v,u]:xyz.append(g['points'][ti,v,u]+shift);uv.append(kt[match.trainIdx].pt)
   if len(xyz)<12:continue
   xyz=np.array(xyz,dtype='float64');uv=np.array(uv,dtype='float64');K=g['intrinsic'][ti].copy();K[0]*=ow/nw;K[1]*=oh/nh;K[0,2]-=left*ow/nw;K[1,2]-=top*oh/nh
   ok,rvec,tvec,inliers=cv2.solvePnPRansac(xyz,uv,K,None,iterationsCount=1000,reprojectionError=2.,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
   if ok and inliers is not None and len(inliers)>=12:candidates.append((len(inliers),ti,xyz,uv,K,rvec,tvec,inliers[:,0]))
  if not candidates:results.append(dict(video_frame=index,status='pose estimation failed'));continue
  _,ti,xyz,uv,K,rvec,tvec,inliers=max(candidates,key=lambda c:c[0]);rvec,tvec=cv2.solvePnPRefineLM(xyz[inliers],uv[inliers],K,None,rvec,tvec);R=cv2.Rodrigues(rvec)[0];center=(-R.T@tvec).ravel();pred=cv2.projectPoints(xyz[inliers],rvec,tvec,K,None)[0][:,0];error=np.linalg.norm(pred-uv[inliers],axis=1)
  m.cam_pos[cid]=center;m.cam_quat[cid]=Rotation.from_matrix(R.T@np.diag([1,-1,-1])).as_quat(scalar_first=True);m.cam_fovy[cid]=np.degrees(2*np.arctan(h/(2*K[1,1])));mujoco.mj_forward(m,d);renderer.update_scene(d,camera=cid);render=renderer.render().copy()
  Image.fromarray(target).save(out/f'{index:04d}_source.png');Image.fromarray(render).save(out/f'{index:04d}_render.png')
  renderer.enable_segmentation_rendering();renderer.update_scene(d,camera=cid);segmentation=renderer.render().copy();renderer.disable_segmentation_rendering();np.save(out/f'{index:04d}_segmentation.npy',segmentation)
  visible={}
  for geom_id,count in zip(*np.unique(segmentation[:,:,0],return_counts=True)):
   if 0<=geom_id<m.ngeom:
    name=m.geom(int(geom_id)).name or ''
    if not name.startswith('photo_'):visible[name]=int(count)
  difference=np.abs(target.astype(float)-render.astype(float));heat=cv2.applyColorMap(np.uint8(np.clip(difference.mean(-1)*3,0,255)),cv2.COLORMAP_INFERNO)[...,::-1]
  row=Image.new('RGB',(w*3,h+25),'#172326');draw=ImageDraw.Draw(row)
  for j,(image,label) in enumerate([(target,f'Actual video {index/fps:.2f} s'),(render,'Reconstructed held-out camera'),(heat,'Absolute RGB error x3')]):row.paste(Image.fromarray(image),(j*w,25));draw.text((j*w+5,5),label,fill='white')
  row.thumbnail((960,600));tiles.append(row)
  results.append(dict(video_frame=index,time_seconds=index/fps,status='rendered',pose_reference_training_frame=Path(str(raw['frames'][ti])).name,correspondences=len(xyz),pnp_inliers=len(inliers),median_reprojection_error_px=float(np.median(error)),mean_absolute_rgb_error_0_255=float(difference.mean()),gray_ssim=ssim(target,render),camera_center_provisional_m=center.tolist(),camera_rotation_cv=R.tolist(),intrinsic=K.tolist(),visible_primitive_pixels=visible))
height=sum(im.height for im in tiles);sheet=Image.new('RGB',(max(im.width for im in tiles),height),'#172326');y=0
for im in tiles:sheet.paste(im,(0,y));y+=im.height
sheet.save(a.scene/'heldout_comparison.jpg')
report=dict(scene_fingerprint=scene_fingerprint(a.scene)['sha256'],method='Unseen video frames; pose-only SIFT/3D PnP against learned input depth; no held-out pixels used as reconstruction texture',training_video_frame_indices=best_index.tolist(),results=results,limitations=['Review poses are estimated using learned geometry.','RGB/SSIM agreement is descriptive and does not verify metric scale or physical parameters.','Views are intermediate positions along the recorded path, not unseen backsides.','Input frames were matched back to decoded source frames by minimum thumbnail RGB error.'])
a.report.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
