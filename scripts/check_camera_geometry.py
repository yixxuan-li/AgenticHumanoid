"""Check camera/depth conventions against independently ray-cast MuJoCo surfaces."""
import os
import json
import sys
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('EGL_PLATFORM','surfaceless')
sys.path.insert(0,str(Path('vendor/vggt').resolve()))
import mujoco
import numpy as np
from vggt.utils.geometry import unproject_depth_map_to_point_map

m=mujoco.MjModel.from_xml_path('outputs/scene/scene.xml');d=mujoco.MjData(m);mujoco.mj_forward(m,d)
h,w=568,320;cid=m.camera('window_view').id
rotation=d.cam_xmat[cid].reshape(3,3);position=d.cam_xpos[cid]
R=np.diag([1,-1,-1])@rotation.T;translation=-R@position;extr=np.column_stack([R,translation])
f=h/(2*np.tan(np.deg2rad(m.cam_fovy[cid])/2));K=np.array([[f,0,w/2],[0,f,h/2],[0,0,1.]])
with mujoco.Renderer(m,height=h,width=w) as renderer:
    renderer.enable_depth_rendering();renderer.update_scene(d,camera='window_view');depth=renderer.render().copy()
cloud=unproject_depth_map_to_point_map(depth[None,:,:,None],extr[None],K[None])[0]
errors=[];rng=np.random.default_rng(1)
for y,x in rng.integers([30,30],[h-30,w-30],size=(100,2)):
    # Pixel centers are half a pixel from integer raster corners.
    local=np.linalg.inv(K)@np.array([x+.5,y+.5,1.]);ray=R.T@local;ray/=np.linalg.norm(ray)
    gid=np.array([-1],dtype=np.int32);distance=mujoco.mj_ray(m,d,position,ray,None,1,-1,gid)
    if distance<0:continue
    expected=position+ray*distance;errors.append(float(np.linalg.norm(cloud[y,x]-expected)))
report=dict(test='camera/depth backprojection compared with independent MuJoCo ray casts',samples=len(errors),median_surface_error_m=float(np.median(errors)),p95_surface_error_m=float(np.percentile(errors,95)),
    caveat='Raster edge pixels and the half-pixel sampling convention can differ. This verifies coordinate handling, not learned depth accuracy.')
report['passed']=report['median_surface_error_m']<.02 and report['p95_surface_error_m']<.10
Path('reports/camera_geometry_check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if not report['passed']:raise RuntimeError('Camera/depth convention check failed')
