"""Record a genuine force-driven chair push and torque-driven cabinet opening."""
import argparse,json,os,sys
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('EGL_PLATFORM','surfaceless')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco,numpy as np
from PIL import Image,ImageDraw
import imageio.v2 as imageio
from scene_reconstruction.provenance import scene_fingerprint
p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,default=Path('outputs/reconstructed_scene'));p.add_argument('--seconds',type=float,default=5.);a=p.parse_args()
m=mujoco.MjModel.from_xml_path(str(a.scene/'scene.xml'));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
chair=m.body('near_green_chair').id;jid=m.joint('cabinet_door_hinge').id;dof=m.jnt_dofadr[jid];qadr=m.jnt_qposadr[jid]
initial=d.xpos[chair].copy();frames=[];samples=[]
with mujoco.Renderer(m,height=852,width=480) as renderer:
 for frame in range(int(a.seconds*30)):
  for _ in range(round(1/(30*m.opt.timestep))):
   d.xfrc_applied[chair,0]=40. if .75<d.time<1.25 else 0.
   d.qfrc_applied[dof]=-3. if d.time>1. else 0.
   mujoco.mj_step(m,d)
  camera='source_020' if frame<int(a.seconds*30)/2 else 'source_000'
  renderer.update_scene(d,camera=camera);rgb=renderer.render().copy();im=Image.fromarray(rgb);draw=ImageDraw.Draw(im);draw.rectangle((0,0,480,40),fill='#172326');draw.text((8,10),'Physical chair push / inferred cabinet hinge; no pose forcing',fill='white');frames.append(np.array(im));samples.append([float(d.time),*d.xpos[chair],float(d.qpos[qadr])])
 for i,name in [(0,'interaction_before'),(len(frames)//2-1,'chair_after'),(-1,'cabinet_after')]:Image.fromarray(frames[i]).save(a.scene/(name+'.png'))
imageio.mimwrite(a.scene/'interaction.mp4',frames,fps=30,macro_block_size=1)
report=dict(scene_fingerprint=scene_fingerprint(a.scene)['sha256'],chair_displacement_m=float(np.linalg.norm(d.xpos[chair]-initial)),cabinet_angle_rad=float(d.qpos[qadr]),chair_push=dict(force_n=40,duration_seconds=.5),door_torque_nm=-3.,object_pose_forcing=False,photo_meshes_attached_to_moving_bodies=True,cabinet_articulation='inferred; not observed opening in source video')
(a.scene/'interaction_validation.json').write_text(json.dumps(report,indent=2));np.savetxt(a.scene/'interaction_trajectory.csv',samples,delimiter=',',header='time,chair_x,chair_y,chair_z,cabinet_angle',comments='');print(json.dumps(report,indent=2),flush=True)
