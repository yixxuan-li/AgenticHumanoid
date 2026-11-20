"""Native PhysX load, cube settling/drop and optional source-camera capture."""
import argparse,json,os,sys,xml.etree.ElementTree as ET
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True);p.add_argument('--report',type=Path,required=True);p.add_argument('--screenshot',type=Path);p.add_argument('--camera',default='source_006');p.add_argument('--fast-exit',action='store_true');AppLauncher.add_app_launcher_args(p);a=p.parse_args();launcher=AppLauncher(a);app=launcher.app
import numpy as np,torch,omni.usd
from scipy.spatial.transform import Rotation
from pxr import UsdPhysics
from isaacsim.core.utils.prims import create_prim
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject,RigidObjectCfg
from scene_reconstruction.provenance import scene_fingerprint

def main():
 mf=json.loads((a.scene.parent/'manifest.json').read_text());sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005,device=a.device));create_prim('/World/Atlas',usd_path=str(a.scene.resolve()));stage=omni.usd.get_context().get_stage()
 for prim in list(stage.Traverse()):
  if prim.IsA(UsdPhysics.Scene) and str(prim.GetPath()).startswith('/World/Atlas'):prim.SetActive(False)
 objects={b['name']:RigidObject(RigidObjectCfg(prim_path='/World/Atlas/Scene/'+b['name'],spawn=None)) for b in mf['bodies'] if b['kind']=='free'};camera=None
 if a.screenshot:
  from isaaclab.sensors import Camera,CameraCfg
  source=ET.parse(a.scene.with_suffix('.xml')).getroot().find(f"worldbody/camera[@name='{a.camera}']")
  if source is None:raise ValueError('Camera missing from matching MJCF')
  pos=np.fromstring(source.get('pos'),sep=' ');axes=np.fromstring(source.get('xyaxes'),sep=' ');orientation=Rotation.from_matrix(np.column_stack([axes[:3],axes[3:],np.cross(axes[:3],axes[3:])])).as_quat(scalar_first=True);focal=36.*852/480/(2*np.tan(np.deg2rad(float(source.get('fovy')))/2))
  cfg=CameraCfg(prim_path='/World/ReviewCamera',height=852,width=480,data_types=['rgb'],update_period=0.,spawn=sim_utils.PinholeCameraCfg(focal_length=float(focal),horizontal_aperture=36.,clipping_range=(.03,100.)));camera=Camera(cfg)
 sim.reset()
 if camera is not None:camera.set_world_poses(torch.tensor([pos.tolist()],dtype=torch.float32,device=sim.device),torch.tensor([orientation.tolist()],dtype=torch.float32,device=sim.device),convention='opengl')
 for _ in range(300):
  sim.step(render=camera is not None)
  for obj in objects.values():obj.update(.005)
  if camera is not None:camera.update(.005)
 resting={n:o.data.root_pos_w[0].cpu().tolist() for n,o in objects.items()};masses={n:float(o.root_physx_view.get_masses().reshape(-1)[0]) for n,o in objects.items()};cube=objects['test_block'];baseline=cube.data.root_pos_w[0].clone();pose=cube.data.root_state_w[:,:7].clone();pose[:,2]+=.5;cube.write_root_pose_to_sim(pose);cube.write_root_velocity_to_sim(torch.zeros((1,6),device=sim.device));minimum_z=float('inf')
 for _ in range(400):
  sim.step(render=camera is not None);cube.update(.005);minimum_z=min(minimum_z,float(cube.data.root_pos_w[0,2]))
  if camera is not None:camera.update(.005)
 final=cube.data.root_pos_w[0].cpu().tolist();expected=float(next(b for b in mf['bodies'] if b['name']=='test_block')['parts'][0]['size'][2])/2;checks=dict(finite_states=bool(np.isfinite(list(resting.values())).all()),assigned_masses=all(abs(masses[b['name']]-b['mass'])<1e-5 for b in mf['bodies'] if b['kind']=='free'),cube_rest_support=abs(resting['test_block'][2]-expected)<.015,cube_drop_support=abs(final[2]-expected)<.015,cube_no_floor_tunneling=minimum_z>-.015)
 report=dict(runtime='Isaac Lab 0.47.2 / Isaac Sim 5.1 / PhysX',scene_fingerprint=scene_fingerprint(a.scene.parent)['sha256'],checks=checks,all_passed=all(checks.values()),settled_positions_m=resting,masses_kg=masses,drop=dict(height_above_settled_m=.5,final_position_m=final,expected_center_height_m=expected,minimum_center_height_m=minimum_z),scale_status=mf['scale_status'],limitations=['Static scene loading and benchmark cube tests do not validate articulated furniture or true metric scale.','Collision boxes and continuous floor are approximations.']);a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2))
 if camera is not None:
  from PIL import Image
  a.screenshot.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(camera.data.output['rgb'][0,:,:,:3].cpu().numpy()).save(a.screenshot)
 print(json.dumps(dict(checks=checks,all_passed=report['all_passed'])),flush=True)
 if not report['all_passed']:raise RuntimeError('World Labs PhysX validation failed')
try:main()
except BaseException:
 import traceback
 traceback.print_exc();sys.stdout.flush();sys.stderr.flush()
 if a.fast_exit:os._exit(1)
 app.close(wait_for_replicator=False);raise
else:
 if a.fast_exit:sys.stdout.flush();sys.stderr.flush();os._exit(0)
 app.close(wait_for_replicator=False)
