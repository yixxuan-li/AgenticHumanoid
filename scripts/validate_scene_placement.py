"""Check native PhysX placement of three referenced scene copies, including hinges."""
import argparse,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,default=Path('outputs/worldlabs_scene/scene.usda'));p.add_argument('--report',type=Path,default=Path('reports/final_placement_validation.json'));p.add_argument('--fast-exit',action='store_true');AppLauncher.add_app_launcher_args(p);a=p.parse_args();launcher=AppLauncher(a);app=launcher.app
import numpy as np,omni.usd
from pxr import UsdGeom,UsdPhysics,Gf
from isaacsim.core.utils.prims import create_prim
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject,RigidObjectCfg
from scene_reconstruction.provenance import scene_fingerprint

def main():
 mf=json.loads((a.scene.parent/'manifest.json').read_text());dynamic={b['name']:b for b in mf['bodies'] if b['kind']!='fixed'};sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005,device=a.device));stage=omni.usd.get_context().get_stage();objects={};offsets={'origin':np.array([0.,0,0]),'shifted':np.array([8.,0,0]),'rotated':np.array([0.,8.,0])};yaws={'origin':0.,'shifted':0.,'rotated':np.pi/2}
 for label,offset in offsets.items():
  path='/World/'+label;yaw=yaws[label];create_prim(path,usd_path=str(a.scene.resolve()),translation=tuple(offset),orientation=(np.cos(yaw/2),0.,0.,np.sin(yaw/2)))
  for name in dynamic:objects[label,name]=RigidObject(RigidObjectCfg(prim_path=path+'/Scene/'+name,spawn=None))
 for prim in list(stage.Traverse()):
  if prim.IsA(UsdPhysics.Scene) and str(prim.GetPath()).startswith(tuple('/World/'+label for label in offsets)):prim.SetActive(False)
 sim.reset()
 for _ in range(300):
  sim.step(render=False)
  for obj in objects.values():obj.update(.005)
 results={}
 for (label,name),obj in objects.items():
  b=dynamic[name];yaw=yaws[label];rotation=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]]);pos=obj.data.root_pos_w[0].cpu().numpy();error=float(np.linalg.norm(pos-(rotation@np.array(b['pos'])+offsets[label])));entry=dict(position_error_m=error)
  if b['kind']=='hinge':
   q=obj.data.root_quat_w[0].cpu().numpy();entry['relative_angle_rad']=float((2*np.arctan2(q[3],q[0])-b['yaw']-yaw+np.pi)%(2*np.pi)-np.pi)
  results[label+'/'+name]=entry
 checks=dict(all_bodies_follow_scene_placement=all(r['position_error_m']<.06 for r in results.values()),hinges_preserve_closed_frame=all(abs(r.get('relative_angle_rad',0))<.05 for r in results.values()));report=dict(runtime='Isaac Lab / PhysX',scene_fingerprint=scene_fingerprint(a.scene.parent)['sha256'],checks=checks,all_passed=all(checks.values()),placements={k:dict(translation_m=v.tolist(),yaw_rad=yaws[k]) for k,v in offsets.items()},results=results,scope='Three referenced scene copies: origin, translated 8 m, and translated/rotated 90 degrees; verifies placement and scene-local hinge behavior, not large-batch training performance');a.report.write_text(json.dumps(report,indent=2));print('PLACEMENT_VALIDATION '+json.dumps(report),flush=True)
 if not report['all_passed']:raise RuntimeError('Translated scene placement check failed')
try:main()
except BaseException:
 import traceback
 traceback.print_exc();sys.stdout.flush();sys.stderr.flush()
 if a.fast_exit:os._exit(1)
 app.close(wait_for_replicator=False);raise
else:
 if a.fast_exit:sys.stdout.flush();sys.stderr.flush();os._exit(0)
 app.close(wait_for_replicator=False)
