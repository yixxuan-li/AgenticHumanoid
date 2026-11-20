"""Generic Isaac Lab scene settling/mass/rest-frame check; no grasp claim."""
import argparse,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True);p.add_argument('--steps',type=int,default=300);p.add_argument('--report',type=Path);p.add_argument('--fast-exit',action='store_true');AppLauncher.add_app_launcher_args(p);a=p.parse_args();launcher=AppLauncher(a);app=launcher.app
import numpy as np
from pxr import UsdPhysics
import omni.usd
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject,RigidObjectCfg
from isaacsim.core.utils.prims import create_prim
from scene_reconstruction.provenance import scene_fingerprint

def main():
 mf=json.loads((a.scene.parent/'manifest.json').read_text());dynamic={b['name']:b for b in mf['bodies'] if b['kind']!='fixed'};sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005,device=a.device));create_prim('/World/Reconstruction',usd_path=str(a.scene.resolve()));stage=omni.usd.get_context().get_stage()
 for prim in list(stage.Traverse()):
  if prim.IsA(UsdPhysics.Scene) and str(prim.GetPath()).startswith('/World/Reconstruction'):prim.SetActive(False)
 objects={n:RigidObject(RigidObjectCfg(prim_path='/World/Reconstruction/Scene/'+n,spawn=None)) for n in dynamic};sim.reset()
 for _ in range(a.steps):
  sim.step(render=False)
  for obj in objects.values():obj.update(.005)
 states={n:obj.data.root_pos_w[0].cpu().tolist() for n,obj in objects.items()};masses={n:float(obj.root_physx_view.get_masses().reshape(-1)[0]) for n,obj in objects.items()};drift={n:float(np.linalg.norm(np.array(states[n])-b['pos'])) for n,b in dynamic.items() if b['kind']=='free'};hinges={}
 for n,b in dynamic.items():
  if b['kind']=='hinge':
   q=objects[n].data.root_quat_w[0].cpu().numpy();hinges[n]=float((2*np.arctan2(q[3],q[0])-b['yaw']+np.pi)%(2*np.pi)-np.pi)
 checks=dict(finite_states=bool(np.isfinite(list(states.values())).all()),mass_consistency=all(abs(masses[n]-b['mass'])<1e-5 for n,b in dynamic.items()),free_body_drift_under_6cm=all(v<.06 for v in drift.values()),hinge_closed_rest_frames=all(abs(v)<.05 for v in hinges.values()));report=dict(runtime='Isaac Lab / PhysX',scene=str(a.scene),scene_fingerprint=scene_fingerprint(a.scene.parent)['sha256'],scale_status=mf['scale_status'],checks=checks,all_passed=all(checks.values()),states_m=states,masses_kg=masses,free_body_drift_m=drift,hinge_relative_rest_angles_rad=hinges,steps=a.steps,dt_seconds=.005,limitations=['Resting/structure checks do not validate task affordances or true metric scale.','Add support/drop, hinge travel and robot/task tests for the actual scene.'])
 (a.report or a.scene.parent/'isaac_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
 if not report['all_passed']:raise RuntimeError('PhysX checks failed')
try:main()
except BaseException:
 import traceback
 traceback.print_exc();sys.stdout.flush();sys.stderr.flush()
 if a.fast_exit:os._exit(1)
 app.close(wait_for_replicator=False);raise
else:
 if a.fast_exit:sys.stdout.flush();sys.stderr.flush();os._exit(0)
 app.close(wait_for_replicator=False)
