"""Map-based skill selection and execution with explicit tested preconditions.

This agent uses simulator state/manifest data. It is not a learned visual agent.
"""
import argparse,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco,numpy as np
from scene_reconstruction.navigation import collision_grid,plan
from scene_reconstruction.provenance import scene_fingerprint
p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,default=Path('outputs/final_scene'));p.add_argument('--task',choices=['navigate','pick'],required=True);p.add_argument('--target',default='test_block');p.add_argument('--run-name',default='agent');p.add_argument('--device',default='cpu');p.add_argument('--video',action='store_true');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
mf=json.loads((a.scene/'manifest.json').read_text());identity=scene_fingerprint(a.scene)['sha256'];report=dict(agent='map-based skill router + A* approach + BFM task tracking',task=a.task,scene=str(a.scene),scene_fingerprint=identity,state_access='Scene manifest and simulator object/root/wrist state',steps=[],status='planning')
try:
 if a.task=='pick':
  target=next((b for b in mf['bodies'] if b['name']==a.target),None)
  if target is None or 'pick' not in target.get('affordances',[]):raise ValueError('Target has no declared pick affordance')
  profile=mf.get('benchmarks',{}).get('pick',{})
  if profile.get('target')!=a.target or 'controller_profile' not in profile:raise ValueError('No contact-validated Dex3 skill profile for this target; a declared affordance alone is insufficient')
  settings=profile['controller_profile'];goal=profile['approach'];block=profile['block']
  model=mujoco.MjModel.from_xml_path(str(a.scene/'scene.xml'));data=mujoco.MjData(model)
  for _ in range(800):mujoco.mj_step(model,data)
  grid=collision_grid(model,data,mf['dimensions'],radius=.32);path=plan(grid,mf['navigation']['start'],goal[:2]);length=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())
  actual=data.xpos[model.body(a.target).id]
  if np.linalg.norm(actual-np.array(block))>.01:raise ValueError('Target is not at the tested support pose; replan/revalidate before executing this grasp profile')
  report['steps']=[dict(skill='navigate',goal=goal[:2],path_length_m=length,waypoints_m=path.tolist()),dict(skill='settle',root_height_m=settings['settled_root_height_m']),dict(skill='reach_close_lift_hold',target=a.target,profile=settings)]
  command=[sys.executable,'scripts/run_pick.py','--scene',str(a.scene/'scene.xml'),'--run-name',a.run_name,'--device',a.device,'--navigate-first','--seconds',str(settings['grasp_seconds']),'--root-x',str(goal[0]),'--root-y',str(goal[1]),'--root-height',str(settings['robot_root_height_m']),'--settled-root-height',str(settings['settled_root_height_m']),'--block-x',str(block[0]),'--block-y',str(block[1]),'--wrist-forward-offset',str(settings['wrist_forward_offset_m']),'--wrist-height-offset',str(settings['wrist_height_offset_m']),'--wrist-lateral-offset',str(settings['wrist_lateral_offset_m'])]
  if settings['pose_servo']:command+=['--pose-servo']
  if settings['slow_grasp']:command+=['--slow-grasp']
  outcome=Path(f'reports/dex3_pick_{a.run_name}.json')
 else:
  command=[sys.executable,'scripts/run_scalebfm.py','--scene',str(a.scene/'scene.xml'),'--run-name',a.run_name,'--device',a.device,'--task','explore','--seconds','38','--speed','.12'];outcome=Path(f'reports/{a.run_name}_scalebfm_explore_mode0.json');report['steps']=[dict(skill='navigate',selection='Screen manifest viewpoints and choose the longest collision-free route at radius 0.32 m')]
 if a.video:command+=['--video']
 report['command']=command;report['status']='planned'
 if not a.dry_run:
  subprocess.run(command,check=True)
  result=json.loads(outcome.read_text());report['execution_report']=str(outcome);report['success']=result['success'];report['status']='succeeded' if result['success'] else 'failed'
  if scene_fingerprint(a.scene)['sha256']!=identity:raise RuntimeError('Scene assets changed during execution; evidence identity is invalid')
except Exception as error:
 report['status']='rejected' if report['status']=='planning' else 'error';report['reason']=str(error)
Path(f'reports/skill_agent_{a.run_name}.json').write_text(json.dumps(report,indent=2));print('SKILL_AGENT_RESULT '+json.dumps(report),flush=True)
if report['status'] in ['rejected','error','failed']:sys.exit(1)
