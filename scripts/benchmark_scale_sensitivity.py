"""Hypothetical scene-scale sensitivity; never a metric calibration certificate."""
import argparse,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scene_reconstruction.provenance import scene_fingerprint
p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,default=Path('outputs/final_scene'));p.add_argument('--device',default='cuda');a=p.parse_args();root=Path('outputs/sensitivity');root.mkdir(exist_ok=True);results=[]
source_identity=scene_fingerprint(a.scene)['sha256']
for factor in [.95,1.05]:
 name=f'scale{round(factor*100):03d}';scene=root/name
 def run(command,log):
  with (root/log).open('w') as f:return subprocess.run(command,stdout=f,stderr=subprocess.STDOUT).returncode
 if not scene.exists():
  rc=run([sys.executable,'scripts/rescale_scene.py','--scene',str(a.scene),'--output',str(scene),'--scale-factor',str(factor)],name+'_build.log')
  if rc:results.append(dict(factor=factor,status='rescale failed'));continue
 if json.loads((scene/'manifest.json').read_text()).get('rescaled_from_sha256')!=source_identity:
  raise ValueError(f'{scene} belongs to an older base scene; preserve it elsewhere before rerunning')
 physics=Path(f'reports/{name}_physics.json');rc=run([sys.executable,'scripts/validate_scene.py','--scene',str(scene),'--report',str(physics)],name+'_physics.log');entry=dict(factor=factor,scene=str(scene),physics_passed=rc==0,physics_report=str(physics));print(json.dumps(entry),flush=True)
 plan_name=name+'_plan';rc=run([sys.executable,'scripts/run_skill_agent.py','--scene',str(scene),'--task','pick','--run-name',plan_name,'--dry-run'],name+'_plan.log');plan=json.loads(Path(f'reports/skill_agent_{plan_name}.json').read_text());entry['planning_status']=plan['status']
 if rc:entry.update(execution='not attempted',reason=plan.get('reason'))
 else:
  run_name=name+'_pick';rc=run([sys.executable,'scripts/run_skill_agent.py','--scene',str(scene),'--task','pick','--run-name',run_name,'--device',a.device],name+'_pick.log');entry.update(execution='attempted',pick_success=rc==0,pick_report=f'reports/dex3_pick_{run_name}.json')
 results.append(entry);print(json.dumps(entry),flush=True)
report=dict(base_scene_fingerprint=source_identity,variation='Chosen hypothetical multipliers, not measured scale or statistical confidence bounds',robot_scale='Published metric robot size and hand geometry remain fixed',results=results);Path('reports/scale_sensitivity.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
