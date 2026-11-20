"""Re-execute selected trials against the delivered scene; no cached successes."""
import argparse,json,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--runtime',choices=['mujoco','isaac'],required=True);a=p.parse_args()
py=sys.executable;scene='outputs/final_scene';wrapper='scripts/run_isaac.sh';logs=Path('outputs/inspection/final_validation');logs.mkdir(exist_ok=True)
if a.runtime=='mujoco':
 jobs=[
 ('physics',[py,'scripts/validate_scene.py','--scene',scene,'--report','reports/final_scene_physics.json']),
 ('assets',[py,'scripts/audit_assets.py','--scene',scene,'--robot-usd','outputs/robots/g1_dex3_v4/g1_dex3.usd']),
 ('generic_export',[py,'skills/reconstruct-video-scene/assets/toolbox/scripts/validate_export.py','--scene',scene,'--report','reports/skill_final_scene_settling.json']),
 ('explore',[py,'scripts/run_skill_agent.py','--scene',scene,'--task','navigate','--run-name','final_explore','--device','cuda','--video']),
 ('pick',[py,'scripts/run_skill_agent.py','--scene',scene,'--task','pick','--run-name','final_pick','--device','cuda','--video']),
 ('walk',[py,'scripts/run_scalebfm.py','--scene',scene+'/scene.xml','--task','navigate','--seconds','12','--run-name','final_nav','--device','cuda']),
 ('squat',[py,'scripts/run_scalebfm.py','--scene',scene+'/scene.xml','--task','squat','--seconds','8','--run-name','final_squat','--device','cuda']),
 ('appearance',[py,'scripts/validate_heldout_views.py','--scene',scene,'--times','1','4','7','10','13','16','--report','reports/final_appearance_validation.json'])]
else:
 jobs=[
 ('physics',[wrapper,'scripts/isaaclab_smoke.py','--scene',scene+'/scene.usda','--report','reports/final_isaaclab_validation.json','--screenshot','--enable_cameras','--camera','source_016']),
 ('generic_export',[wrapper,'skills/reconstruct-video-scene/assets/toolbox/scripts/validate_isaac.py','--scene',scene+'/scene.usda','--report','reports/skill_native_isaac_validation.json']),
 ('pick',[wrapper,'scripts/run_pick_isaac.py','--scene',scene+'/scene.usda','--navigate-first','--plan','reports/skill_agent_final_plan.json','--run-name','final_sequence','--video','--enable_cameras']),
 ('walk',[wrapper,'scripts/run_scalebfm_isaac.py','--scene',scene+'/scene.usda','--task','navigate','--seconds','12','--run-name','final_native_nav']),
 ('squat',[wrapper,'scripts/run_scalebfm_isaac.py','--scene',scene+'/scene.usda','--task','squat','--seconds','8','--run-name','final_native_squat'])]
results=[]
for name,cmd in jobs:
 if a.runtime=='isaac':cmd+=['--headless','--fast-exit','--device','cuda:0']
 start=time.time();print('START',name,flush=True)
 try:
  with (logs/(a.runtime+'_'+name+'.log')).open('w') as f:rc=subprocess.run(['timeout','-k','10','600',*cmd],stdout=f,stderr=subprocess.STDOUT).returncode
 except Exception as e:rc=-1;print(repr(e),flush=True)
 results.append(dict(name=name,command=cmd,returncode=rc,elapsed_seconds=time.time()-start));print('DONE',name,rc,flush=True)
 (logs/(a.runtime+'_jobs.json')).write_text(json.dumps(results,indent=2))
sys.exit(0 if all(r['returncode']==0 for r in results) else 1)
