import os,json,shutil,subprocess,tempfile,sys
from pathlib import Path
import numpy as np,trimesh
from PIL import Image
repo=Path(__file__).resolve().parents[1];python=Path(sys.executable);results={}
with tempfile.TemporaryDirectory(prefix='worldlabs-skill-') as tmp:
 root=Path(tmp)/'standalone';shutil.copytree(repo/'skills/worldlabs-scene-reconstruction/assets/toolbox',root)
 def run(*args,ok=True,env=None):
  p=subprocess.run([str(python),*map(str,args)],cwd=root,text=True,capture_output=True,env=env)
  if (p.returncode==0)!=ok:raise RuntimeError(p.stdout+p.stderr)
  return p
 workspace=root/'outputs/worldlabs';workspace.mkdir(parents=True)
 source=root/'frames';source.mkdir()
 for i in range(1,7):
  Image.fromarray(np.random.default_rng(i).integers(0,256,(64,64,3),dtype=np.uint8)).save(source/f'frame_{i:04d}.jpg')
 run('scripts/select_worldlabs_frames.py','--frames',source,'--workspace',workspace,'--count',3)
 selected=json.loads((workspace/'selected_frames.json').read_text());assert len(selected)==3 and [x['source_extracted_index'] for x in selected]==sorted(x['source_extracted_index'] for x in selected)
 results['portable_frame_selection']=True
 # Fake transport exercises resumability without credentials, uploads or paid API calls.
 binary=root/'bin';binary.mkdir();fake=binary/'worldlabs';fake.write_text('''#!/usr/bin/env python3
import json,sys,pathlib
args=sys.argv[1:]
with pathlib.Path('calls.jsonl').open('a') as f:f.write(json.dumps(args)+'\\n')
print(json.dumps({'id':'asset_'+pathlib.Path(args[2]).stem} if args[:2]==['assets','upload'] else {'id':'offline-operation','done':False,'metadata':{'state':'QUEUED'}}))
''');fake.chmod(0o700);env=os.environ.copy();env['PATH']=str(binary)+os.pathsep+env['PATH'];env.pop('WLT_API_KEY',None)
 run('scripts/worldlabs_scene.py','pose','--workspace',workspace,'--resolution',640,640,env=env)
 run('scripts/worldlabs_scene.py','pose','--workspace',workspace,'--resolution',640,640,env=env)
 calls=[json.loads(x) for x in (root/'calls.jsonl').read_text().splitlines()];assert sum(x[:2]==['tasks','run'] for x in calls)==1 and sum(x[:2]==['assets','upload'] for x in calls)==3
 run('scripts/worldlabs_scene.py','pose','--workspace',workspace,'--resolution',720,1280,env=env,ok=False)
 Image.new('RGB',(64,64),'red').save(selected[0]['path']);run('scripts/worldlabs_scene.py','pose','--workspace',workspace,'--resolution',640,640,env=env,ok=False)
 run('scripts/worldlabs_scene.py','status','--stage','../outside','--workspace',workspace,env=env,ok=False)
 results['offline_transport_reuses_job']=True;results['changed_request_and_frame_rejected']=True;results['stage_path_traversal_rejected']=True
 config=dict(floor_points=[[-2,.2,-2],[2,.2,-2],[2,.2,2],[-2,.2,2]],anchor=dict(points=[[0,.2,0],[0,2.2,0]],meters=1.,measured=True,description='Synthetic known-length fixture'))
 (root/'points.json').write_text(json.dumps(config));run('scripts/calibrate_worldlabs_scene.py','--config','points.json','--output','calibration.json');cal=json.loads((root/'calibration.json').read_text());assert abs(cal['scale']-.5)<1e-8 and abs(cal['floor_y']-.2)<1e-8
 config['floor_points']=[[0,0,0],[1,0,0],[2,0,0]];(root/'invalid.json').write_text(json.dumps(config));run('scripts/calibrate_worldlabs_scene.py','--config','invalid.json','--output','invalid_cal.json',ok=False)
 results['metric_anchor_and_floor_fit']=True;results['degenerate_floor_rejected']=True
 # Standalone mesh/collider export to a floor-only synthetic test environment.
 mesh=trimesh.Trimesh(vertices=[[-4,.2,-4],[4,.2,-4],[4,.2,4],[-4,.2,4]],faces=[[0,2,1],[0,3,2]],process=False)
 mesh.visual=trimesh.visual.texture.TextureVisuals(uv=[[0,0],[1,0],[1,1],[0,1]],material=trimesh.visual.material.PBRMaterial(baseColorTexture=Image.new('RGB',(32,32),'white')))
 mesh.export(root/'mesh.glb');collider=trimesh.creation.box(extents=[8,.12,8]);collider.apply_translation([0,.14,0]);collider.export(root/'colliders.glb')
 camera=dict(intrinsics=dict(width=640,height=640,fx=500,fy=500,cx=320,cy=320),extrinsics=dict(position=[0,2.2,0],quaternion=[0,0,0,1]))
 (workspace/'pose_operation.json').write_text(json.dumps(dict(response=dict(frames=[dict(camera=camera)]))))
 run('scripts/export_worldlabs_scene.py','--mesh','mesh.glb','--colliders','colliders.glb','--calibration','calibration.json','--workspace',workspace,'--output','outputs/scene')
 run('scripts/validate_worldlabs_mujoco.py','--scene','outputs/scene','--report','reports/physics.json');run('scripts/validate_export.py','--scene','outputs/scene');run('scripts/audit_assets.py','--scene','outputs/scene')
 run('scripts/rescale_scene.py','--scene','outputs/scene','--output','outputs/rescaled','--scale-factor',1.1)
 scaled=json.loads((root/'outputs/rescaled/manifest.json').read_text());assert scaled['desk_height'] is None and np.isclose(scaled['worldlabs_native_to_sim']['scale'],.55) and np.allclose(scaled['worldlabs_native_to_sim']['translation_xy'],np.array(json.loads((root/'outputs/scene/manifest.json').read_text())['worldlabs_native_to_sim']['translation_xy'])*1.1)
 run('scripts/validate_export.py','--scene','outputs/rescaled')
 results['portable_rescaling_without_desk_prior']=True
 results['portable_textured_mesh_export']=True;results['portable_mujoco_physics']=True;results['portable_usd_and_dependencies']=True
 print(json.dumps(dict(all_passed=True,checks=results,paid_api_calls=0),indent=2))
 (repo/'reports/worldlabs_skill_validation.json').write_text(json.dumps(dict(all_passed=True,checks=results,paid_api_calls=0),indent=2))
