"""Test assigned physics of the Atlas room; never certifies true room scale."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco,numpy as np
from scene_reconstruction.provenance import scene_fingerprint
p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,default=Path('outputs/worldlabs_scene'));p.add_argument('--report',type=Path,default=Path('reports/worldlabs_mujoco_validation.json'));a=p.parse_args();mf=json.loads((a.scene/'manifest.json').read_text());cube_spec=next(b for b in mf['bodies'] if b['name']=='test_block');expected=float(cube_spec['parts'][0]['size'][2])/2;assigned_mass=float(cube_spec['mass']);push_force=.8*assigned_mass/.08;m=mujoco.MjModel.from_xml_path(str(a.scene/'scene.xml'));d=mujoco.MjData(m);mujoco.mj_forward(m,d);initial=[float(c.dist) for c in d.contact if c.dist<-.005]
for _ in range(1000):mujoco.mj_step(m,d)
bid=m.body('test_block').id;jid=m.joint('test_block_free').id;qa=m.jnt_qposadr[jid];va=m.jnt_dofadr[jid];rest=d.xpos[bid].copy();baseline=d.qpos.copy();d.qpos[qa+2]+=.5;d.qvel[:]=0;mujoco.mj_forward(m,d);minimum=float('inf')
for _ in range(600):mujoco.mj_step(m,d);minimum=min(minimum,float(d.xpos[bid,2]))
final=d.xpos[bid].copy();drop_error=abs(final[2]-expected);d.qpos[:]=baseline;d.qvel[:]=0;mujoco.mj_forward(m,d);start=d.xpos[bid].copy();direction=np.array(mf['navigation']['start'])-start[:2];direction/=max(np.linalg.norm(direction),1e-6)
force_steps=int(np.ceil(.15/m.opt.timestep))
for _ in range(force_steps):d.xfrc_applied[bid,:2]=push_force*direction;mujoco.mj_step(m,d)
d.xfrc_applied[:]=0
for _ in range(600):mujoco.mj_step(m,d)
displacement=float(np.linalg.norm(d.xpos[bid,:2]-start[:2]));speed=float(np.linalg.norm(d.qvel[va:va+3]));penetration=max([max(0,-c.dist) for c in d.contact],default=0.);checks=dict(finite_state=bool(np.isfinite(d.qpos).all()),no_initial_penetrations_over_5mm=not initial,cube_positive_mass=abs(float(m.body_mass[bid])-assigned_mass)<1e-7,cube_rest_support=abs(rest[2]-expected)<.006,cube_drop_support=drop_error<.006,cube_no_floor_tunneling=minimum>-.015,cube_moves_under_horizontal_force=.015<displacement<.25,cube_friction_stops_motion=speed<.02,cube_stays_supported=abs(float(d.xpos[bid,2])-expected)<.006,settled_penetration_under_8mm=penetration<.008)
checks={k:bool(v) for k,v in checks.items()}
report=dict(scene_fingerprint=scene_fingerprint(a.scene)['sha256'],checks=checks,all_passed=all(checks.values()),initial_penetrations_m=initial,rest_position_m=rest.tolist(),drop=dict(final_position_m=final.tolist(),support_error_m=drop_error,minimum_center_height_m=minimum),push=dict(force_n=push_force,duration_seconds=force_steps*m.opt.timestep,displacement_m=displacement,final_speed_mps=speed),scale_status=mf['scale_status'],limitations=['The room is static; furniture articulation was not recovered.','Tests validate assigned parameters and proxy contacts, not true dimensions, mass or friction.','The red cube is an added benchmark object.']);a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2));print(json.dumps(dict(checks=checks,all_passed=report['all_passed']),indent=2))
if not report['all_passed']:raise RuntimeError('Atlas scene MuJoCo physics checks failed')
