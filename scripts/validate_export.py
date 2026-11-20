"""Generic real settling/contact checks; task-specific affordances need more tests."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco,numpy as np
from pxr import Usd,UsdGeom,UsdPhysics
from scene_reconstruction.provenance import scene_fingerprint
p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True);p.add_argument('--seconds',type=float,default=4.);p.add_argument('--report',type=Path);a=p.parse_args()
manifest=json.loads((a.scene/'manifest.json').read_text());m=mujoco.MjModel.from_xml_path(str(a.scene/'scene.xml'));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
initial=[dict(a=m.geom(c.geom1).name,b=m.geom(c.geom2).name,penetration_m=float(-c.dist)) for c in d.contact if c.dist<-.005]
for _ in range(round(a.seconds/m.opt.timestep)):mujoco.mj_step(m,d)
free={b['name']:dict(displacement_m=float(np.linalg.norm(d.xpos[m.body(b['name']).id]-b['pos'])),mass_kg=float(m.body_mass[m.body(b['name']).id])) for b in manifest['bodies'] if b['kind']=='free'}
max_pen=float(max([-c.dist for c in d.contact]+[0.]))
stage=Usd.Stage.Open(str(a.scene/'scene.usda'));rigid=[str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI) and not p.GetCustomDataByKey('sceneJointAnchor')]
checks=dict(finite_state=bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()),no_initial_penetrations_over_5mm=not initial,settled_penetration_under_8mm=max_pen<.008,free_bodies_drift_under_6cm=all(b['displacement_m']<.06 for b in free.values()),positive_dynamic_masses=all(b['mass']>0 for b in manifest['bodies'] if b['kind']!='fixed'),mass_consistency=all(abs(m.body_mass[m.body(b['name']).id]-b['mass'])<1e-6 for b in manifest['bodies'] if b['mass']>0),usd_meters=UsdGeom.GetStageMetersPerUnit(stage)==1.,usd_z_up=str(UsdGeom.GetStageUpAxis(stage))=='Z',usd_rigid_count=len(rigid)==sum(b['mass']>0 for b in manifest['bodies']))
report=dict(scene=str(a.scene),scene_fingerprint=scene_fingerprint(a.scene)['sha256'],checks=checks,all_passed=all(checks.values()),scale_status=manifest['scale_status'],free_bodies=free,initial_penetrations=initial,max_settled_penetration_m=max_pen,simulated_seconds=float(d.time),limitations=['Settling is not task/affordance validation.','USD structure checks are not an Isaac Lab runtime test.','Scale and mass/friction provenance must be assessed separately.'])
(a.report or a.scene/'settling_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if not report['all_passed']:sys.exit(1)
