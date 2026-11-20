"""Real MuJoCo physics checks and USD structure checks; emits measured results."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import mujoco
from pxr import Usd,UsdGeom,UsdPhysics
from scene_reconstruction.provenance import scene_fingerprint


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,default=Path('outputs/scene'));p.add_argument('--report',type=Path,default=Path('reports/physics_validation.json'));args=p.parse_args()
    manifest=json.loads((args.scene/'manifest.json').read_text())
    scale=manifest.get('global_scale',1.)
    m=mujoco.MjModel.from_xml_path(str(args.scene/'scene.xml'));d=mujoco.MjData(m)
    mujoco.mj_forward(m,d)
    initial_penetration=[dict(a=m.geom(c.geom1).name,b=m.geom(c.geom2).name,depth_m=float(-c.dist)) for c in d.contact if c.dist<-.005]
    for _ in range(3000):mujoco.mj_step(m,d)
    max_pen=max([max(0.,-c.dist) for c in d.contact],default=0.)
    settled={}
    for body in manifest['bodies']:
        if body['kind']!='free':continue
        bid=m.body(body['name']).id
        joint=m.joint(body['name']+'_free').id;adr=m.jnt_dofadr[joint]
        settled[body['name']]=dict(position_m=d.xpos[bid].tolist(),quaternion_wxyz=d.xquat[bid].tolist(),
            linear_speed_mps=float(np.linalg.norm(d.qvel[adr:adr+3])),
            displacement_m=float(np.linalg.norm(d.xpos[bid]-body['pos'])),mass_kg=float(m.body_mass[bid]))
    baseline=d.qpos.copy()
    # Drop a genuinely free object onto the physical table; no weld, teleport or
    # controller is involved during the trajectory.
    joint=m.joint('test_block_free').id;qa=m.jnt_qposadr[joint];va=m.jnt_dofadr[joint]
    block=next(b for b in manifest['bodies'] if b['name']=='test_block')
    d.qpos[qa:qa+3]=[*block['pos'][:2],manifest['desk_height']+.55];d.qpos[qa+3:qa+7]=[1,0,0,0];d.qvel[:]=0
    for _ in range(1500):mujoco.mj_step(m,d)
    drop_z=float(d.qpos[qa+2]);expected=manifest['desk_height']+block['parts'][0]['size'][2]/2
    drop_error=abs(drop_z-expected)
    # Sliding/friction behavior: apply a short horizontal force and release.
    start=d.qpos[qa:qa+3].copy();bid=m.body('test_block').id
    table=next(b for b in manifest['bodies'] if b['name']=='central_table_0');direction=np.array(table['pos'][:2])-start[:2];direction/=np.linalg.norm(direction)
    for _ in range(80):
        d.xfrc_applied[bid,:2]=direction*.8*scale**3;mujoco.mj_step(m,d)
    d.xfrc_applied[:]=0
    for _ in range(500):mujoco.mj_step(m,d)
    push_distance=float(np.linalg.norm(d.qpos[qa:qa+2]-start[:2]))
    push_final_speed=float(np.linalg.norm(d.qvel[va:va+3]))
    push_final_z=float(d.qpos[qa+2])
    # Cabinet limit under external torque, separately from the prop test.
    d.qpos[:]=baseline;d.qvel[:]=0;jid=m.joint('cabinet_door_hinge').id;ja=m.jnt_qposadr[jid];jv=m.jnt_dofadr[jid]
    for _ in range(1500):
        d.qfrc_applied[jv]=-3.;mujoco.mj_step(m,d)
    hinge_position=float(d.qpos[ja]);hinge_limit=m.jnt_range[jid].tolist();d.qfrc_applied[:]=0
    stage=Usd.Stage.Open(str(args.scene/'scene.usda'))
    rigid=[str(prim.GetPath()) for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI) and not prim.GetCustomDataByKey('sceneJointAnchor')]
    collisions=[str(prim.GetPath()) for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)]
    mass_match={b['name']:abs(float(m.body_mass[m.body(b['name']).id])-b['mass'])<1e-6 for b in manifest['bodies'] if b['mass']}
    checks=dict(finite_state=bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()),
        no_initial_deep_penetration=len(initial_penetration)==0,
        furniture_stays_in_place=all(v['displacement_m']<.06 for v in settled.values()),
        mass_consistency=all(mass_match.values()),drop_on_table=drop_error<.004,
        friction_stops_prop=push_final_speed<.02,
        friction_probe_remains_on_table=abs(push_final_z-expected)<.004,
        push_moves_free_prop=push_distance>.005,
        hinge_limits=hinge_limit[0]-.03<=hinge_position<=hinge_limit[1]+.03,
        settled_penetration=max_pen<.008,
        usd_metric_units=UsdGeom.GetStageMetersPerUnit(stage)==1.,
        usd_z_up=str(UsdGeom.GetStageUpAxis(stage))=='Z',
        usd_rigid_body_count=len(rigid)==sum(b['mass']>0 for b in manifest['bodies']))
    report=dict(checks=checks,all_passed=all(checks.values()),scale_status=manifest['scale_status'],scene=str(args.scene),scene_fingerprint=scene_fingerprint(args.scene)['sha256'],
        model=dict(bodies=m.nbody,geometries=m.ngeom,dofs=m.nv,dt_seconds=m.opt.timestep),
        initial_penetrations_over_5mm=initial_penetration,settled_max_penetration_m=float(max_pen),settled_bodies=settled,
        drop=dict(final_z_m=drop_z,expected_z_m=expected,error_m=drop_error),push_displacement_m=push_distance,push_final_speed_mps=push_final_speed,push_final_z_m=push_final_z,push_direction_xy=direction.tolist(),
        cabinet_hinge=dict(angle_rad=hinge_position,limits_rad=hinge_limit),
        usd=dict(dynamic_rigid_bodies=len(rigid),kinematic_joint_anchors=sum(bool(p.GetCustomDataByKey('sceneJointAnchor')) for p in stage.Traverse()),collision_shapes=len(collisions)),
        limitations=['Metric dimensions are provisional until a measurement is supplied.','Mass and friction are assumed.','Scripted probes are not humanoid policy execution.','USD structure validation is not an Isaac Lab runtime test.'])
    args.report.parent.mkdir(exist_ok=True,parents=True);args.report.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ['settled_bodies','initial_penetrations_over_5mm']},indent=2))
    if not all(checks.values()):sys.exit(1)


if __name__=='__main__':main()
