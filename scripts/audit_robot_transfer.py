"""Compare imported G1/Dex3 mass, rest kinematics and joint limits to MJCF."""
import argparse,json
from pathlib import Path
import mujoco,numpy as np
from pxr import Usd,UsdGeom,UsdPhysics
from scipy.spatial.transform import Rotation
p=argparse.ArgumentParser();p.add_argument('--usd',type=Path,default=Path('outputs/robots/g1_dex3_v4/g1_dex3.usd'));p.add_argument('--report',type=Path,default=Path('reports/robot_transfer_audit.json'));a=p.parse_args()
m=mujoco.MjModel.from_xml_path('vendor/ScaleBFM/ScaleBridge/scalebridge/data/robot/g1_29dof/g1_29dof_dex3.xml');d=mujoco.MjData(m);jid=np.where(m.jnt_type==mujoco.mjtJoint.mjJNT_FREE)[0][0];qa=m.jnt_qposadr[jid];d.qpos[qa:qa+7]=[0,0,0,1,0,0,0];mujoco.mj_forward(m,d);s=Usd.Stage.Open(str(a.usd));cache=UsdGeom.XformCache();links={prim.GetName():prim for prim in s.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)};joints={prim.GetName():UsdPhysics.RevoluteJoint(prim) for prim in s.Traverse() if prim.IsA(UsdPhysics.RevoluteJoint)};body_results=[];joint_results=[]
for bid in range(1,m.nbody):
 name=m.body(bid).name
 if m.body_mass[bid]<=0:continue
 prim=links.get(name)
 if prim is None:body_results.append(dict(name=name,error='Missing positive-mass body'));continue
 mass=float(UsdPhysics.MassAPI(prim).GetMassAttr().Get());transform=cache.GetLocalToWorldTransform(prim);pos=np.array(transform.ExtractTranslation());q=transform.ExtractRotationQuat();quat=np.r_[q.GetReal(),q.GetImaginary()];angle=float((Rotation.from_quat(quat,scalar_first=True).inv()*Rotation.from_quat(d.xquat[bid],scalar_first=True)).magnitude())
 body_results.append(dict(name=name,source_mass_kg=float(m.body_mass[bid]),usd_mass_kg=mass,mass_error_kg=abs(mass-m.body_mass[bid]),rest_position_error_m=float(np.linalg.norm(pos-d.xpos[bid])),rest_rotation_error_rad=angle))
for ji in range(m.njnt):
 if m.jnt_type[ji]!=mujoco.mjtJoint.mjJNT_HINGE:continue
 name=m.joint(ji).name;joint=joints.get(name)
 if joint is None:joint_results.append(dict(name=name,error='Missing revolute joint'));continue
 limits=np.radians([joint.GetLowerLimitAttr().Get(),joint.GetUpperLimitAttr().Get()]);q=joint.GetLocalRot1Attr().Get();quat=np.r_[q.GetReal(),q.GetImaginary()];axis=np.eye(3)[{'X':0,'Y':1,'Z':2}[str(joint.GetAxisAttr().Get())]];direction=Rotation.from_quat(quat,scalar_first=True).apply(axis)
 joint_results.append(dict(name=name,limit_error_rad=float(np.max(abs(limits-m.jnt_range[ji]))),signed_axis_alignment=float(direction@m.jnt_axis[ji])))
checks=dict(meter_units=UsdGeom.GetStageMetersPerUnit(s)==1.,all_massive_links_present=all('error' not in b for b in body_results),mass_consistency=all(b.get('mass_error_kg',float('inf'))<1e-5 for b in body_results),rest_position_consistency=all(b.get('rest_position_error_m',float('inf'))<1e-5 for b in body_results),rest_rotation_consistency=all(b.get('rest_rotation_error_rad',float('inf'))<1e-3 for b in body_results),all_43_joint_limits=bool(len(joint_results)==43 and all(j.get('limit_error_rad',float('inf'))<1e-5 for j in joint_results)),all_joint_axis_signs=all(j.get('signed_axis_alignment',0)>.9999 for j in joint_results))
r=dict(robot_asset=str(a.usd),checks=checks,all_passed=all(checks.values()),body_comparisons=body_results,joint_comparisons=joint_results,tolerances=dict(mass_kg=1e-5,position_m=1e-5,rotation_rad=1e-3,joint_limits_rad=1e-5),rotation_note='Importer rounds nearly identity link orientations; the largest observed difference is reported explicitly.',scope='Authored mass, zero-joint rest link transforms, joint ranges and signed child-frame axes; actual dynamics are tested separately');a.report.write_text(json.dumps(r,indent=2));print(json.dumps(dict(checks=checks,all_passed=r['all_passed']),indent=2))
